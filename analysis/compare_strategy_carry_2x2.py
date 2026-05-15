"""
2전략 × 2 carry_over 모드 비교
  전략 1: 레짐조합_수정전략_코스피_cap30%_top30_tx30bp_월간 + 손절율15%(고점)
  전략 2: 레짐조합_가격모멘텀 + 손절율15%(고점)
  모드  A) carry off  — 매 리밸런싱마다 peak 리셋 (= "고점 매월 갱신")
  모드  C) carry over (자연 설계) — Bull/Bear 모두 peak 갱신

production 코드는 건드리지 않고 monkey-patch로 동작 비교.
실행: python analysis/compare_strategy_carry_2x2.py
"""
import os, sys, json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np


# ─── 비교 대상 정의 ───
STRATEGIES = [
    ("수정전략", "수정전략_코스피_cap30%_top30_tx30bp_월간"),
    ("가격모멘텀", "가격모멘텀"),
]
BEAR_KEY = "cap30%_손절율15%(고점)"
UNIVERSE = "KOSPI"
REBAL = "monthly"
REGIME_MODE = "ai"


def run_one(bull_key: str, mode: str):
    """단일 (전략, 모드) 백테스트 실행. mode: 'off' | 'nat'."""
    from lib.data import run_regime_combo_backtest
    import step7_backtest as bt

    _orig_calc_plain = bt.calc_portfolio_return
    _orig_calc_sl = bt.calc_portfolio_return_with_stoploss

    if mode == "off":
        # 매 호출마다 carry_over={} 강제 → peak 리셋
        def _calc_sl_off(conn, stocks, start_date, end_date, **kw):
            kw["carry_over"] = {}
            ret, evs, _ = _orig_calc_sl(conn, stocks, start_date, end_date, **kw)
            return ret, evs, {}
        bt.calc_portfolio_return_with_stoploss = _calc_sl_off
        try:
            result = run_regime_combo_backtest(
                bull_key=bull_key, bear_key=BEAR_KEY,
                universe=UNIVERSE, rebal_type=REBAL, regime_mode=REGIME_MODE,
            )
        finally:
            bt.calc_portfolio_return_with_stoploss = _orig_calc_sl

    else:  # mode == "nat"
        # Bull/Bear 양쪽 모두 peak 갱신 + carry 공유
        _shared_carry = [{}]

        def _calc_plain_nat(conn, stocks, start_date, end_date):
            # Bull 케이스: pct=9999로 호출 → 손절 발동 X, peak만 추적
            ret, _evs, new_carry = _orig_calc_sl(
                conn, stocks, start_date, end_date,
                stop_loss_pct=9999, stop_loss_mode="sell", stop_loss_basis="peak",
                carry_over=_shared_carry[0],
            )
            _shared_carry[0] = new_carry
            return ret

        def _calc_sl_nat(conn, stocks, start_date, end_date, **kw):
            kw["carry_over"] = _shared_carry[0]
            ret, evs, new_carry = _orig_calc_sl(conn, stocks, start_date, end_date, **kw)
            _shared_carry[0] = new_carry
            return ret, evs, new_carry

        _shared_carry[0] = {}
        bt.calc_portfolio_return = _calc_plain_nat
        bt.calc_portfolio_return_with_stoploss = _calc_sl_nat
        try:
            result = run_regime_combo_backtest(
                bull_key=bull_key, bear_key=BEAR_KEY,
                universe=UNIVERSE, rebal_type=REBAL, regime_mode=REGIME_MODE,
            )
        finally:
            bt.calc_portfolio_return = _orig_calc_plain
            bt.calc_portfolio_return_with_stoploss = _orig_calc_sl

    return result.get("REGIME_COMBO", {}) if result else {}


def main():
    grid = {}  # (strategy_label, mode) -> combo_result
    for strat_label, bull_key in STRATEGIES:
        for mode in ("off", "nat"):
            print("\n" + "=" * 70)
            print(f"  실행 중: {strat_label}  /  mode={mode}")
            print("=" * 70)
            grid[(strat_label, mode)] = run_one(bull_key, mode)

    # ═════════════════════════════════════════════════════════
    # Part 1: 종합 비교 테이블 (2전략 × 2모드)
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 86)
    print(f"  결과: 2전략 × 2 carry 모드  (bear={BEAR_KEY})")
    print("=" * 86)

    cols = [(s, m) for s, _ in STRATEGIES for m in ("off", "nat")]
    header = f"  {'지표':<14}"
    for strat_label, mode in cols:
        header += f"  {strat_label[:6]}/{mode:<3}".rjust(16)
    print(header)
    print("  " + "-" * (14 + 16 * len(cols)))

    metric_keys = [
        ("total_return", "누적 수익률", ".1%"),
        ("cagr",         "CAGR",        ".1%"),
        ("mdd",          "MDD",         ".1%"),
        ("sharpe",       "Sharpe",      ".2f"),
        ("avg_monthly_return", "월평균", ".2%"),
        ("monthly_std",  "월 표준편차", ".2%"),
    ]
    for key, label, fmt in metric_keys:
        row = f"  {label:<14}"
        for col in cols:
            v = grid[col].get(key, 0)
            row += f"  {format(v, fmt):>14}"
        print(row)

    # ─── 대비 분석 (carry 효과 + 전략 우위) ───
    print("\n  ▸ carry 효과 (nat − off, CAGR 기준)")
    for strat_label, _ in STRATEGIES:
        off = grid[(strat_label, "off")].get("cagr", 0)
        nat = grid[(strat_label, "nat")].get("cagr", 0)
        diff = nat - off
        print(f"    {strat_label:<10}  off={off*100:+.2f}%  nat={nat*100:+.2f}%  diff={diff*100:+.2f}%p")

    print("\n  ▸ 전략 우위 (수정전략 − 가격모멘텀, CAGR 기준)")
    for mode in ("off", "nat"):
        a = grid[("수정전략", mode)].get("cagr", 0)
        b = grid[("가격모멘텀", mode)].get("cagr", 0)
        print(f"    mode={mode:<3}  수정={a*100:+.2f}%  모멘텀={b*100:+.2f}%  diff={(a-b)*100:+.2f}%p")

    # ═════════════════════════════════════════════════════════
    # JSON 저장
    # ═════════════════════════════════════════════════════════
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"compare_strategy_carry_2x2_{stamp}.json"

    save_keys = ["total_return", "cagr", "mdd", "sharpe",
                 "avg_monthly_return", "monthly_std",
                 "monthly_returns", "rebalance_dates"]
    payload = {
        "meta": {
            "strategies": [{"label": s, "bull_key": k} for s, k in STRATEGIES],
            "bear_key": BEAR_KEY,
            "universe": UNIVERSE,
            "rebal_type": REBAL,
            "regime_mode": REGIME_MODE,
            "modes": {
                "off": "carry off (매 리밸런싱마다 peak 리셋)",
                "nat": "carry over 자연 설계 (Bull/Bear 모두 peak 갱신)",
            },
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "results": {
            strat_label: {
                mode: {k: grid[(strat_label, mode)].get(k) for k in save_keys}
                for mode in ("off", "nat")
            }
            for strat_label, _ in STRATEGIES
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 결과 저장: {out_path.relative_to(Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
