"""
FCF Yield 팩터 추가 백테스트

레짐별 전략:
  Bull: 기존 전략 그대로 (수정전략_코스피_cap30%_top30_tx30bp_월간)
  Bear: F_PBR 3% + FCF_YIELD 2% (cap30%_손절율15%(고점) 기반)

실행: python analysis/backtest_fcf_bear.py
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.db import get_conn
from lib.factor_engine import score_stocks_from_strategy, code_to_module, clear_factor_cache

BULL_KEY = "수정전략_코스피_cap30%_top30_tx30bp_월간"
BEAR_KEY = "cap30%_손절율15%(고점)"


# ═══════════════════════════════════════════════════════
# AI 레짐 로드
# ═══════════════════════════════════════════════════════

def _load_ai_regime() -> dict:
    """AI 레짐 결과 파일 로드 → {YYYY-MM: "Bull"/"Bear"}."""
    _env = os.environ.get("AI_REGIME_RESULTS_PATH", "")
    ai_path = Path(_env) if _env else Path(__file__).parent / "regime_agent_results.json"
    if not ai_path.exists():
        print("[REGIME] AI 레짐 파일 없음 → 전 구간 Bull로 처리")
        return {}

    er_map = {}
    with open(ai_path, encoding="utf-8") as f:
        for r in json.load(f):
            ym = r.get("as_of", "")[:7]
            er_map[ym] = r.get("expected_return", 0)

    regime_map: dict[str, str] = {}
    prev = "Bull"
    for ym in sorted(er_map):
        er = er_map[ym]
        if prev == "Bull":
            regime_map[ym] = "Bear" if er <= -2 else "Bull"
        else:
            regime_map[ym] = "Bull" if er >= 1 else "Bear"
        prev = regime_map[ym]

    bull_cnt = sum(1 for v in regime_map.values() if v == "Bull")
    print(f"[REGIME] {len(regime_map)}개월 로드 — Bull {bull_cnt} / Bear {len(regime_map)-bull_cnt}")
    return regime_map


# ═══════════════════════════════════════════════════════
# FCF-Bear 백테스트
# ═══════════════════════════════════════════════════════

def run_fcf_bear_backtest() -> dict:
    """
    FCF-Bear 백테스트 실행.
    Bull: 기존 Bull 전략 / Bear: F_PBR 3% + FCF_YIELD 2%
    """
    from lib.data import load_strategy
    from config.settings import BACKTEST_CONFIG
    from step7_backtest import run_backtest, get_universe_stocks

    bull_data = load_strategy(BULL_KEY)
    bear_data = load_strategy(BEAR_KEY)
    bull_module = code_to_module(bull_data["code"])

    # Bear 모듈: F_PBR 5% → 3%, FCF_YIELD 2% 추가 (가중치 합 1.0 유지)
    bear_modified_code = bear_data["code"].replace('"F_PBR": .05,', '"F_PBR": .03,')
    if '"FCF_YIELD"' not in bear_modified_code:
        bear_modified_code = re.sub(
            r'(WEIGHTS_LARGE\s*=\s*\{[^}]*)\}',
            r'\1    "FCF_YIELD": .02,\n}',
            bear_modified_code, count=1, flags=re.DOTALL,
        )
    else:
        bear_modified_code = bear_modified_code.replace('"FCF_YIELD": 0,', '"FCF_YIELD": .02,')

    # SCORE_MAP에 FCF_YIELD → fcf_yield_score 매핑 추가
    if '"fcf_yield_score"' not in bear_modified_code:
        bear_modified_code = re.sub(
            r'(SCORE_MAP\s*=\s*\{[^}]*)\}',
            r'\1    "FCF_YIELD": "fcf_yield_score",\n}',
            bear_modified_code, count=1, flags=re.DOTALL,
        )

    # SCORING_RULES에 fcf_yield 점수화 규칙 추가 (높을수록 좋음 → rule2)
    if '"fcf_yield"' not in bear_modified_code:
        bear_modified_code = re.sub(
            r'(SCORING_RULES\s*=\s*\{[^}]*)\}',
            r'\1    "fcf_yield": "rule2",\n}',
            bear_modified_code, count=1, flags=re.DOTALL,
        )

    bear_module = code_to_module(bear_modified_code)

    bull_params = getattr(bull_module, "PARAMS", {})
    bear_params = getattr(bear_module, "PARAMS", {})
    top_n        = bull_params.get("top_n", 30)
    tx_cost_bp   = int((bull_params.get("tx_cost_bp", 30) + bear_params.get("tx_cost_bp", 30)) / 2)
    bull_cap     = bull_params.get("weight_cap_pct", 30)
    bear_cap     = bear_params.get("weight_cap_pct", 30)
    bear_sl      = bear_params.get("stop_loss_enabled", False)
    bear_sl_pct  = bear_params.get("stop_loss_pct", 15)
    bear_sl_mode = bear_params.get("stop_loss_mode", "sell")

    ai_regime_map = _load_ai_regime()

    from lib.factor_engine import prefetch_all_data
    from step7_backtest import get_db as _get_db
    _pf_conn = _get_db()
    prefetch_all_data(_pf_conn)
    _pf_conn.close()

    orig = {k: BACKTEST_CONFIG.get(k) for k in [
        "top_n_stocks", "transaction_cost_bp", "weight_cap_pct",
        "stop_loss_enabled", "stop_loss_pct", "stop_loss_mode",
        "universe", "rebal_type",
    ]}
    try:
        BACKTEST_CONFIG["top_n_stocks"]        = top_n
        BACKTEST_CONFIG["transaction_cost_bp"] = tx_cost_bp
        BACKTEST_CONFIG["universe"]            = "KOSPI"
        BACKTEST_CONFIG["rebal_type"]          = "monthly"
        BACKTEST_CONFIG["weight_cap_pct"]      = bull_cap
        BACKTEST_CONFIG["stop_loss_enabled"]   = False

        def selector(conn, calc_date, _top_n):
            regime = ai_regime_map.get(calc_date[:7], "Bull")
            universe_set = get_universe_stocks(conn, calc_date, "monthly", 0)

            if regime == "Bull":
                BACKTEST_CONFIG["weight_cap_pct"]    = bull_cap
                BACKTEST_CONFIG["stop_loss_enabled"] = False
                cands = score_stocks_from_strategy(conn, calc_date, bull_module)
            else:
                BACKTEST_CONFIG["weight_cap_pct"]    = bear_cap
                BACKTEST_CONFIG["stop_loss_enabled"] = bear_sl
                BACKTEST_CONFIG["stop_loss_pct"]     = bear_sl_pct
                BACKTEST_CONFIG["stop_loss_mode"]    = bear_sl_mode
                cands = score_stocks_from_strategy(conn, calc_date, bear_module)

            return [(c, s) for c, s in cands if c in universe_set][:_top_n]

        result = run_backtest("FCF_BEAR", stock_selector=selector, rebal_type="monthly")
        if result:
            result["strategy"] = "FCF-Bear"
        return result or {}

    finally:
        for k, v in orig.items():
            if v is not None:
                BACKTEST_CONFIG[k] = v


# ═══════════════════════════════════════════════════════
# 수익률 그래프
# ═══════════════════════════════════════════════════════

def plot_cumulative_returns(labeled_results: list[tuple[str, dict | None]]):
    """누적 수익률 그래프 저장 및 표시."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    _, ax = plt.subplots(figsize=(13, 6))

    styles = [
        ("Regime Combo (Bull+Bear)", "steelblue",  "-",  2.0),
        ("FCF-Bear",                 "darkorange",  "-",  2.0),
        ("KODEX 200",                "dimgray",     ":",  1.4),
    ]
    style_map = {s[0]: s[1:] for s in styles}

    for label, r in labeled_results:
        if not r:
            continue
        dates  = r.get("rebalance_dates", [])
        values = r.get("portfolio_values", [])
        if len(dates) != len(values) or not dates:
            continue
        color, ls, lw = style_map.get(label, ("black", "-", 1.2))
        ax.plot(dates, values, label=label, color=color, linestyle=ls, linewidth=lw)

    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (base = 1)")
    ax.set_title("Strategy Performance Comparison")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fcf_bear_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"[CHART] Saved: {out_path}")
    plt.show()


# ═══════════════════════════════════════════════════════
# 결과 출력 헬퍼
# ═══════════════════════════════════════════════════════

def _fmt(r: dict | None, label: str):
    if not r:
        print(f"  {label:<36}  결과 없음")
        return
    print(
        f"  {label:<36}  "
        f"누적 {r['total_return']:>+7.1%}  "
        f"CAGR {r['cagr']:>+7.1%}  "
        f"MDD {r['mdd']:>6.1%}  "
        f"Sharpe {r['sharpe']:>5.2f}"
    )


# ═══════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════

def main():
    from lib.data import run_regime_combo_backtest
    from step7_backtest import get_rebalance_dates, calc_all_benchmarks

    conn = get_conn()

    # ── 1. 기준선: 기존 레짐 조합 ─────────────────────────────
    print("\n" + "=" * 60)
    print("[기준선] 기존 레짐 조합")
    print("=" * 60)
    baseline = run_regime_combo_backtest(
        bull_key=BULL_KEY, bear_key=BEAR_KEY,
        universe="KOSPI", rebal_type="monthly", regime_mode="ai",
    )
    clear_factor_cache()

    # ── 2. FCF-Bear ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[FCF-Bear] Bear: F_PBR 3% + FCF_YIELD 2%")
    print("=" * 60)
    result_fcf = run_fcf_bear_backtest()
    clear_factor_cache()

    # ── 3. 벤치마크 ────────────────────────────────────────────
    rebal_dates = get_rebalance_dates(conn, "monthly")
    bm = calc_all_benchmarks(conn, rebal_dates) if len(rebal_dates) >= 2 else {}
    conn.close()

    # ── 4. 비교 출력 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("결과 비교")
    print("=" * 60)
    print(f"  {'전략':<36}  {'누적':>8}  {'CAGR':>8}  {'MDD':>7}  {'Sharpe':>7}")
    print("  " + "-" * 70)
    _fmt(baseline.get("REGIME_COMBO"), "기존 레짐 조합 (Bull+Bear)")
    _fmt(result_fcf,                   "FCF-Bear 추가")
    bm_r = baseline.get("KOSPI") or bm.get("KOSPI")
    if bm_r:
        print("  " + "-" * 70)
        _fmt(bm_r, "KODEX 200 (벤치마크)")
    print()

    # ── 5. 그래프 ──────────────────────────────────────────────
    plot_cumulative_returns([
        ("Regime Combo (Bull+Bear)", baseline.get("REGIME_COMBO")),
        ("FCF-Bear",                 result_fcf),
        ("KODEX 200",                bm_r),
    ])


if __name__ == "__main__":
    main()
