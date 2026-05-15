"""
carry_over 3가지 동작 비교: 레짐조합_수정전략 기준
  A) carry off              — 매 리밸런싱마다 peak 리셋
  B) carry over (Bull skip) — 현재 production: Bear만 peak 누적, Bull에선 carry 얼어붙음
  C) carry over (자연 설계) — Bull/Bear 모두 peak 갱신

production 코드는 건드리지 않고 monkey-patch로 세 가지 동작을 비교.
Part 1: 누적/CAGR/MDD/Sharpe 비교 테이블
Part 2: 손절 이벤트 상세 (Part 1 실행 중에 캡처한 데이터 재활용 — 추가 연산 없음)

실행: python analysis/compare_carry_over.py
"""
import os, sys, json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np


def run_comparison():
    from lib.data import run_regime_combo_backtest
    import step7_backtest as bt

    bull_key = "수정전략_코스피_cap30%_top30_tx30bp_월간"
    bear_key = "cap30%_손절율15%(고점)"

    _orig_calc_plain = bt.calc_portfolio_return
    _orig_calc_sl = bt.calc_portfolio_return_with_stoploss

    # 모드별 손절 이벤트 + 수익률 캡처용 (Part 2 용도)
    events = {"off": {}, "bs": {}, "nat": {}}
    returns = {"off": {}, "bs": {}, "nat": {}}

    # ── 공통 실행 헬퍼 ──
    def _run_once(label):
        print("\n" + "=" * 60)
        print(f"  {label} 실행 중...")
        print("=" * 60)
        result = run_regime_combo_backtest(
            bull_key=bull_key, bear_key=bear_key,
            universe="KOSPI", rebal_type="monthly", regime_mode="ai",
        )
        return result.get("REGIME_COMBO", {})

    # ── A) carry off — 매번 빈 dict 강제 + 이벤트 캡처 ──
    def _calc_sl_off(conn, stocks, start_date, end_date, **kw):
        kw["carry_over"] = {}
        ret, evs, _ = _orig_calc_sl(conn, stocks, start_date, end_date, **kw)
        events["off"][start_date] = evs
        returns["off"][start_date] = ret
        return ret, evs, {}

    bt.calc_portfolio_return_with_stoploss = _calc_sl_off
    try:
        combo_off = _run_once("[1/3] A) carry off (매번 peak 리셋)")
    finally:
        bt.calc_portfolio_return_with_stoploss = _orig_calc_sl

    # ── B) carry over (Bull skip) — production 그대로 + 이벤트 캡처 ──
    def _calc_sl_bs(conn, stocks, start_date, end_date, **kw):
        ret, evs, new_carry = _orig_calc_sl(conn, stocks, start_date, end_date, **kw)
        events["bs"][start_date] = evs
        returns["bs"][start_date] = ret
        return ret, evs, new_carry

    bt.calc_portfolio_return_with_stoploss = _calc_sl_bs
    try:
        combo_bullskip = _run_once("[2/3] B) carry over with Bull skip (production 현재)")
    finally:
        bt.calc_portfolio_return_with_stoploss = _orig_calc_sl

    # ── C) carry over 자연 설계 — Bull/Bear 모두 갱신 + 이벤트 캡처 ──
    _shared_carry = [{}]

    def _calc_plain_nat(conn, stocks, start_date, end_date):
        # Bull 케이스: stop_loss_pct=9999로 호출 → 손절 발동 X, peak만 추적
        ret, _evs, new_carry = _orig_calc_sl(
            conn, stocks, start_date, end_date,
            stop_loss_pct=9999, stop_loss_mode="sell", stop_loss_basis="peak",
            carry_over=_shared_carry[0],
        )
        _shared_carry[0] = new_carry
        returns["nat"][start_date] = ret
        events["nat"][start_date] = []  # Bull = 손절 없음
        return ret

    def _calc_sl_nat(conn, stocks, start_date, end_date, **kw):
        kw["carry_over"] = _shared_carry[0]
        ret, evs, new_carry = _orig_calc_sl(conn, stocks, start_date, end_date, **kw)
        _shared_carry[0] = new_carry
        events["nat"][start_date] = evs
        returns["nat"][start_date] = ret
        return ret, evs, new_carry

    _shared_carry[0] = {}
    bt.calc_portfolio_return = _calc_plain_nat
    bt.calc_portfolio_return_with_stoploss = _calc_sl_nat
    try:
        combo_natural = _run_once("[3/3] C) carry over 자연 설계 (Bull/Bear 모두 갱신)")
    finally:
        bt.calc_portfolio_return = _orig_calc_plain
        bt.calc_portfolio_return_with_stoploss = _orig_calc_sl

    # ═════════════════════════════════════════════════════════
    # Part 1: 종합 비교 테이블
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  Part 1) carry_over 3가지 동작 비교")
    print("  전략: 레짐조합_수정전략_코스피_cap30%_top30_tx30bp_월간↑_cap30%_손절율15%(고점)↓")
    print("=" * 78)

    print(f"\n  {'지표':<16} {'A) carry off':>14} {'B) Bull skip':>16} {'C) 자연 설계':>16}")
    print(f"  {'-'*62}")
    for key, label, fmt in [
        ("total_return", "누적 수익률", ".1%"),
        ("cagr", "CAGR", ".1%"),
        ("mdd", "MDD", ".1%"),
        ("sharpe", "Sharpe", ".2f"),
        ("avg_monthly_return", "월평균 수익률", ".2%"),
        ("monthly_std", "월 표준편차", ".2%"),
    ]:
        v_off = combo_off.get(key, 0)
        v_bs = combo_bullskip.get(key, 0)
        v_nat = combo_natural.get(key, 0)
        print(f"  {label:<16} {format(v_off, fmt):>14} {format(v_bs, fmt):>16} {format(v_nat, fmt):>16}")

    # 월별 차이 상위 — 3개 모드 모두 비교 (spread = max-min)
    mr_off = combo_off.get("monthly_returns", [])
    mr_bs = combo_bullskip.get("monthly_returns", [])
    mr_nat = combo_natural.get("monthly_returns", [])
    dates = combo_natural.get("rebalance_dates", [])

    if mr_off and mr_bs and mr_nat and len(mr_off) == len(mr_bs) == len(mr_nat):
        rows = []
        for i, (ro, rb, rn) in enumerate(zip(mr_off, mr_bs, mr_nat)):
            d = dates[i] if i < len(dates) else f"period_{i}"
            spread = max(ro, rb, rn) - min(ro, rb, rn)
            rows.append({"date": d, "off": ro, "bs": rb, "nat": rn, "spread": spread})
        rows.sort(key=lambda x: x["spread"], reverse=True)

        print(f"\n  3개 모드 수익률 차이 상위 5개월 (spread = max-min):")
        print(f"  {'날짜':>12}  {'A(off)':>9}  {'B(skip)':>9}  {'C(자연)':>9}  {'C-B':>7}  {'C-A':>7}")
        for r in rows[:5]:
            print(f"  {r['date']:>12}  {r['off']*100:>+8.2f}%  {r['bs']*100:>+8.2f}%  {r['nat']*100:>+8.2f}%  {(r['nat']-r['bs'])*100:>+6.2f}%  {(r['nat']-r['off'])*100:>+6.2f}%")

        diff_months = [r for r in rows if r["spread"] > 0.001]
        print(f"\n  세 모드 중 하나라도 차이 발생한 월: {len(diff_months)}개 / 전체 {len(rows)}개")
        if diff_months:
            avg_cb = np.mean([r["nat"] - r["bs"] for r in diff_months])
            avg_ca = np.mean([r["nat"] - r["off"] for r in diff_months])
            avg_ba = np.mean([r["bs"] - r["off"] for r in diff_months])
            print(f"  차이 발생 월 평균: C-B={avg_cb*100:+.2f}%, C-A={avg_ca*100:+.2f}%, B-A={avg_ba*100:+.2f}%")

    # ═════════════════════════════════════════════════════════
    # Part 2: 손절 이벤트 상세 비교 (캡처한 events/returns 활용 — 추가 연산 X)
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  Part 2) 손절 이벤트 상세 비교 (3가지 모드)")
    print("=" * 78)

    # 종목코드 → 종목명 맵
    code_to_name_path = Path(__file__).parent.parent / "data" / "code_to_name.json"
    if code_to_name_path.exists():
        with open(code_to_name_path, encoding="utf-8") as f:
            code_to_name = json.load(f)
    else:
        code_to_name = {}
    def nm(code):
        return code_to_name.get(code, code)

    all_dates = sorted(set(events["off"]) | set(events["bs"]) | set(events["nat"]))
    detail_log = []
    for d in all_dates:
        ev_off = events["off"].get(d, [])
        ev_bs = events["bs"].get(d, [])
        ev_nat = events["nat"].get(d, [])
        codes_off = {e[0] for e in ev_off}
        codes_bs = {e[0] for e in ev_bs}
        codes_nat = {e[0] for e in ev_nat}
        if codes_off == codes_bs == codes_nat:
            continue  # 세 모드 모두 동일 → 스킵
        detail_log.append({
            "date": d,
            "ret_off": returns["off"].get(d),
            "ret_bs": returns["bs"].get(d),
            "ret_nat": returns["nat"].get(d),
            "codes_off": codes_off,
            "codes_bs": codes_bs,
            "codes_nat": codes_nat,
        })

    if not detail_log:
        print("\n  세 모드 모두 손절 이벤트 동일 (차이 없음)")
    else:
        print(f"\n  차이 발생: {len(detail_log)}개월\n")
        for d in detail_log:
            r_off = d["ret_off"]
            r_bs = d["ret_bs"]
            r_nat = d["ret_nat"]
            r_off_s = f"{r_off*100:+.2f}%" if r_off is not None else "  -  "
            r_bs_s = f"{r_bs*100:+.2f}%" if r_bs is not None else "  -  "
            r_nat_s = f"{r_nat*100:+.2f}%" if r_nat is not None else "  -  "
            print(f"  {d['date']}  A={r_off_s}  B={r_bs_s}  C={r_nat_s}")

            bs_vs_off = d["codes_bs"] - d["codes_off"]
            nat_vs_off = d["codes_nat"] - d["codes_off"]
            nat_vs_bs = d["codes_nat"] - d["codes_bs"]
            bs_vs_nat = d["codes_bs"] - d["codes_nat"]
            if bs_vs_off:
                print(f"    B 추가 손절 (vs A): {', '.join(nm(c) for c in bs_vs_off)}")
            if nat_vs_off:
                print(f"    C 추가 손절 (vs A): {', '.join(nm(c) for c in nat_vs_off)}")
            if nat_vs_bs:
                print(f"    ★ C(자연) 만 추가 손절 (vs B): {', '.join(nm(c) for c in nat_vs_bs)}")
            if bs_vs_nat:
                print(f"    B(skip) 만 손절 (C에선 안 발동): {', '.join(nm(c) for c in bs_vs_nat)}")
            print()

    # ═════════════════════════════════════════════════════════
    # JSON 저장 — Part 1 지표 + 월별 수익률 + Part 2 손절 이벤트
    # ═════════════════════════════════════════════════════════
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"compare_carry_over_{stamp}.json"

    metric_keys = ["total_return", "cagr", "mdd", "sharpe",
                   "avg_monthly_return", "monthly_std",
                   "monthly_returns", "rebalance_dates"]
    payload = {
        "meta": {
            "bull_key": bull_key,
            "bear_key": bear_key,
            "universe": "KOSPI",
            "rebal_type": "monthly",
            "regime_mode": "ai",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "part1_metrics": {
            "A_off": {k: combo_off.get(k) for k in metric_keys},
            "B_bullskip": {k: combo_bullskip.get(k) for k in metric_keys},
            "C_natural": {k: combo_natural.get(k) for k in metric_keys},
        },
        "part2_diff_events": [
            {
                "date": d["date"],
                "ret_off": d["ret_off"],
                "ret_bs": d["ret_bs"],
                "ret_nat": d["ret_nat"],
                "codes_off": sorted(d["codes_off"]),
                "codes_bs": sorted(d["codes_bs"]),
                "codes_nat": sorted(d["codes_nat"]),
                "names_off": [nm(c) for c in sorted(d["codes_off"])],
                "names_bs": [nm(c) for c in sorted(d["codes_bs"])],
                "names_nat": [nm(c) for c in sorted(d["codes_nat"])],
            }
            for d in detail_log
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 결과 저장: {out_path.relative_to(Path(__file__).parent.parent)}")


if __name__ == "__main__":
    run_comparison()
