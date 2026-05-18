"""
C vs D 방식 비교:
  C) carry over 자연 설계 — 손절 후 리밸런싱까지 현금 보유
  D) carry over + 즉시 대체 — 손절 후 value score 31위~부터 즉시 교체

C 방식은 compare_carry_over.py의 C 방식과 완전히 동일한 monkey-patch 경로 사용.
D 방식은 C와 동일한 infrastructure에 Bear 구간 즉시 대체 로직 추가.

성과: 누적수익률, CAGR, MDD, Sharpe, 월평균수익률, 월표준편차 + 누적수익률 그래프

실행: python analysis/compare_stoploss_replacement.py
"""
import sys, json
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
# D 방식 핵심 함수: 손절 발생 시 reserve에서 즉시 대체
# ═══════════════════════════════════════════════════════════════

def calc_portfolio_return_with_replacement(
    conn, stocks, start_date, end_date,
    reserve_stocks,
    stop_loss_pct=15,
    stop_loss_basis="peak",
    carry_over=None,
):
    """D 방식: 손절 발생 시 reserve_stocks에서 순서대로 즉시 대체 매수.

    Returns:
        (portfolio_return, stoploss_events, updated_carry_over)
    """
    from config.settings import BACKTEST_CONFIG
    from step7_backtest import _apply_mcap_cap

    if not stocks:
        return 0.0, [], {}
    if carry_over is None:
        carry_over = {}

    threshold = -stop_loss_pct / 100
    codes = [c for c, _ in stocks]
    reserve_deque = deque(c for c, _ in reserve_stocks)
    used_codes = set(codes)

    # ── 시작일 시총/가격 조회 ──
    ph = ",".join(["?"] * len(codes))
    start_rows = conn.execute(f"""
        SELECT dp.stock_code, dp.market_cap, dp.adj_close
        FROM daily_price dp
        INNER JOIN (
            SELECT stock_code, MIN(trade_date) as first_date
            FROM daily_price
            WHERE stock_code IN ({ph}) AND trade_date >= ?
            GROUP BY stock_code
        ) t ON dp.stock_code = t.stock_code AND dp.trade_date = t.first_date
    """, (*codes, start_date)).fetchall()
    start_map = {r[0]: (r[1], r[2]) for r in start_rows}

    # ── 가중치 계산 ──
    raw_mcaps = [start_map.get(c, (0, 0))[0] or 0 for c in codes]
    cap_pct = BACKTEST_CONFIG.get("weight_cap_pct", 10)
    weights = _apply_mcap_cap(raw_mcaps, cap=cap_pct / 100)
    weight_map = {c: weights[i] for i, c in enumerate(codes)}

    # ── 전체 기간 일별 가격 조회 (원래 종목 + 대기 종목 전부) ──
    all_codes = list(codes) + [c for c in reserve_deque]
    ph_all = ",".join(["?"] * len(all_codes))
    daily_rows = conn.execute(f"""
        SELECT stock_code, trade_date, adj_close
        FROM daily_price
        WHERE stock_code IN ({ph_all})
          AND trade_date >= ? AND trade_date <= ?
          AND adj_close > 0
        ORDER BY trade_date
    """, (*all_codes, start_date, end_date)).fetchall()

    daily_map: dict[str, list] = {}
    for code, dt, price in daily_rows:
        daily_map.setdefault(code, []).append((dt, price))

    stoploss_events = []
    replacement_events = []
    updated_carry_over = {}
    total_return = 0.0

    for _, init_code in enumerate(codes):
        w = weight_map.get(init_code, 0)
        if w <= 0:
            continue

        # ── 슬롯(가중치 w) 시뮬레이션: 손절 시 대체 종목으로 이어달리기 ──
        slot_return = 0.0
        current_code = init_code
        slot_start = start_date
        is_first = True

        while True:
            prices = [(dt, p) for dt, p in daily_map.get(current_code, [])
                      if dt >= slot_start and dt <= end_date]
            if not prices:
                break

            entry_price = prices[0][1]

            if is_first and current_code in carry_over:
                prev = carry_over[current_code]
                first_entry_price = prev["first_entry_price"]
                peak_price = max(prev["peak_price"], entry_price)
            else:
                first_entry_price = entry_price
                peak_price = entry_price
            is_first = False

            triggered = False
            trigger_date = None
            trigger_price = None

            for dt, price in prices:
                if dt <= slot_start:
                    continue
                if stop_loss_basis == "peak":
                    if price > peak_price:
                        peak_price = price
                    if (price - peak_price) / peak_price <= threshold:
                        trigger_date, trigger_price = dt, price
                        triggered = True
                        break
                else:
                    if (price - first_entry_price) / first_entry_price <= threshold:
                        trigger_date, trigger_price = dt, price
                        triggered = True
                        break

            if triggered:
                trigger_ret = (trigger_price - entry_price) / entry_price
                slot_return += trigger_ret * w
                stoploss_events.append((current_code, trigger_date, trigger_ret, w))

                next_code = None
                while reserve_deque:
                    candidate = reserve_deque.popleft()
                    if candidate in used_codes:
                        continue
                    if any(dt >= trigger_date for dt, _ in daily_map.get(candidate, [])):
                        next_code = candidate
                        used_codes.add(candidate)
                        break

                if next_code:
                    replacement_events.append({
                        "stopped_code": current_code,
                        "replacement_code": next_code,
                        "entry_date": trigger_date,
                        "slot_weight": round(w, 6),
                        "stopped_ret": round(trigger_ret, 6),
                    })
                    current_code = next_code
                    slot_start = trigger_date
                    continue
                else:
                    break
            else:
                end_price = prices[-1][1]
                ret = (end_price - entry_price) / entry_price
                slot_return += ret * w
                updated_carry_over[current_code] = {
                    "first_entry_price": first_entry_price,
                    "peak_price": peak_price,
                }
                break

        total_return += slot_return

    return total_return, stoploss_events, updated_carry_over, replacement_events


# ═══════════════════════════════════════════════════════════════
# C 방식: compare_carry_over.py의 C와 완전히 동일한 코드
# ═══════════════════════════════════════════════════════════════

def _run_method_c(bull_key, bear_key):
    from lib.data import run_regime_combo_backtest
    import step7_backtest as bt

    _orig_calc_plain = bt.calc_portfolio_return
    _orig_calc_sl = bt.calc_portfolio_return_with_stoploss
    _shared_carry = [{}]

    def _calc_plain_nat(conn, stocks, start_date, end_date):
        # Bull: stop_loss_pct=9999 → 손절 불가, peak만 추적
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
            bull_key=bull_key, bear_key=bear_key,
            universe="KOSPI", rebal_type="monthly", regime_mode="ai",
        )
    finally:
        bt.calc_portfolio_return = _orig_calc_plain
        bt.calc_portfolio_return_with_stoploss = _orig_calc_sl

    r = result.get("REGIME_COMBO", {})
    if "mdd" in r:
        r["mdd"] = -abs(r["mdd"])
    return r


# ═══════════════════════════════════════════════════════════════
# D 방식: C와 완전히 동일하되 Bear 손절 시 현금 대신 즉시 대체 매수
# ═══════════════════════════════════════════════════════════════

def _run_method_d(bull_key, bear_key):
    from lib.data import run_regime_combo_backtest
    import step7_backtest as bt
    import lib.factor_engine as _fe

    _orig_calc_plain = bt.calc_portfolio_return
    _orig_calc_sl = bt.calc_portfolio_return_with_stoploss
    _orig_score = _fe.score_stocks_from_strategy
    _shared_carry = [{}]
    _reserve_map = {}        # {calc_date: [(code, score), ...]} — universe 필터 전 전체 목록
    _replacement_log = []    # 전체 기간 대체 이벤트 누적

    def _capturing_score(conn, calc_date, module):
        full_list = _orig_score(conn, calc_date, module)
        _reserve_map[calc_date] = full_list
        return full_list

    def _calc_plain_nat(conn, stocks, start_date, end_date):
        # C와 완전히 동일한 Bull 로직
        ret, _evs, new_carry = _orig_calc_sl(
            conn, stocks, start_date, end_date,
            stop_loss_pct=9999, stop_loss_mode="sell", stop_loss_basis="peak",
            carry_over=_shared_carry[0],
        )
        _shared_carry[0] = new_carry
        return ret

    def _calc_sl_d(conn, stocks, start_date, end_date, **kw):
        # C와 다른 점: 손절 발생 시 현금 대신 다음 차순위 종목 즉시 매수
        stock_codes = {c for c, _ in stocks}
        from step7_backtest import get_universe_stocks as _get_univ
        universe_set = _get_univ(conn, start_date)
        all_candidates = _reserve_map.get(start_date, [])
        reserves = [(c, s) for c, s in all_candidates
                    if c in universe_set and c not in stock_codes]

        ret, evs, new_carry, rep_evs = calc_portfolio_return_with_replacement(
            conn, stocks, start_date, end_date,
            reserve_stocks=reserves,
            stop_loss_pct=kw.get("stop_loss_pct", 15),
            stop_loss_basis=kw.get("stop_loss_basis", "peak"),
            carry_over=_shared_carry[0],
        )
        for ev in rep_evs:
            _replacement_log.append({**ev, "period_start": start_date})
        _shared_carry[0] = new_carry
        return ret, evs, new_carry

    _shared_carry[0] = {}
    _fe.score_stocks_from_strategy = _capturing_score
    bt.calc_portfolio_return = _calc_plain_nat
    bt.calc_portfolio_return_with_stoploss = _calc_sl_d
    try:
        result = run_regime_combo_backtest(
            bull_key=bull_key, bear_key=bear_key,
            universe="KOSPI", rebal_type="monthly", regime_mode="ai",
        )
    finally:
        bt.calc_portfolio_return = _orig_calc_plain
        bt.calc_portfolio_return_with_stoploss = _orig_calc_sl
        _fe.score_stocks_from_strategy = _orig_score

    r = result.get("REGIME_COMBO", {})
    if "mdd" in r:
        r["mdd"] = -abs(r["mdd"])
    r["replacement_log"] = _replacement_log
    return r


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════

def run_comparison():
    bull_key = "수정전략_코스피_cap30%_top30_tx30bp_월간"
    bear_key = "cap30%_손절율15%(고점)"

    print("\n" + "=" * 60)
    print("  C) carry over 자연 설계 (손절 후 현금) 실행 중...")
    print("=" * 60)
    result_c = _run_method_c(bull_key, bear_key)

    print("\n" + "=" * 60)
    print("  D) carry over + 즉시 대체 실행 중...")
    print("=" * 60)
    result_d = _run_method_d(bull_key, bear_key)

    # ── 성과 테이블 ──
    print("\n" + "=" * 70)
    print("  C vs D 성과 비교")
    print(f"  Bull: {bull_key}")
    print(f"  Bear: {bear_key}")
    print("=" * 70)
    print(f"\n  {'지표':<16} {'C) 현금 보유':>16} {'D) 즉시 대체':>16}")
    print(f"  {'-' * 50}")

    for key, label, fmt in [
        ("total_return",       "누적 수익률",   ".1%"),
        ("cagr",               "CAGR",          ".1%"),
        ("mdd",                "MDD",            ".1%"),
        ("sharpe",             "Sharpe",         ".2f"),
        ("avg_monthly_return", "월평균 수익률",  ".2%"),
        ("monthly_std",        "월 표준편차",    ".2%"),
    ]:
        vc = result_c.get(key, 0)
        vd = result_d.get(key, 0)
        print(f"  {label:<16} {format(vc, fmt):>16} {format(vd, fmt):>16}")

    # ── 월별 차이 상위 ──
    mr_c  = result_c.get("monthly_returns", [])
    mr_d  = result_d.get("monthly_returns", [])
    dates = result_c.get("rebalance_dates", [])

    if mr_c and mr_d and len(mr_c) == len(mr_d):
        rows = []
        for i, (rc, rd) in enumerate(zip(mr_c, mr_d)):
            d = dates[i] if i < len(dates) else f"period_{i}"
            rows.append({"date": d, "c": rc, "d": rd, "diff": rd - rc})
        rows.sort(key=lambda x: abs(x["diff"]), reverse=True)

        print(f"\n  차이 큰 상위 5개월:")
        print(f"  {'날짜':>12}  {'C':>9}  {'D':>9}  {'D-C':>7}")
        for r in rows[:5]:
            print(f"  {r['date']:>12}  {r['c']*100:>+8.2f}%  {r['d']*100:>+8.2f}%  {r['diff']*100:>+6.2f}%")

        diff_months = [r for r in rows if abs(r["diff"]) > 0.001]
        print(f"\n  차이 발생 월: {len(diff_months)}개 / 전체 {len(rows)}개")
        if diff_months:
            avg_diff = np.mean([r["diff"] for r in diff_months])
            print(f"  차이 발생 월 평균 D-C: {avg_diff*100:+.2f}%")

    # ── 누적수익률 그래프 ──
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pv_c     = result_c.get("portfolio_values", [])
    pv_d     = result_d.get("portfolio_values", [])
    rb_dates = result_c.get("rebalance_dates", [])

    if pv_c and pv_d and rb_dates:
        _, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(pv_c)), pv_c, label="C) Hold Cash", color="#2196F3", linewidth=1.8)
        ax.plot(range(len(pv_d)), pv_d, label="D) Buy Immediately", color="#FF5722", linewidth=1.8)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        step = max(1, len(rb_dates) // 10)
        tick_pos = list(range(0, len(rb_dates), step))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([rb_dates[j][:7] for j in tick_pos], rotation=45, ha="right", fontsize=8)

        ax.set_title("C vs D Cumulative Return", fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative Return (1.0 = initial)")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        chart_path = out_dir / f"compare_CD_{stamp}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"\n  그래프 저장: {chart_path.relative_to(Path(__file__).parent.parent)}")

    # ── JSON 저장 ──
    json_path = out_dir / f"compare_CD_{stamp}.json"
    payload = {
        "meta": {
            "bull_key": bull_key,
            "bear_key": bear_key,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "C": {k: result_c.get(k) for k in
              ["total_return", "cagr", "mdd", "sharpe",
               "avg_monthly_return", "monthly_std",
               "monthly_returns", "rebalance_dates"]},
        "D": {k: result_d.get(k) for k in
              ["total_return", "cagr", "mdd", "sharpe",
               "avg_monthly_return", "monthly_std",
               "monthly_returns", "rebalance_dates",
               "replacement_log"]},
    }

    # ── 종목코드 → 종목명 매핑 ──
    code_to_name = {}
    name_map_path = Path(__file__).parent.parent / "data" / "code_to_name.json"
    if name_map_path.exists():
        with open(name_map_path, encoding="utf-8") as f:
            code_to_name = json.load(f)
    def _nm(code):
        return code_to_name.get(code, code)

    # ── D 대체 이벤트 요약 출력 ──
    rep_log = result_d.get("replacement_log", [])
    if rep_log:
        print(f"\n  [D] 대체 매수 이벤트 총 {len(rep_log)}건")
        print(f"  {'기간시작':>12}  {'손절종목':<14}  {'대체종목':<14}  {'진입일':>12}  {'손절수익률':>10}  {'비중':>7}")
        for ev in rep_log:
            print(f"  {ev['period_start']:>12}  {_nm(ev['stopped_code']):<14}  {_nm(ev['replacement_code']):<14}"
                  f"  {ev['entry_date']:>12}  {ev['stopped_ret']*100:>+9.1f}%  {ev['slot_weight']*100:>6.1f}%")
    else:
        print("\n  [D] 대체 매수 이벤트 없음")

    # replacement_log에 종목명 추가
    for ev in rep_log:
        ev["stopped_name"] = _nm(ev["stopped_code"])
        ev["replacement_name"] = _nm(ev["replacement_code"])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  JSON 저장: {json_path.relative_to(Path(__file__).parent.parent)}")


if __name__ == "__main__":
    run_comparison()
