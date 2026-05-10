"""
손절 발동 빈도 비교: 매입가 15% vs 고점 대비 15%
- 리밸런싱 기간별 손절 발동 종목 수, 비율, 확정 수익률 분포
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from step7_backtest import (
    get_db, get_rebalance_dates, calc_portfolio_return_with_stoploss,
    _apply_mcap_cap, BACKTEST_CONFIG,
)
from lib.data import run_strategy_backtest
from lib.factor_engine import DEFAULT_STRATEGY_CODE, prefetch_all_data, clear_prefetch_cache

# ── 공통 설정 ──
UNIVERSE = "KOSPI"
REBAL_TYPE = "monthly"
TOP_N = 30
CAP_PCT = 30

configs = [
    {"label": "매입가 15%", "basis": "entry", "pct": 15},
    {"label": "고점 대비 15%", "basis": "peak", "pct": 15},
]

# ── 백테스트 + 손절 이벤트 수집 ──
conn = get_db()
rebal_dates = get_rebalance_dates(conn, REBAL_TYPE)

# stock_selector 준비
from lib.factor_engine import score_stocks_from_strategy, code_to_module
from step7_backtest import get_universe_stocks
strategy_module = code_to_module(DEFAULT_STRATEGY_CODE)

prefetch_all_data(conn)

def get_stocks(calc_date):
    universe_set = get_universe_stocks(conn, calc_date)
    candidates = score_stocks_from_strategy(conn, calc_date, strategy_module)
    filtered = [(c, s) for c, s in candidates if c in universe_set][:TOP_N]
    return filtered

print("=" * 80)
print("손절 발동 빈도 비교: 매입가 15% vs 고점 대비 15%")
print(f"유니버스: {UNIVERSE} | 리밸런싱: {REBAL_TYPE} | Top {TOP_N} | Cap {CAP_PCT}%")
print("=" * 80)

for cfg in configs:
    label = cfg["label"]
    basis = cfg["basis"]
    pct = cfg["pct"]

    BACKTEST_CONFIG["weight_cap_pct"] = CAP_PCT

    total_stops = 0
    total_stocks = 0
    all_trigger_rets = []  # 손절 시점 확정 수익률
    period_details = []
    prev_stocks = []

    for i in range(len(rebal_dates) - 1):
        start = rebal_dates[i]
        end = rebal_dates[i + 1]

        stocks = get_stocks(start)
        if not stocks and prev_stocks:
            stocks = prev_stocks
        prev_stocks = stocks

        if not stocks:
            continue

        raw_return, sl_events, _ = calc_portfolio_return_with_stoploss(
            conn, stocks, start, end,
            stop_loss_pct=pct, stop_loss_mode="sell", stop_loss_basis=basis,
        )

        n_stocks = len(stocks)
        n_stopped = len(sl_events)
        total_stops += n_stopped
        total_stocks += n_stocks

        for code, dt, ret, wt in sl_events:
            all_trigger_rets.append(ret)

        if n_stopped > 0:
            period_details.append((start, end, n_stocks, n_stopped, sl_events))

    # ── 요약 출력 ──
    n_periods = len(rebal_dates) - 1
    stop_rate = total_stops / total_stocks * 100 if total_stocks else 0
    periods_with_stop = len(period_details)

    print(f"\n{'─' * 40}")
    print(f"▶ {label}")
    print(f"{'─' * 40}")
    print(f"  총 리밸런싱 기간:       {n_periods}회")
    print(f"  손절 발동 기간:         {periods_with_stop}회 ({periods_with_stop/n_periods*100:.0f}%)")
    print(f"  총 종목·기간:           {total_stocks}건")
    print(f"  총 손절 발동:           {total_stops}건 ({stop_rate:.1f}%)")

    if all_trigger_rets:
        rets = np.array(all_trigger_rets)
        print(f"\n  손절 시점 확정 수익률 분포:")
        print(f"    평균:    {np.mean(rets):+.2%}")
        print(f"    중간값:  {np.median(rets):+.2%}")
        print(f"    최소:    {np.min(rets):+.2%}")
        print(f"    최대:    {np.max(rets):+.2%}")
        n_profit = sum(1 for r in rets if r > 0)
        print(f"    이익 중 손절: {n_profit}건 ({n_profit/len(rets)*100:.0f}%)")
        print(f"    손실 중 손절: {len(rets)-n_profit}건 ({(len(rets)-n_profit)/len(rets)*100:.0f}%)")

    # 상세 (발동 많은 기간 Top 5)
    if period_details:
        print(f"\n  손절 많은 기간 Top 5:")
        sorted_periods = sorted(period_details, key=lambda x: x[3], reverse=True)[:5]
        for start, end, n_st, n_sl, events in sorted_periods:
            print(f"    {start} ~ {end}: {n_sl}/{n_st}종목 손절")
            for code, dt, ret, wt in events[:3]:
                print(f"      {code} @ {dt} ({ret:+.1%}, 비중 {wt:.1%})")
            if len(events) > 3:
                print(f"      ... 외 {len(events)-3}건")

conn.close()
print(f"\n{'=' * 80}")
print("완료")
