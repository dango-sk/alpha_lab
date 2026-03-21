"""
시장 레짐 필터 백테스트

KOSPI 12개월 이동평균 하회 시 현금 비중 확대

테스트:
- A0 기준 (100% 투자)
- 하락장 시 현금 30%
- 하락장 시 현금 50%
- 하락장 시 현금 70%
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config.settings import BACKTEST_CONFIG
from lib.db import get_conn
from scripts.step7_backtest import (
    get_monthly_rebalance_dates, calc_etf_return, _apply_mcap_cap,
)
from lib.factor_engine import (
    score_stocks_from_strategy, code_to_module, DEFAULT_STRATEGY_CODE,
    clear_factor_cache,
)


def get_kospi_ma_signal(conn, rb_dates, lookback=12):
    """각 리밸런싱 시점에서 KOSPI가 12개월 이동평균 상회/하회 판단"""
    kospi_prices = {}
    for d in rb_dates:
        row = conn.execute("""
            SELECT close FROM daily_price
            WHERE stock_code = 'KS200' AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 1
        """, (d,)).fetchone()
        kospi_prices[d] = row[0] if row else None

    signals = []
    for i, d in enumerate(rb_dates):
        current = kospi_prices.get(d)
        if current is None or i < lookback:
            signals.append("bull")
            continue

        past_prices = []
        for j in range(max(0, i - lookback), i):
            p = kospi_prices.get(rb_dates[j])
            if p:
                past_prices.append(p)

        if not past_prices:
            signals.append("bull")
            continue

        ma = np.mean(past_prices)
        signals.append("bear" if current < ma else "bull")

    return signals


def get_strategy_stocks(conn, calc_date, top_n=30):
    strategy_module = code_to_module(DEFAULT_STRATEGY_CODE)
    candidates = score_stocks_from_strategy(conn, calc_date, strategy_module)
    filtered = []
    for code, score in candidates:
        if len(filtered) >= top_n:
            break
        price_exists = conn.execute("""
            SELECT COUNT(*) FROM daily_price
            WHERE stock_code = ? AND trade_date >= date(?, '-5 days')
              AND trade_date <= ?
        """, (code, calc_date, calc_date)).fetchone()[0]
        if price_exists == 0:
            continue
        vol_data = conn.execute("""
            SELECT AVG(close * volume) FROM daily_price
            WHERE stock_code = ? AND trade_date <= ?
              AND trade_date >= date(?, '-30 days')
        """, (code, calc_date, calc_date)).fetchone()
        if vol_data and vol_data[0] and vol_data[0] >= 100_000_000:
            filtered.append((code, score))
    return filtered


def calc_portfolio_return(conn, stocks, start_date, end_date):
    if not stocks:
        return 0.0
    raw_mcaps = []
    for code, _ in stocks:
        row = conn.execute("""
            SELECT market_cap FROM daily_price
            WHERE stock_code = ? AND trade_date >= ?
            ORDER BY trade_date ASC LIMIT 1
        """, (code, start_date)).fetchone()
        raw_mcaps.append(row[0] if row and row[0] else 0)

    cap = BACKTEST_CONFIG.get("weight_cap_pct", 10) / 100
    weights = _apply_mcap_cap(raw_mcaps, cap=cap)

    weighted_ret = 0.0
    for i, (code, _) in enumerate(stocks):
        sp = conn.execute("""
            SELECT close FROM daily_price
            WHERE stock_code = ? AND trade_date >= ?
            ORDER BY trade_date ASC LIMIT 1
        """, (code, start_date)).fetchone()
        ep = conn.execute("""
            SELECT close FROM daily_price
            WHERE stock_code = ? AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 1
        """, (code, end_date)).fetchone()
        if sp and ep and sp[0] > 0:
            ret = (ep[0] - sp[0]) / sp[0]
            weighted_ret += ret * weights[i]
    return weighted_ret


def calc_stats(monthly_returns):
    rets = np.array(monthly_returns)
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak

    total = cum[-1] - 1
    n_years = len(rets) / 12
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = rets.mean() / rets.std() * np.sqrt(12) if rets.std() > 0 else 0
    mdd = float(dd.min())
    win_rate = (rets > 0).sum() / len(rets)

    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "monthly_returns": rets,
    }


def main():
    conn = get_conn()
    rb_dates = get_monthly_rebalance_dates(conn)
    BACKTEST_CONFIG["weight_cap_pct"] = 10

    print("=" * 70)
    print("시장 레짐 필터 백테스트")
    print("KOSPI 12개월 이동평균 하회 시 현금 비중 확대")
    print("=" * 70)

    # 레짐 신호
    signals = get_kospi_ma_signal(conn, rb_dates, lookback=12)
    bear_count = sum(1 for s in signals[:-1] if s == "bear")
    total = len(rb_dates) - 1
    print(f"\n전체 {total}개월 중 하락장 신호: {bear_count}개월 ({bear_count/total:.0%})")

    bear_dates = [rb_dates[i] for i in range(total) if signals[i] == "bear"]
    print(f"하락장 시점: {', '.join(bear_dates[:8])}{'...' if len(bear_dates) > 8 else ''}")

    # 기준 수익률 계산 (한 번만)
    print("\n▶ 월별 수익률 계산 중...")
    tx_cost = BACKTEST_CONFIG["transaction_cost_bp"] / 10000
    raw_rets = []
    prev_stocks = set()
    prev_stocks_list = []

    for i in range(total):
        start, end = rb_dates[i], rb_dates[i + 1]
        stocks = get_strategy_stocks(conn, start, 30)
        if not stocks and prev_stocks_list:
            stocks = prev_stocks_list
        prev_stocks_list = stocks

        current_codes = {c for c, _ in stocks}
        if prev_stocks:
            changed = len(current_codes - prev_stocks) + len(prev_stocks - current_codes)
            turnover = changed / (2 * max(len(current_codes), 1))
        else:
            turnover = 1.0
        prev_stocks = current_codes

        raw_ret = calc_portfolio_return(conn, stocks, start, end)
        net_ret = raw_ret - turnover * tx_cost * 2
        raw_rets.append(net_ret)

        if (i + 1) % 12 == 0:
            print(f"  ... {i+1}/{total}개월")

        clear_factor_cache()

    raw_rets = np.array(raw_rets)

    # KOSPI
    kospi_rets = []
    for i in range(total):
        r = calc_etf_return(conn, "KS200", rb_dates[i], rb_dates[i + 1])
        kospi_rets.append(r if r is not None else 0.0)
    kospi_rets = np.array(kospi_rets)

    # 각 현금 비중별 결과
    results = {}
    for cash_pct, label in [(0, "A0 기준 (현금 0%)"), (30, "하락장 현금 30%"),
                             (50, "하락장 현금 50%"), (70, "하락장 현금 70%")]:
        adjusted = []
        for i in range(total):
            if signals[i] == "bear" and cash_pct > 0:
                # 주식 비중만큼만 수익, 나머지 현금 (수익률 0)
                stock_ratio = 1 - cash_pct / 100
                adjusted.append(raw_rets[i] * stock_ratio)
            else:
                adjusted.append(raw_rets[i])

        stats = calc_stats(adjusted)
        results[label] = stats

        bear_rets = [adjusted[i] for i in range(total) if signals[i] == "bear"]
        bull_rets = [adjusted[i] for i in range(total) if signals[i] == "bull"]

        print(f"\n▶ {label}")
        print(f"  수익률: {stats['total_return']:.1%}, CAGR: {stats['cagr']:.1%}, "
              f"Sharpe: {stats['sharpe']:.2f}, MDD: {stats['mdd']:.1%}, 승률: {stats['win_rate']:.0%}")
        if bear_rets:
            print(f"  하락장 월평균: {np.mean(bear_rets)*100:+.2f}%")
        if bull_rets:
            print(f"  상승장 월평균: {np.mean(bull_rets)*100:+.2f}%")

    # 비교 테이블
    print(f"\n{'='*70}")
    print("결과 비교")
    print(f"{'='*70}")
    print(f"{'전략':<22} {'수익률':>8} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'승률':>6}")
    print("-" * 62)
    for name, stats in results.items():
        print(f"{name:<22} {stats['total_return']:>+7.1%} {stats['cagr']:>+7.1%} "
              f"{stats['sharpe']:>7.2f} {stats['mdd']:>7.1%} {stats['win_rate']:>5.0%}")

    # KOSPI 대비
    print(f"\n{'='*70}")
    print("KOSPI 대비")
    print(f"{'='*70}")
    for name, stats in results.items():
        rets = stats["monthly_returns"]
        excess = rets - kospi_rets
        under = (excess < 0).sum()
        print(f"  {name}: 언더퍼폼 {under}/{total}개월 ({under/total:.0%})")

    conn.close()
    print("\n완료!")


if __name__ == "__main__":
    main()
