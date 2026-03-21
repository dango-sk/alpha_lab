"""
개선안 백테스트: 섹터 캡 + 밸류 트랩 필터

문제 1: 업종 쏠림 → 섹터 종목 수 상한
문제 2: 밸류 트랩 → 직전 N개월 연속 하락 종목 제외

테스트:
- A0 기준 (현재)
- 섹터 캡 25% only
- 밸류 트랩 필터 only (직전 3개월 연속 하락 제외)
- 섹터 캡 25% + 밸류 트랩 필터
"""
import sys
import numpy as np
from pathlib import Path
from collections import Counter

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


def get_sector(conn, code):
    r = conn.execute("SELECT sector FROM stock_master WHERE stock_code = ?", (code,)).fetchone()
    if r and r[0]:
        return r[0].replace("코스피 ", "").replace("코스닥 ", "")
    return "미분류"


def get_candidates(conn, calc_date, max_n=80):
    strategy_module = code_to_module(DEFAULT_STRATEGY_CODE)
    candidates = score_stocks_from_strategy(conn, calc_date, strategy_module)
    filtered = []
    for code, score in candidates:
        if len(filtered) >= max_n:
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


def check_value_trap(conn, code, calc_date, lookback_months=3):
    """직전 N개월 연속 하락이면 True (= 밸류 트랩 의심)"""
    # calc_date 기준으로 과거 N+1개월치 월초 가격 조회
    prices = conn.execute("""
        SELECT trade_date, close FROM daily_price
        WHERE stock_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT 90
    """, (code, calc_date)).fetchall()

    if len(prices) < 20:
        return False

    # 월별 수익률 계산 (대략 월초 가격 비교)
    monthly_prices = []
    current_month = ""
    for td, close in prices:
        month = td[:7]
        if month != current_month:
            monthly_prices.append(close)
            current_month = month
        if len(monthly_prices) > lookback_months + 1:
            break

    if len(monthly_prices) < lookback_months + 1:
        return False

    # monthly_prices[0] = 최근, [1] = 1개월 전, ...
    consecutive_down = 0
    for i in range(lookback_months):
        if monthly_prices[i] < monthly_prices[i + 1]:
            consecutive_down += 1
        else:
            break

    return consecutive_down >= lookback_months


def select_stocks(conn, candidates, calc_date, top_n=30,
                  sector_cap_pct=None, filter_value_trap=False, trap_months=3):
    """종목 선정: 섹터 캡 + 밸류 트랩 필터 적용"""

    max_per_sector = None
    if sector_cap_pct is not None:
        max_per_sector = max(1, int(top_n * sector_cap_pct / 100))

    selected = []
    sector_count = {}
    trap_skipped = 0

    for code, score in candidates:
        if len(selected) >= top_n:
            break

        # 밸류 트랩 필터
        if filter_value_trap:
            if check_value_trap(conn, code, calc_date, trap_months):
                trap_skipped += 1
                continue

        # 섹터 캡
        if max_per_sector is not None:
            sector = get_sector(conn, code)
            current = sector_count.get(sector, 0)
            if current >= max_per_sector:
                continue
            sector_count[sector] = current + 1

        selected.append((code, score))

    return selected, trap_skipped


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


def run_test(conn, rb_dates, label, sector_cap_pct=None,
             filter_value_trap=False, trap_months=3):
    print(f"\n▶ {label}...")
    tx_cost = BACKTEST_CONFIG["transaction_cost_bp"] / 10000
    monthly_rets = []
    prev_stocks = set()
    prev_stocks_list = []
    total_trap_skipped = 0

    for i in range(len(rb_dates) - 1):
        start, end = rb_dates[i], rb_dates[i + 1]

        candidates = get_candidates(conn, start, max_n=80)
        stocks, trap_skipped = select_stocks(
            conn, candidates, start, top_n=30,
            sector_cap_pct=sector_cap_pct,
            filter_value_trap=filter_value_trap,
            trap_months=trap_months,
        )
        total_trap_skipped += trap_skipped

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
        monthly_rets.append(net_ret)

        if (i + 1) % 12 == 0:
            print(f"  ... {i+1}/{len(rb_dates)-1}개월")

        clear_factor_cache()

    stats = calc_stats(monthly_rets)
    avg_trap = total_trap_skipped / (len(rb_dates) - 1) if filter_value_trap else 0

    print(f"  수익률: {stats['total_return']:.1%}, CAGR: {stats['cagr']:.1%}, "
          f"Sharpe: {stats['sharpe']:.2f}, MDD: {stats['mdd']:.1%}, 승률: {stats['win_rate']:.0%}")
    if filter_value_trap:
        print(f"  밸류트랩 제외: 월평균 {avg_trap:.1f}종목")

    return stats


def main():
    conn = get_conn()
    rb_dates = get_monthly_rebalance_dates(conn)
    BACKTEST_CONFIG["weight_cap_pct"] = 10

    print("=" * 70)
    print("개선안 백테스트: 섹터 캡 + 밸류 트랩 필터")
    print("=" * 70)

    # KOSPI
    kospi_rets = []
    for i in range(len(rb_dates) - 1):
        r = calc_etf_return(conn, "KS200", rb_dates[i], rb_dates[i + 1])
        kospi_rets.append(r if r is not None else 0.0)
    kospi_rets = np.array(kospi_rets)

    results = {}
    configs = [
        ("A0 기준", {}),
        ("섹터 캡 25%", {"sector_cap_pct": 25}),
        ("밸류트랩 필터 (3개월)", {"filter_value_trap": True, "trap_months": 3}),
        ("섹터25% + 트랩필터", {"sector_cap_pct": 25, "filter_value_trap": True, "trap_months": 3}),
    ]

    for label, kwargs in configs:
        stats = run_test(conn, rb_dates, label, **kwargs)
        results[label] = stats

    # ── 비교 ──
    print(f"\n{'='*70}")
    print("결과 비교")
    print(f"{'='*70}")
    print(f"{'전략':<26} {'수익률':>8} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'승률':>6}")
    print("-" * 70)
    for name, stats in results.items():
        print(f"{name:<26} {stats['total_return']:>+7.1%} {stats['cagr']:>+7.1%} "
              f"{stats['sharpe']:>7.2f} {stats['mdd']:>7.1%} {stats['win_rate']:>5.0%}")

    # KOSPI 대비
    print(f"\n{'='*70}")
    print("KOSPI 대비 언더퍼폼 비교")
    print(f"{'='*70}")
    for name, stats in results.items():
        rets = stats["monthly_returns"]
        min_len = min(len(rets), len(kospi_rets))
        excess = rets[:min_len] - kospi_rets[:min_len]
        under_count = (excess < 0).sum()
        under_avg = excess[excess < 0].mean() * 100 if (excess < 0).any() else 0
        worst = excess.min() * 100
        print(f"  {name}:")
        print(f"    언더퍼폼: {under_count}/{min_len}개월 ({under_count/min_len:.0%}), "
              f"평균 {under_avg:+.2f}%p, 최악 {worst:+.1f}%p")

    conn.close()
    print("\n완료!")


if __name__ == "__main__":
    main()
