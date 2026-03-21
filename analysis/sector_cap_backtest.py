"""
섹터 비중 상한 백테스트

현재 문제: 특정 업종(금융·운송·자동차)에 50~65% 쏠림 → MDD 악화
해결: 섹터별 비중 상한 도입

테스트:
- A0 기준 (제한 없음)
- 섹터 상한 25%
- 섹터 상한 20%
- 섹터 상한 15%
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


def get_stock_name(conn, code):
    r = conn.execute("SELECT stock_name FROM stock_master WHERE stock_code = ?", (code,)).fetchone()
    return r[0] if r else code


def get_candidates(conn, calc_date, max_n=60):
    """점수순 후보 종목 (넉넉히 뽑아서 섹터 필터링용)"""
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


def apply_sector_cap(conn, candidates, calc_date, top_n=30, sector_cap_pct=None):
    """
    섹터 종목 수 상한 적용하여 종목 선정.

    방식: 점수 순으로 종목을 담되, 해당 섹터의 종목 수가
    top_n * sector_cap_pct/100 을 초과하면 스킵.
    예: top_n=30, sector_cap_pct=20 → 한 섹터 최대 6종목
    """
    if sector_cap_pct is None:
        return candidates[:top_n]

    max_per_sector = max(1, int(top_n * sector_cap_pct / 100))

    selected = []
    sector_count = {}

    for code, score in candidates:
        if len(selected) >= top_n:
            break

        sector = get_sector(conn, code)
        current = sector_count.get(sector, 0)

        if current >= max_per_sector:
            continue

        selected.append((code, score))
        sector_count[sector] = current + 1

    return selected


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


def run_test(conn, rb_dates, sector_cap_pct, label):
    print(f"\n▶ {label}...")
    tx_cost = BACKTEST_CONFIG["transaction_cost_bp"] / 10000
    monthly_rets = []
    prev_stocks = set()
    prev_stocks_list = []
    sector_logs = []

    for i in range(len(rb_dates) - 1):
        start, end = rb_dates[i], rb_dates[i + 1]

        candidates = get_candidates(conn, start, max_n=60)
        stocks = apply_sector_cap(conn, candidates, start, top_n=30, sector_cap_pct=sector_cap_pct)

        if not stocks and prev_stocks_list:
            stocks = prev_stocks_list
        prev_stocks_list = stocks

        # 섹터 분포 기록
        sec_counter = Counter()
        for code, _ in stocks:
            sec_counter[get_sector(conn, code)] += 1
        top_sec = sec_counter.most_common(1)[0] if sec_counter else ("", 0)
        sector_logs.append({"date": start, "top_sector": top_sec[0], "top_count": top_sec[1], "n_sectors": len(sec_counter)})

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

    # 섹터 집중도 통계
    avg_n_sectors = np.mean([s["n_sectors"] for s in sector_logs])
    avg_top_count = np.mean([s["top_count"] for s in sector_logs])

    print(f"  수익률: {stats['total_return']:.1%}, CAGR: {stats['cagr']:.1%}, "
          f"Sharpe: {stats['sharpe']:.2f}, MDD: {stats['mdd']:.1%}, 승률: {stats['win_rate']:.0%}")
    print(f"  평균 섹터 수: {avg_n_sectors:.1f}, 최다섹터 평균 종목수: {avg_top_count:.1f}/30")

    return stats, sector_logs


def main():
    conn = get_conn()
    rb_dates = get_monthly_rebalance_dates(conn)
    BACKTEST_CONFIG["weight_cap_pct"] = 10

    print("=" * 70)
    print("섹터 비중 상한 백테스트")
    print("=" * 70)

    # KOSPI 월별 수익률
    kospi_rets = []
    for i in range(len(rb_dates) - 1):
        r = calc_etf_return(conn, "KS200", rb_dates[i], rb_dates[i + 1])
        kospi_rets.append(r if r is not None else 0.0)
    kospi_rets = np.array(kospi_rets)

    results = {}
    for cap, label in [
        (None, "A0 기준 (섹터 제한 없음)"),
        (25, "섹터 상한 25%"),
        (20, "섹터 상한 20%"),
        (15, "섹터 상한 15%"),
    ]:
        stats, logs = run_test(conn, rb_dates, cap, label)
        results[label] = stats

    # ── 비교 테이블 ──
    print(f"\n{'='*70}")
    print("결과 비교")
    print(f"{'='*70}")
    print(f"{'전략':<28} {'수익률':>8} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'승률':>6}")
    print("-" * 70)
    for name, stats in results.items():
        print(f"{name:<28} {stats['total_return']:>+7.1%} {stats['cagr']:>+7.1%} "
              f"{stats['sharpe']:>7.2f} {stats['mdd']:>7.1%} {stats['win_rate']:>5.0%}")

    # KOSPI 대비 언더퍼폼 비교
    print(f"\n{'='*70}")
    print("KOSPI 대비 언더퍼폼 분석")
    print(f"{'='*70}")
    for name, stats in results.items():
        rets = stats["monthly_returns"]
        min_len = min(len(rets), len(kospi_rets))
        excess = rets[:min_len] - kospi_rets[:min_len]
        under_count = (excess < 0).sum()
        under_avg = excess[excess < 0].mean() * 100 if (excess < 0).any() else 0
        worst_month = excess.min() * 100
        print(f"  {name}:")
        print(f"    언더퍼폼 월: {under_count}/{min_len} ({under_count/min_len:.0%}), "
              f"평균 {under_avg:+.2f}%p, 최악 {worst_month:+.1f}%p")

    conn.close()
    print("\n완료!")


if __name__ == "__main__":
    main()
