"""
섹터 캡 25% + 시장 레짐 필터 조합 백테스트
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
        past = [kospi_prices.get(rb_dates[j]) for j in range(max(0, i-lookback), i)]
        past = [p for p in past if p]
        signals.append("bear" if past and current < np.mean(past) else "bull")
    return signals


def get_sector(conn, code):
    r = conn.execute("SELECT sector FROM stock_master WHERE stock_code = ?", (code,)).fetchone()
    if r and r[0]:
        return r[0].replace("코스피 ", "").replace("코스닥 ", "")
    return "미분류"


def get_candidates(conn, calc_date, max_n=60):
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


def select_with_sector_cap(conn, candidates, top_n=30, sector_cap_pct=None):
    if sector_cap_pct is None:
        return candidates[:top_n]
    max_per = max(1, int(top_n * sector_cap_pct / 100))
    selected, sec_count = [], {}
    for code, score in candidates:
        if len(selected) >= top_n:
            break
        sec = get_sector(conn, code)
        if sec_count.get(sec, 0) >= max_per:
            continue
        selected.append((code, score))
        sec_count[sec] = sec_count.get(sec, 0) + 1
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
    ret = 0.0
    for i, (code, _) in enumerate(stocks):
        sp = conn.execute("SELECT close FROM daily_price WHERE stock_code=? AND trade_date>=? ORDER BY trade_date ASC LIMIT 1", (code, start_date)).fetchone()
        ep = conn.execute("SELECT close FROM daily_price WHERE stock_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1", (code, end_date)).fetchone()
        if sp and ep and sp[0] > 0:
            ret += ((ep[0] - sp[0]) / sp[0]) * weights[i]
    return ret


def calc_stats(rets):
    rets = np.array(rets)
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    total = cum[-1] - 1
    n = len(rets) / 12
    return {
        "total_return": total,
        "cagr": (1 + total) ** (1/n) - 1 if n > 0 else 0,
        "sharpe": rets.mean() / rets.std() * np.sqrt(12) if rets.std() > 0 else 0,
        "mdd": float(dd.min()),
        "win_rate": (rets > 0).sum() / len(rets),
    }


def run_strategy(conn, rb_dates, signals, sector_cap=None, cash_pct=0, label=""):
    print(f"\n▶ {label}...")
    tx_cost = BACKTEST_CONFIG["transaction_cost_bp"] / 10000
    rets, prev_stocks, prev_list = [], set(), []

    for i in range(len(rb_dates) - 1):
        start, end = rb_dates[i], rb_dates[i + 1]
        candidates = get_candidates(conn, start, 60)
        stocks = select_with_sector_cap(conn, candidates, 30, sector_cap)
        if not stocks and prev_list:
            stocks = prev_list
        prev_list = stocks

        cur = {c for c, _ in stocks}
        if prev_stocks:
            changed = len(cur - prev_stocks) + len(prev_stocks - cur)
            turnover = changed / (2 * max(len(cur), 1))
        else:
            turnover = 1.0
        prev_stocks = cur

        raw = calc_portfolio_return(conn, stocks, start, end)
        net = raw - turnover * tx_cost * 2

        if signals[i] == "bear" and cash_pct > 0:
            net = net * (1 - cash_pct / 100)

        rets.append(net)
        if (i + 1) % 12 == 0:
            print(f"  ... {i+1}/{len(rb_dates)-1}")
        clear_factor_cache()

    stats = calc_stats(rets)
    print(f"  수익률: {stats['total_return']:.1%}, CAGR: {stats['cagr']:.1%}, "
          f"Sharpe: {stats['sharpe']:.2f}, MDD: {stats['mdd']:.1%}, 승률: {stats['win_rate']:.0%}")
    return stats


def main():
    conn = get_conn()
    rb_dates = get_monthly_rebalance_dates(conn)
    BACKTEST_CONFIG["weight_cap_pct"] = 10
    signals = get_kospi_ma_signal(conn, rb_dates, 12)

    print("=" * 70)
    print("섹터 캡 + 시장 레짐 필터 조합 백테스트")
    print("=" * 70)

    results = {}
    configs = [
        ("A0 기준", None, 0),
        ("섹터 캡 25%", 25, 0),
        ("레짐 필터 (현금 50%)", None, 50),
        ("섹터25% + 레짐50%", 25, 50),
        ("레짐 필터 (현금 70%)", None, 70),
        ("섹터25% + 레짐70%", 25, 70),
    ]

    for label, sec_cap, cash in configs:
        stats = run_strategy(conn, rb_dates, signals, sec_cap, cash, label)
        results[label] = stats

    print(f"\n{'='*70}")
    print("결과 비교")
    print(f"{'='*70}")
    print(f"{'전략':<24} {'수익률':>8} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'승률':>6}")
    print("-" * 66)
    for name, s in results.items():
        print(f"{name:<24} {s['total_return']:>+7.1%} {s['cagr']:>+7.1%} "
              f"{s['sharpe']:>7.2f} {s['mdd']:>7.1%} {s['win_rate']:>5.0%}")

    conn.close()
    print("\n완료!")


if __name__ == "__main__":
    main()
