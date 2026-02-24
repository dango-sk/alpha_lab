"""
Step 7: 백테스트 (4개 밸류 전략)

전략:
  A0:   원본 사분위 밸류 (value_score_orig)
  A:    v3 십분위 밸류 (value_score)
  A+M:  밸류 + 모멘텀 (value_score + tech_score)
  ATT2: 회귀 매력도만 (ATT_PBR + ATT_EVIC)

비중: 시총 비례 + 15% 캡 (전 전략 공통)
"""
import json
import sqlite3
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DB_PATH, BACKTEST_CONFIG, CACHE_DIR


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_monthly_rebalance_dates(conn):
    """매월 첫 거래일 리스트 + 종료일 끝점"""
    trade_dates = conn.execute("""
        SELECT DISTINCT trade_date FROM daily_price
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, (BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"])).fetchall()

    monthly = []
    current_month = ""
    for (td,) in trade_dates:
        month = td[:7]
        if month != current_month:
            monthly.append(td)
            current_month = month

    if trade_dates:
        last_trade = trade_dates[-1][0]
        if monthly and last_trade > monthly[-1]:
            monthly.append(last_trade)

    return monthly


def get_portfolio_stocks(conn, calc_date, strategy, top_n=30):
    """전략별 포트폴리오 종목 선정"""
    CANDIDATE_POOL = int(top_n * 2.0)

    if strategy == "A":
        rows = conn.execute("""
            SELECT vf.stock_code, vf.value_score as score
            FROM valuation_factors vf
            WHERE vf.quality_pass = 1 AND vf.calc_date = ?
            ORDER BY vf.value_score DESC
            LIMIT ?
        """, (calc_date, CANDIDATE_POOL)).fetchall()

    elif strategy == "A0":
        rows = conn.execute("""
            SELECT vf.stock_code, vf.value_score_orig as score
            FROM valuation_factors vf
            WHERE vf.quality_pass = 1 AND vf.calc_date = ?
              AND vf.value_score_orig IS NOT NULL
            ORDER BY vf.value_score_orig DESC
            LIMIT ?
        """, (calc_date, CANDIDATE_POOL)).fetchall()

    elif strategy == "VM":
        rows = conn.execute("""
            SELECT s.stock_code, (vf.value_score + COALESCE(s.tech_score, 0)) as score
            FROM signals s
            JOIN valuation_factors vf ON s.stock_code = vf.stock_code
                AND vf.calc_date = s.calc_date
            WHERE s.calc_date = ? AND vf.quality_pass = 1
            ORDER BY score DESC
            LIMIT ?
        """, (calc_date, CANDIDATE_POOL)).fetchall()

    elif strategy == "ATT2":
        # att2_score: step3에서 A0와 동일 체계로 사전 계산
        # 대형=사분위(0~4)/4*100, 중소=십분위(0~10)/10*100 → 0~100
        rows = conn.execute("""
            SELECT vf.stock_code, vf.att2_score as score
            FROM valuation_factors vf
            WHERE vf.quality_pass = 1 AND vf.calc_date = ?
              AND vf.att2_score IS NOT NULL
            ORDER BY vf.att2_score DESC
            LIMIT ?
        """, (calc_date, CANDIDATE_POOL)).fetchall()

    else:
        return []

    # ─── 유동성 + 생존자 필터 ───
    filtered = []
    for code, score in rows:
        if len(filtered) >= top_n:
            break

        vol_data = conn.execute("""
            SELECT AVG(close * volume) as avg_trade_amount
            FROM daily_price
            WHERE stock_code = ? AND trade_date <= ?
              AND trade_date >= date(?, '-30 days')
        """, (code, calc_date, calc_date)).fetchone()

        price_exists = conn.execute("""
            SELECT COUNT(*) FROM daily_price
            WHERE stock_code = ? AND trade_date >= date(?, '-5 days')
              AND trade_date <= ?
        """, (code, calc_date, calc_date)).fetchone()[0]

        if price_exists == 0:
            continue
        if vol_data and vol_data[0] and vol_data[0] >= 100_000_000:
            filtered.append((code, score))

    return filtered


def _apply_mcap_cap(raw_weights, cap=0.15):
    """시총 비중에 상한선 적용 (초과분 반복 재배분)"""
    weights = [w / sum(raw_weights) for w in raw_weights]
    for _ in range(20):
        capped = [min(w, cap) for w in weights]
        excess = sum(w - min(w, cap) for w in weights)
        if excess < 1e-9:
            break
        uncapped_sum = sum(w for w in capped if w < cap)
        if uncapped_sum <= 0:
            break
        weights = [
            min(w + excess * (w / uncapped_sum), cap) if w < cap else cap
            for w in capped
        ]
    return weights


def _calc_slippage(market_cap):
    """시총 구간별 슬리피지 (매수+매도 합산)"""
    if market_cap >= 1_000_000_000_000:
        return 0.0010
    elif market_cap >= 300_000_000_000:
        return 0.0020
    elif market_cap >= 100_000_000_000:
        return 0.0030
    else:
        return 0.0050


def calc_portfolio_return(conn, stocks, start_date, end_date):
    """포트폴리오 수익률 계산 (시총비중 + 15% 캡)"""
    if not stocks:
        return 0.0

    # ─── 시총 비중 + 15% 캡 ───
    raw_mcaps = []
    for code, _ in stocks:
        row = conn.execute("""
            SELECT market_cap FROM daily_price
            WHERE stock_code = ? AND trade_date >= ?
            ORDER BY trade_date ASC LIMIT 1
        """, (code, start_date)).fetchone()
        raw_mcaps.append(row[0] if row and row[0] else 0)

    cap = BACKTEST_CONFIG.get("weight_cap_pct", 15) / 100
    weights = _apply_mcap_cap(raw_mcaps, cap=cap)

    # ─── 종목별 수익률 ───
    weighted_returns = []
    for i, (code, _) in enumerate(stocks):
        start_price = conn.execute("""
            SELECT close FROM daily_price
            WHERE stock_code = ? AND trade_date >= ?
            ORDER BY trade_date ASC LIMIT 1
        """, (code, start_date)).fetchone()

        end_price = conn.execute("""
            SELECT close FROM daily_price
            WHERE stock_code = ? AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 1
        """, (code, end_date)).fetchone()

        if start_price and start_price[0] > 0:
            if end_price and end_price[0] > 0:
                ret = (end_price[0] - start_price[0]) / start_price[0]
                weighted_returns.append(ret * weights[i])
            else:
                # 상폐: 마지막 가격으로 계산
                last_price = conn.execute("""
                    SELECT close FROM daily_price
                    WHERE stock_code = ? AND trade_date >= ?
                    ORDER BY trade_date DESC LIMIT 1
                """, (code, start_date)).fetchone()

                if last_price and last_price[0] > 0:
                    ret = (last_price[0] - start_price[0]) / start_price[0]
                    weighted_returns.append(ret * weights[i])
                else:
                    weighted_returns.append(-1.0 * weights[i])

    return sum(weighted_returns) if weighted_returns else 0.0


def calc_etf_return(conn, etf_code, start_date, end_date):
    """단일 ETF 수익률 계산"""
    start_price = conn.execute("""
        SELECT close FROM daily_price
        WHERE stock_code = ? AND trade_date >= ?
        ORDER BY trade_date ASC LIMIT 1
    """, (etf_code, start_date)).fetchone()

    end_price = conn.execute("""
        SELECT close FROM daily_price
        WHERE stock_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT 1
    """, (etf_code, end_date)).fetchone()

    if start_price and end_price and start_price[0] > 0:
        return (end_price[0] - start_price[0]) / start_price[0]
    return None


def calc_etf_monthly_returns(conn, etf_code, rebalance_dates):
    """ETF의 월별 수익률 배열"""
    monthly_returns = []
    for i in range(len(rebalance_dates) - 1):
        ret = calc_etf_return(conn, etf_code, rebalance_dates[i], rebalance_dates[i + 1])
        monthly_returns.append(ret if ret is not None else 0.0)
    return monthly_returns


def calc_all_benchmarks(conn, rebalance_dates):
    """벤치마크 수익률 + MDD + Sharpe"""
    benchmarks = {"KOSPI": ("KS200", "KOSPI 200")}
    results = {}

    for key, (etf_code, name) in benchmarks.items():
        ret = calc_etf_return(conn, etf_code, rebalance_dates[0], rebalance_dates[-1])
        if ret is None:
            continue

        months = len(rebalance_dates) - 1
        cagr = ((1 + ret) ** (12.0 / months) - 1.0) if months > 0 else 0

        monthly_rets = calc_etf_monthly_returns(conn, etf_code, rebalance_dates)
        returns_array = np.array(monthly_rets)

        cumulative = 1.0
        portfolio_values = [1.0]
        for r in monthly_rets:
            cumulative *= (1 + r)
            portfolio_values.append(cumulative)

        peak = portfolio_values[0]
        mdd = 0
        for v in portfolio_values:
            if v > peak:
                peak = v
            drawdown = (peak - v) / peak
            if drawdown > mdd:
                mdd = drawdown

        sharpe = 0
        if returns_array.std() > 0:
            sharpe = (returns_array.mean() / returns_array.std()) * np.sqrt(12)

        results[key] = {
            "strategy": name,
            "total_return": ret,
            "cagr": cagr,
            "mdd": mdd,
            "sharpe": sharpe,
            "months": months,
            "monthly_returns": monthly_rets,
            "portfolio_values": portfolio_values,
            "rebalance_dates": list(rebalance_dates),
        }
    return results


def run_backtest(strategy_name, stock_selector=None, progress_callback=None):
    """단일 전략 백테스트 실행.

    stock_selector: 커스텀 종목 선정 콜백 (conn, calc_date, top_n) -> [(code, score), ...]
                    None이면 기존 DB 기반 get_portfolio_stocks 사용.
    progress_callback: 진행률 콜백 (current, total) -> None
    """
    conn = get_db()
    rebalance_dates = get_monthly_rebalance_dates(conn)

    if len(rebalance_dates) < 2:
        print("  리밸런싱 날짜가 부족합니다.")
        conn.close()
        return None

    top_n = BACKTEST_CONFIG["top_n_stocks"]
    tx_cost = BACKTEST_CONFIG["transaction_cost_bp"] / 10000

    cumulative = 1.0
    monthly_returns = []
    portfolio_values = [1.0]
    turnover_list = []
    prev_stocks = set()
    portfolio_sizes = []
    prev_stocks_list = []

    total_periods = len(rebalance_dates) - 1
    for i in range(total_periods):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1]

        if progress_callback:
            progress_callback(i, total_periods)

        if stock_selector:
            stocks = stock_selector(conn, start, top_n)
        else:
            stocks = get_portfolio_stocks(conn, start, strategy_name, top_n)
        if not stocks and prev_stocks_list:
            stocks = prev_stocks_list
        prev_stocks_list = stocks

        current_codes = set(code for code, _ in stocks)
        portfolio_sizes.append(len(stocks))

        # 턴오버
        if prev_stocks:
            changed = len(current_codes - prev_stocks) + len(prev_stocks - current_codes)
            turnover = changed / (2 * max(len(current_codes), 1))
        else:
            turnover = 1.0
        turnover_list.append(turnover)
        prev_stocks = current_codes

        # 수익률 (거래비용 = 턴오버 x (수수료 + 슬리피지) x 양방향)
        raw_return = calc_portfolio_return(conn, stocks, start, end)
        avg_slippage = np.mean([_calc_slippage(
            (conn.execute("SELECT market_cap FROM daily_price WHERE stock_code=? AND trade_date>=? ORDER BY trade_date ASC LIMIT 1",
                          (c, start)).fetchone() or [0])[0] or 0
        ) for c, _ in stocks]) if stocks else 0
        net_return = raw_return - (turnover * (tx_cost + avg_slippage) * 2)

        monthly_returns.append(net_return)
        cumulative *= (1 + net_return)
        portfolio_values.append(cumulative)

    conn.close()

    if not monthly_returns:
        return None

    returns_array = np.array(monthly_returns)
    total_return = cumulative - 1.0
    months = len(monthly_returns)
    cagr = (cumulative ** (12.0 / months) - 1.0) if months > 0 else 0

    peak = portfolio_values[0]
    mdd = 0
    for v in portfolio_values:
        if v > peak:
            peak = v
        drawdown = (peak - v) / peak
        if drawdown > mdd:
            mdd = drawdown

    sharpe = (returns_array.mean() / returns_array.std()) * np.sqrt(12) if returns_array.std() > 0 else 0

    return {
        "strategy": strategy_name,
        "total_return": total_return,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "months": months,
        "avg_monthly_return": returns_array.mean(),
        "monthly_std": returns_array.std(),
        "avg_turnover": np.mean(turnover_list) if turnover_list else 0,
        "avg_portfolio_size": np.mean(portfolio_sizes) if portfolio_sizes else 0,
        "monthly_returns": monthly_returns,
        "portfolio_values": portfolio_values,
        "portfolio_sizes": portfolio_sizes,
        "rebalance_dates": get_monthly_rebalance_dates(get_db()),
    }


# ─── 전략 정의 ───
STRATEGIES = [
    ("A0",   "A0",   "A0: 원본 사분위 밸류"),
    ("ATT2", "ATT2", "ATT2: 회귀 매력도 (ATT_PBR+ATT_EVIC)"),
]


def run_all_backtests():
    """4개 전략 + 벤치마크 백테스트"""
    print("\n" + "=" * 60)
    print("Step 7: 백테스트")
    print(f"   기간: {BACKTEST_CONFIG['start']} ~ {BACKTEST_CONFIG['end']}")
    print(f"   리밸런싱: 월 1회, 상위 {BACKTEST_CONFIG['top_n_stocks']}종목")
    print(f"   거래비용: 편도 {BACKTEST_CONFIG['transaction_cost_bp']}bp + 슬리피지")
    print(f"   비중: 시총비례 + 15% 캡")
    print("=" * 60)

    results = {}
    for key, strat, desc in STRATEGIES:
        print(f"\n  {key} ({desc}) 백테스트 중...")
        result = run_backtest(strat)
        if result:
            result["strategy"] = key
            results[key] = result
            print(f"     평균 포트폴리오: {result['avg_portfolio_size']:.0f}종목")

    conn = get_db()
    rebalance_dates = get_monthly_rebalance_dates(conn)
    if len(rebalance_dates) >= 2:
        bm_results = calc_all_benchmarks(conn, rebalance_dates)
        results.update(bm_results)
    conn.close()

    return results


# ─── 캐시 저장/로드 ───

def _numpy_to_python(obj):
    """numpy 타입을 JSON 직렬화 가능한 Python 타입으로 변환"""
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


BACKTEST_CACHE = CACHE_DIR / "backtest_results.json"


def save_backtest_cache(results):
    """백테스트 결과를 JSON 캐시로 저장"""
    CACHE_DIR.mkdir(exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(),
        "config": dict(BACKTEST_CONFIG),
        "results": _numpy_to_python(results),
    }
    BACKTEST_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"  캐시 저장: {BACKTEST_CACHE}")


def load_backtest_cache():
    """JSON 캐시에서 백테스트 결과 로드 (없으면 None)"""
    if not BACKTEST_CACHE.exists():
        return None
    data = json.loads(BACKTEST_CACHE.read_text())
    return data["results"]


def show_comparison(results):
    """전략 비교 출력"""
    print("\n" + "=" * 60)
    print("백테스트 결과 비교")
    print("=" * 60)

    print(f"\n  {'전략':<30} {'총수익률':>10} {'CAGR':>10} {'MDD':>10} {'Sharpe':>10}")
    print(f"  {'─'*72}")

    for key, _, desc in STRATEGIES:
        if key not in results:
            continue
        r = results[key]
        print(f"  {desc:<30} {r['total_return']:>+9.1%} {r['cagr']:>+9.1%} "
              f"{r['mdd']:>9.1%} {r['sharpe']:>10.2f}")

    print(f"  {'─'*72}")
    if "KOSPI" in results:
        r = results["KOSPI"]
        print(f"  {'BM: KOSPI 200':<30} {r['total_return']:>+9.1%} {r['cagr']:>+9.1%} "
              f"{r['mdd']:>9.1%} {r['sharpe']:>10.2f}")

    # ── IS/OOS 분할 ──
    is_end = BACKTEST_CONFIG.get("insample_end", "2024-06-30")
    oos_start = BACKTEST_CONFIG.get("oos_start", "2024-07-01")

    print(f"\n  IS: {BACKTEST_CONFIG['start']} ~ {is_end}")
    print(f"  OOS: {oos_start} ~ {BACKTEST_CONFIG['end']}")
    print(f"\n  {'전략':<30} | {'IS Sharpe':>10} {'IS 수익률':>10} | {'OOS Sharpe':>10} {'OOS 수익률':>10}")
    print(f"  {'─'*74}")

    ref_key = next((k for k, _, _ in STRATEGIES if k in results), None)
    if ref_key:
        rb_dates = results[ref_key].get("rebalance_dates", [])
        split_idx = next((i for i, d in enumerate(rb_dates) if d >= oos_start), 0)

        for key, _, desc in STRATEGIES:
            if key not in results:
                continue
            rets = np.array(results[key]["monthly_returns"])
            is_rets = rets[:split_idx]
            oos_rets = rets[split_idx:]
            is_cum = np.prod(1 + is_rets) - 1
            oos_cum = np.prod(1 + oos_rets) - 1
            is_sh = (is_rets.mean() / is_rets.std() * np.sqrt(12)) if is_rets.std() > 0 else 0
            oos_sh = (oos_rets.mean() / oos_rets.std() * np.sqrt(12)) if oos_rets.std() > 0 else 0
            print(f"  {desc:<30} | {is_sh:>10.2f} {is_cum:>+9.1%} | {oos_sh:>10.2f} {oos_cum:>+9.1%}")

        if "KOSPI" in results:
            conn = get_db()
            bm_monthly = calc_etf_monthly_returns(conn, "KS200", rb_dates)
            conn.close()
            bm_is = np.array(bm_monthly[:split_idx])
            bm_oos = np.array(bm_monthly[split_idx:])
            bm_is_sh = (bm_is.mean() / bm_is.std() * np.sqrt(12)) if len(bm_is) > 0 and bm_is.std() > 0 else 0
            bm_oos_sh = (bm_oos.mean() / bm_oos.std() * np.sqrt(12)) if len(bm_oos) > 0 and bm_oos.std() > 0 else 0
            is_ret = calc_etf_return(get_db(), "KS200", BACKTEST_CONFIG["start"], is_end)
            oos_ret = calc_etf_return(get_db(), "KS200", oos_start, BACKTEST_CONFIG["end"])
            print(f"  {'─'*74}")
            print(f"  {'BM: KOSPI 200':<30} | {bm_is_sh:>10.2f} {is_ret or 0:>+9.1%} | {bm_oos_sh:>10.2f} {oos_ret or 0:>+9.1%}")

    # ── 통계적 유의성 ──
    print(f"\n  통계적 유의성 (vs KOSPI 200)")
    print(f"  {'─'*60}")
    if ref_key:
        conn = get_db()
        bm_monthly = np.array(calc_etf_monthly_returns(conn, "KS200", rb_dates))
        conn.close()

        for key, _, desc in STRATEGIES:
            if key not in results:
                continue
            strat_rets = np.array(results[key]["monthly_returns"])
            n = min(len(strat_rets), len(bm_monthly))
            diff = strat_rets[:n] - bm_monthly[:n]
            t_stat, p_val = stats.ttest_rel(strat_rets[:n], bm_monthly[:n])
            verdict = "유의 (p<0.05)" if p_val < 0.05 else "유의하지 않음"
            print(f"  {desc}: 월 {diff.mean()*100:+.3f}%, t={t_stat:.2f}, p={p_val:.4f} -> {verdict}")

    print(f"\n  Step 7 완료!")


if __name__ == "__main__":
    print("Step 7: 백테스트")
    print(f"   DB: {DB_PATH}")

    results = run_all_backtests()
    if results:
        show_comparison(results)
        save_backtest_cache(results)
