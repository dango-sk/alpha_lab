"""
Step 7: 백테스트

전략:
  A0: 원본 사분위 밸류 (멀티팩터)

데이터: Railway PostgreSQL (alpha_lab 스키마)
유니버스: universe 테이블 (사전 필터 완료)
수익률: adj_close 기반
비중: 시총 비례 + 비중상한 캡 (전 전략 공통)
"""
import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import BACKTEST_CONFIG, CACHE_DIR
from lib.factor_engine import (
    score_stocks_from_strategy, code_to_module,
    DEFAULT_STRATEGY_CODE, clear_factor_cache,
)
from lib.db import get_conn


def get_db():
    return get_conn()


# ═══════════════════════════════════════════════════════
# 유니버스 & 리밸런싱 (PG universe 테이블 기반)
# ═══════════════════════════════════════════════════════

def get_rebalance_dates(conn, rebal_type="monthly"):
    """universe 테이블에서 리밸런싱 날짜 조회"""
    rows = conn.execute("""
        SELECT DISTINCT rebal_date FROM universe
        WHERE rebal_type = ? AND rebal_date >= ? AND rebal_date <= ?
        ORDER BY rebal_date
    """, (rebal_type, BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"])).fetchall()
    dates = [r[0] for r in rows]

    # 마지막 거래일 추가 (종료일까지의 수익률 계산용)
    if dates:
        last_trade = conn.execute("""
            SELECT MAX(trade_date) FROM daily_price
            WHERE trade_date <= ? AND trade_date > ?
        """, (BACKTEST_CONFIG["end"], dates[-1])).fetchone()
        if last_trade and last_trade[0] and last_trade[0] > dates[-1]:
            dates.append(last_trade[0])

    return dates


def get_universe_stocks(conn, rebal_date, rebal_type="monthly", min_market_cap=0):
    """universe 테이블에서 해당 날짜의 종목 set 반환"""
    rows = conn.execute("""
        SELECT stock_code, market_cap FROM universe
        WHERE rebal_date = ? AND rebal_type = ? AND market_cap >= ?
    """, (rebal_date, rebal_type, min_market_cap)).fetchall()
    return {code: mcap for code, mcap in rows}


# 하위 호환용
def get_monthly_rebalance_dates(conn):
    return get_rebalance_dates(conn, rebal_type=BACKTEST_CONFIG.get("rebal_type", "monthly"))


# ═══════════════════════════════════════════════════════
# 비중 & 슬리피지
# ═══════════════════════════════════════════════════════

def _apply_mcap_cap(raw_weights, cap=0.15):
    """시총 비중에 상한선 적용 (초과분 반복 재배분)"""
    total = sum(raw_weights)
    if total <= 0:
        n = len(raw_weights)
        return [1 / n] * n if n > 0 else []
    weights = [w / total for w in raw_weights]
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


# ═══════════════════════════════════════════════════════
# 수익률 계산 (adj_close 기반)
# ═══════════════════════════════════════════════════════

def calc_portfolio_return(conn, stocks, start_date, end_date):
    """포트폴리오 수익률 계산 (시총비중 + 비중상한 캡, adj_close 기반) — 배치 쿼리"""
    if not stocks:
        return 0.0

    codes = [code for code, _ in stocks]
    placeholders = ",".join(["?"] * len(codes))

    # ─── 배치: 시작일 이후 첫 거래일 데이터 (시총 + adj_close) ───
    start_rows = conn.execute(f"""
        SELECT dp.stock_code, dp.market_cap, dp.adj_close
        FROM daily_price dp
        INNER JOIN (
            SELECT stock_code, MIN(trade_date) as first_date
            FROM daily_price
            WHERE stock_code IN ({placeholders}) AND trade_date >= ?
            GROUP BY stock_code
        ) t ON dp.stock_code = t.stock_code AND dp.trade_date = t.first_date
    """, (*codes, start_date)).fetchall()
    start_map = {r[0]: (r[1], r[2]) for r in start_rows}  # code -> (mcap, adj_close)

    # ─── 배치: 종료일 이전 마지막 거래일 adj_close ───
    end_rows = conn.execute(f"""
        SELECT dp.stock_code, dp.adj_close
        FROM daily_price dp
        INNER JOIN (
            SELECT stock_code, MAX(trade_date) as last_date
            FROM daily_price
            WHERE stock_code IN ({placeholders}) AND trade_date <= ? AND adj_close > 0
            GROUP BY stock_code
        ) t ON dp.stock_code = t.stock_code AND dp.trade_date = t.last_date
    """, (*codes, end_date)).fetchall()
    end_map = {r[0]: r[1] for r in end_rows}  # code -> adj_close

    # ─── 배치: 시작일 이후 마지막 거래일 adj_close (상폐 대비) ───
    last_rows = conn.execute(f"""
        SELECT dp.stock_code, dp.adj_close
        FROM daily_price dp
        INNER JOIN (
            SELECT stock_code, MAX(trade_date) as last_date
            FROM daily_price
            WHERE stock_code IN ({placeholders}) AND trade_date >= ? AND adj_close > 0
            GROUP BY stock_code
        ) t ON dp.stock_code = t.stock_code AND dp.trade_date = t.last_date
    """, (*codes, start_date)).fetchall()
    last_map = {r[0]: r[1] for r in last_rows}  # code -> adj_close

    # ─── 시총 비중 + 비중상한 캡 ───
    raw_mcaps = [start_map.get(code, (0, 0))[0] or 0 for code in codes]

    cap_pct = BACKTEST_CONFIG.get("weight_cap_pct", 10)
    if cap_pct > 0:
        weights = _apply_mcap_cap(raw_mcaps, cap=cap_pct / 100)
    else:
        total = sum(raw_mcaps)
        weights = [w / total for w in raw_mcaps] if total > 0 else [1/len(raw_mcaps)] * len(raw_mcaps)

    # ─── 종목별 수익률 (adj_close) ───
    weighted_returns = []
    for i, code in enumerate(codes):
        sp = start_map.get(code, (0, 0))[1] or 0
        if sp <= 0:
            continue
        ep = end_map.get(code, 0) or 0
        if ep > 0:
            ret = (ep - sp) / sp
            weighted_returns.append(ret * weights[i])
        else:
            lp = last_map.get(code, 0) or 0
            if lp > 0:
                ret = (lp - sp) / sp
                weighted_returns.append(ret * weights[i])
            else:
                weighted_returns.append(-1.0 * weights[i])

    return sum(weighted_returns) if weighted_returns else 0.0


def calc_etf_return(conn, etf_code, start_date, end_date):
    """단일 ETF 수익률 계산 (adj_close)"""
    start_price = conn.execute("""
        SELECT adj_close FROM daily_price
        WHERE stock_code = ? AND trade_date >= ? AND adj_close > 0
        ORDER BY trade_date ASC LIMIT 1
    """, (etf_code, start_date)).fetchone()

    end_price = conn.execute("""
        SELECT adj_close FROM daily_price
        WHERE stock_code = ? AND trade_date <= ? AND adj_close > 0
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
    universe = BACKTEST_CONFIG.get("universe", "KOSPI")
    if universe == "KOSPI+KOSDAQ":
        benchmarks = {"KOSPI": ("292150", "KRX 300")}
    else:
        benchmarks = {"KOSPI": ("069500", "KODEX 200")}
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


# ═══════════════════════════════════════════════════════
# 백테스트 실행
# ═══════════════════════════════════════════════════════

def run_backtest(strategy_name, stock_selector=None, rebal_type="monthly", progress_callback=None):
    """단일 전략 백테스트 실행.

    stock_selector: 커스텀 종목 선정 콜백 (conn, calc_date, top_n) -> [(code, score), ...]
    rebal_type: "monthly" 또는 "biweekly"
    """
    conn = get_db()
    rebalance_dates = get_rebalance_dates(conn, rebal_type)

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
            stocks = []
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
        if stocks:
            slip_codes = [c for c, _ in stocks]
            slip_ph = ",".join(["?"] * len(slip_codes))
            slip_rows = conn.execute(f"""
                SELECT dp.stock_code, dp.market_cap
                FROM daily_price dp
                INNER JOIN (
                    SELECT stock_code, MIN(trade_date) as first_date
                    FROM daily_price
                    WHERE stock_code IN ({slip_ph}) AND trade_date >= ?
                    GROUP BY stock_code
                ) t ON dp.stock_code = t.stock_code AND dp.trade_date = t.first_date
            """, (*slip_codes, start)).fetchall()
            slip_mcap_map = {r[0]: r[1] or 0 for r in slip_rows}
            avg_slippage = np.mean([_calc_slippage(slip_mcap_map.get(c, 0)) for c, _ in stocks])
        else:
            avg_slippage = 0
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
        "rebalance_dates": list(rebalance_dates),
    }


# ─── 전략 정의 ───
STRATEGIES = [
    ("A0",   "A0",   "A0: 멀티팩터 전략"),
]

# 기본 전략 코드 맵 (factor_engine 기반 파이프라인)
_BASE_STRATEGY_CODES = {
    "A0": DEFAULT_STRATEGY_CODE,
}


def make_engine_selector(strategy_key, rebal_type="monthly", min_market_cap=0):
    """factor_engine 기반 stock_selector 콜백 생성.

    universe 테이블과 교집합하여 종목 선정 (N+1 유동성 쿼리 제거).
    """
    code = _BASE_STRATEGY_CODES.get(strategy_key)
    if not code:
        return None
    strategy_module = code_to_module(code)

    def selector(conn, calc_date, top_n):
        # universe 테이블에서 해당 날짜 종목 조회
        universe_set = get_universe_stocks(conn, calc_date, rebal_type, min_market_cap)
        if not universe_set:
            return []

        # factor_engine으로 스코어링
        candidates = score_stocks_from_strategy(conn, calc_date, strategy_module)

        # universe와 교집합 → top_n
        filtered = [(code_str, score) for code_str, score in candidates
                     if code_str in universe_set][:top_n]
        return filtered

    return selector


def run_all_backtests(rebal_type=None, min_market_cap=None):
    """기본 전략 + 벤치마크 백테스트 (factor_engine 파이프라인 사용)"""
    if rebal_type is None:
        rebal_type = BACKTEST_CONFIG.get("rebal_type", "monthly")
    if min_market_cap is None:
        min_market_cap = BACKTEST_CONFIG.get("min_market_cap", 500_000_000_000)

    rebal_label = "격주" if rebal_type == "biweekly" else "월간"
    print("\n" + "=" * 60)
    print("Step 7: 백테스트")
    print(f"   기간: {BACKTEST_CONFIG['start']} ~ {BACKTEST_CONFIG['end']}")
    print(f"   리밸런싱: {rebal_label}, 상위 {BACKTEST_CONFIG['top_n_stocks']}종목")
    print(f"   거래비용: 편도 {BACKTEST_CONFIG['transaction_cost_bp']}bp + 슬리피지")
    print(f"   비중: 시총비례 + {BACKTEST_CONFIG.get('weight_cap_pct', 10)}% 캡")
    print(f"   시총하한: {min_market_cap/1e8:,.0f}억원")
    print(f"   파이프라인: factor_engine (퀄리티필터→스코어링)")
    print("=" * 60)

    results = {}
    for key, strat, desc in STRATEGIES:
        print(f"\n  {key} ({desc}) 백테스트 중...")
        selector = make_engine_selector(key, rebal_type, min_market_cap)
        result = run_backtest(strat, stock_selector=selector, rebal_type=rebal_type)
        clear_factor_cache()
        if result:
            result["strategy"] = key
            results[key] = result
            print(f"     평균 포트폴리오: {result['avg_portfolio_size']:.0f}종목")

    conn = get_db()
    rebalance_dates = get_rebalance_dates(conn, rebal_type)
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


HOLDINGS_CACHE = CACHE_DIR / "holdings_cache.json"
ATTRIBUTION_CACHE = CACHE_DIR / "attribution_cache.json"


def save_portfolio_cache(results):
    """모든 리밸런싱 날짜의 보유종목 + 기여도를 JSON 캐시로 저장."""
    conn = get_db()
    CACHE_DIR.mkdir(exist_ok=True)

    holdings_all = {}   # {strategy: {date: [rows]}}
    attr_all = {}       # {strategy: {"start_end": [rows]}}

    for key in results:
        if key == "KOSPI":
            continue
        rb_dates = results[key].get("rebalance_dates", [])
        if len(rb_dates) < 2:
            continue

        code = _BASE_STRATEGY_CODES.get(key)
        if not code:
            continue
        strategy_module = code_to_module(code)
        rebal_type = BACKTEST_CONFIG.get("rebal_type", "monthly")
        min_mcap = BACKTEST_CONFIG.get("min_market_cap", 0)
        top_n = BACKTEST_CONFIG.get("top_n_stocks", 30)
        cap = BACKTEST_CONFIG.get("weight_cap_pct", 10) / 100

        # 마스터 데이터: snapshot_date별로 로드
        from lib.db import read_sql
        master_all = read_sql(
            "SELECT stock_code, stock_name, COALESCE(sec_cd_nm, '기타') as sector, "
            "snapshot_date FROM fnspace_master",
            conn,
        )
        # snapshot_date → {stock_code: (name, sector)} 맵
        _master_by_snap = {}
        for r in master_all.itertuples():
            _master_by_snap.setdefault(r.snapshot_date, {})[r.stock_code] = (r.stock_name, r.sector)

        holdings_all[key] = {}
        attr_all[key] = {}

        for idx in range(len(rb_dates) - 1):
            calc_date = rb_dates[idx]

            # 해당 월 마스터 선택
            _snap = calc_date[:7]
            _snaps = sorted(s for s in _master_by_snap if s <= _snap)
            name_map = _master_by_snap.get(_snaps[-1], {}) if _snaps else {}

            # 종목 선정
            universe_set = get_universe_stocks(conn, calc_date, rebal_type, min_mcap)
            if not universe_set:
                continue
            candidates = score_stocks_from_strategy(conn, calc_date, strategy_module)
            clear_factor_cache()
            stocks = [(c, s) for c, s in candidates if c in universe_set][:top_n]
            if not stocks:
                continue

            codes = [c for c, _ in stocks]
            placeholders = ",".join(["?"] * len(codes))

            # 시총 + adj_close (시작)
            start_rows = conn.execute(f"""
                SELECT dp.stock_code, dp.market_cap, dp.adj_close
                FROM daily_price dp
                INNER JOIN (
                    SELECT stock_code, MIN(trade_date) as d
                    FROM daily_price WHERE stock_code IN ({placeholders}) AND trade_date >= ?
                    GROUP BY stock_code
                ) t ON dp.stock_code = t.stock_code AND dp.trade_date = t.d
            """, (*codes, calc_date)).fetchall()
            start_map = {r[0]: (r[1], r[2]) for r in start_rows}
            raw_mcaps = [start_map.get(c, (0, 0))[0] or 0 for c in codes]
            weights = _apply_mcap_cap(raw_mcaps, cap=cap)

            # 밸류에이션
            fin_rows = conn.execute(f"""
                SELECT ff.stock_code, ff.per, ff.pbr, ff.ev_ebitda
                FROM fnspace_finance ff
                INNER JOIN (
                    SELECT stock_code, MAX(fiscal_year) as my
                    FROM fnspace_finance
                    WHERE fiscal_quarter='Annual'
                      AND stock_code IN ({','.join(['?']*len(codes))})
                    GROUP BY stock_code
                ) t ON ff.stock_code = t.stock_code AND ff.fiscal_year = t.my
                    AND ff.fiscal_quarter = 'Annual'
            """, tuple(f"A{c}" for c in codes)).fetchall()
            fin_map = {r[0]: (r[1], r[2], r[3]) for r in fin_rows}

            # Holdings 생성
            h_rows = []
            for i, (code, score) in enumerate(stocks):
                acode = f"A{code}"
                nm = name_map.get(acode, (code, "기타"))
                fin = fin_map.get(acode, (None, None, None))
                h_rows.append({
                    "종목코드": code, "종목명": nm[0], "섹터": nm[1],
                    "비중(%)": round(weights[i] * 100, 2),
                    "점수": round(score, 1), "value_score": round(score, 1),
                    "PER": round(fin[0], 1) if fin[0] else None,
                    "PBR": round(fin[1], 2) if fin[1] else None,
                    "EV/EBITDA": round(fin[2], 1) if fin[2] else None,
                    "시가총액": raw_mcaps[i],
                })
            holdings_all[key][calc_date] = h_rows

            # Attribution (종료일 adj_close)
            if idx < len(rb_dates) - 1:
                end_date = rb_dates[idx + 1]
                ep_rows = conn.execute(f"""
                    SELECT dp.stock_code, dp.adj_close FROM daily_price dp
                    INNER JOIN (
                        SELECT stock_code, MAX(trade_date) as d FROM daily_price
                        WHERE stock_code IN ({placeholders}) AND trade_date <= ? AND adj_close > 0
                        GROUP BY stock_code
                    ) t ON dp.stock_code = t.stock_code AND dp.trade_date = t.d
                """, (*codes, end_date)).fetchall()
                ep_map = {r[0]: r[1] for r in ep_rows}

                a_rows = []
                for i, (code, _) in enumerate(stocks):
                    sp = start_map.get(code, (0, 0))[1] or 0
                    ep = ep_map.get(code, 0) or 0
                    if sp <= 0:
                        continue
                    ret = (ep - sp) / sp if ep > 0 else -1.0
                    acode = f"A{code}"
                    nm = name_map.get(acode, (code, "기타"))
                    a_rows.append({
                        "종목명": nm[0], "섹터": nm[1].replace("코스피 ", ""),
                        "비중(%)": round(weights[i] * 100, 1),
                        "종목수익률(%)": round(ret * 100, 1),
                        "기여도(%)": round(ret * weights[i] * 100, 2),
                    })
                attr_all[key][f"{calc_date}_{end_date}"] = a_rows

            if (idx + 1) % 20 == 0:
                print(f"    [{idx+1}/{len(rb_dates)-1}] {key} 포트폴리오 캐시...")

    HOLDINGS_CACHE.write_text(json.dumps(
        _numpy_to_python({"created_at": datetime.now().isoformat(), "data": holdings_all}),
        ensure_ascii=False, indent=2,
    ))
    ATTRIBUTION_CACHE.write_text(json.dumps(
        _numpy_to_python({"created_at": datetime.now().isoformat(), "data": attr_all}),
        ensure_ascii=False, indent=2,
    ))
    print(f"  포트폴리오 캐시 저장: {HOLDINGS_CACHE}, {ATTRIBUTION_CACHE}")
    conn.close()


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
        bm_name = r.get("strategy", "KODEX 200")
        print(f"  {f'BM: {bm_name}':<30} {r['total_return']:>+9.1%} {r['cagr']:>+9.1%} "
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
            universe = BACKTEST_CONFIG.get("universe", "KOSPI")
            _bm_code = "292150" if universe == "KOSPI+KOSDAQ" else "069500"
            _bm_name = "KRX 300" if _bm_code == "292150" else "KODEX 200"
            conn = get_db()
            bm_monthly = calc_etf_monthly_returns(conn, _bm_code, rb_dates)
            conn.close()
            bm_is = np.array(bm_monthly[:split_idx])
            bm_oos = np.array(bm_monthly[split_idx:])
            bm_is_sh = (bm_is.mean() / bm_is.std() * np.sqrt(12)) if len(bm_is) > 0 and bm_is.std() > 0 else 0
            bm_oos_sh = (bm_oos.mean() / bm_oos.std() * np.sqrt(12)) if len(bm_oos) > 0 and bm_oos.std() > 0 else 0
            is_ret = calc_etf_return(get_db(), _bm_code, BACKTEST_CONFIG["start"], is_end)
            oos_ret = calc_etf_return(get_db(), _bm_code, oos_start, BACKTEST_CONFIG["end"])
            print(f"  {'─'*74}")
            print(f"  {f'BM: {_bm_name}':<30} | {bm_is_sh:>10.2f} {is_ret or 0:>+9.1%} | {bm_oos_sh:>10.2f} {oos_ret or 0:>+9.1%}")

    # ── 통계적 유의성 ──
    universe = BACKTEST_CONFIG.get("universe", "KOSPI")
    _bm_code = "292150" if universe == "KOSPI+KOSDAQ" else "069500"
    _bm_name = "KRX 300" if _bm_code == "292150" else "KODEX 200"
    print(f"\n  통계적 유의성 (vs {_bm_name})")
    print(f"  {'─'*60}")
    if ref_key:
        conn = get_db()
        bm_monthly = np.array(calc_etf_monthly_returns(conn, _bm_code, rb_dates))
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

    results = run_all_backtests()
    if results:
        show_comparison(results)
        save_backtest_cache(results)
