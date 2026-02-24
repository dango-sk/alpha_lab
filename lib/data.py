"""
Dashboard data layer: cached loaders + constants.
"""
import json
import shutil
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ─── Path setup ───
ALPHA_LAB_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ALPHA_LAB_DIR))
sys.path.insert(0, str(ALPHA_LAB_DIR / "scripts"))

from config.settings import DB_PATH, BACKTEST_CONFIG, CACHE_DIR
from lib.factor_engine import (
    validate_strategy_code, code_to_module, score_stocks_from_strategy,
    DEFAULT_STRATEGY_CODE, clear_factor_cache,
)

# ─── Strategy constants (기본 전략) ───
BASE_STRATEGY_KEYS = ["A0", "ATT2"]
STRATEGY_KEYS = ["A0", "ATT2"]  # 동적으로 갱신됨
ALL_KEYS = ["A0", "ATT2", "KOSPI"]  # 동적으로 갱신됨

STRATEGY_LABELS = {
    "A0":    "기존전략",
    "ATT2":  "회귀only",
    "KOSPI": "KOSPI 200",
}

STRATEGY_COLORS = {
    "A0":    "#42A5F5",   # 밝은 파랑
    "ATT2":  "#26C6DA",   # 청록
    "KOSPI": "#90A4AE",   # 회색
}

# 삭제된 전략 — 캐시에 남아있을 수 있으므로 로딩 시 필터링
_REMOVED_STRATEGIES = {"A", "A+M", "VM"}

# 커스텀 전략용 팔레트
_CUSTOM_PALETTE = [
    "#FF6B6B",  # 코랄
    "#4ECDC4",  # 민트
    "#F7DC6F",  # 옐로
    "#BB8FCE",  # 퍼플
    "#F0B27A",  # 오렌지
    "#85C1E9",  # 스카이블루
]


def _update_strategy_registry(results: dict):
    """백테스트 결과 dict를 기반으로 전역 STRATEGY_KEYS/ALL_KEYS/LABELS/COLORS를 갱신.

    리스트는 in-place로 변경하여 다른 모듈의 import 참조가 자동 반영되도록 한다.
    """
    custom_keys = [k for k in results if k not in BASE_STRATEGY_KEYS and k != "KOSPI"]

    STRATEGY_KEYS.clear()
    STRATEGY_KEYS.extend(BASE_STRATEGY_KEYS)
    STRATEGY_KEYS.extend(custom_keys)

    ALL_KEYS.clear()
    ALL_KEYS.extend(STRATEGY_KEYS)
    if "KOSPI" in results:
        ALL_KEYS.append("KOSPI")

    for i, key in enumerate(custom_keys):
        if key not in STRATEGY_LABELS:
            STRATEGY_LABELS[key] = key
        if key not in STRATEGY_COLORS:
            STRATEGY_COLORS[key] = _CUSTOM_PALETTE[i % len(_CUSTOM_PALETTE)]

# step7 uses these internal codes
_STRAT_CODE = {"A0": "A0", "ATT2": "ATT2"}

# ─── 기존 전략 팩터 가중치 (step3 기준) ───
BASE_STRATEGY_WEIGHTS = {
    "A0": {
        "weights_large": {
            "T_PER": .05, "F_PER": .05, "T_EVEBITDA": .05, "F_EVEBITDA": .05,
            "T_PBR": .05, "F_PBR": .05, "T_PCF": .05,
            "ATT_PBR": .05, "ATT_EVIC": .05, "ATT_PER": .10, "ATT_EVEBIT": .10,
            "T_SPSG": .10, "F_SPSG": .10,
            "F_EPS_M": .15,
        },
        "weights_small": {
            "T_PER": .05, "F_PER": .05, "T_EVEBITDA": .05, "F_EVEBITDA": .05,
            "T_PBR": .05, "F_PBR": .05, "T_PCF": .05,
            "ATT_PBR": .05, "ATT_EVIC": .05, "ATT_PER": .10, "ATT_EVEBIT": .10,
            "T_SPSG": .10, "F_SPSG": .10,
            "PRICE_M": .05, "NDEBT_EBITDA": .05, "CURRENT": .05,
        },
        "regression_models": ["pbr_roe", "evic_roic", "fper_epsg", "fevebit_ebitg"],
        "scoring": {"large": "quartile", "small": "decile"},
        "large_only": ["F_EPS_M"],
        "small_only": ["PRICE_M", "NDEBT_EBITDA", "CURRENT"],
    },
    "ATT2": {
        "weights_large": {
            "ATT_PBR": .50, "ATT_EVIC": .50,
        },
        "weights_small": {
            "ATT_PBR": .50, "ATT_EVIC": .50,
        },
        "regression_models": ["pbr_roe", "evic_roic"],
        "scoring": {"large": "quartile", "small": "decile"},
        "large_only": [],
        "small_only": [],
    },
}


# ─── DB helper ───
def _get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@st.cache_data(ttl=3600)
def get_latest_price_date() -> str | None:
    """개별 종목 주가가 충분히 들어온 최신 거래일을 반환."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT trade_date FROM daily_price "
        "GROUP BY trade_date HAVING COUNT(DISTINCT stock_code) >= 100 "
        "ORDER BY trade_date DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def _get_strategy_stocks(conn, strategy: str, calc_date: str, top_n: int = 30):
    """기본 전략은 DB 쿼리, 커스텀 전략은 factor_engine으로 종목 선정."""
    from step7_backtest import get_portfolio_stocks

    if strategy in _STRAT_CODE:
        return get_portfolio_stocks(conn, calc_date, _STRAT_CODE[strategy], top_n)

    # 커스텀 전략: 저장된 코드 → 모듈 변환 → 채점
    data = load_strategy(strategy)
    if not data or "code" not in data:
        return []

    strategy_module = code_to_module(data["code"])
    all_stocks = score_stocks_from_strategy(conn, calc_date, strategy_module)

    # 유동성 + 생존자 필터 (step7_backtest.get_portfolio_stocks 동일 로직)
    filtered = []
    for code, score in all_stocks:
        if len(filtered) >= top_n:
            break
        vol_data = conn.execute(
            "SELECT AVG(close * volume) as avg_trade_amount "
            "FROM daily_price "
            "WHERE stock_code = ? AND trade_date <= ? "
            "  AND trade_date >= date(?, '-30 days')",
            (code, calc_date, calc_date),
        ).fetchone()
        price_exists = conn.execute(
            "SELECT COUNT(*) FROM daily_price "
            "WHERE stock_code = ? AND trade_date >= date(?, '-5 days') "
            "  AND trade_date <= ?",
            (code, calc_date, calc_date),
        ).fetchone()[0]
        if price_exists == 0:
            continue
        if vol_data and vol_data[0] and vol_data[0] >= 100_000_000:
            filtered.append((code, score))

    return filtered


# ═══════════════════════════════════════════════════════
# Tier 1: Pre-computed cache (instant) → fallback to computation
# ═══════════════════════════════════════════════════════

_BACKTEST_CACHE = CACHE_DIR / "backtest_results.json"
_ROBUSTNESS_CACHE = CACHE_DIR / "robustness_results.json"


def _period_cache_path(start: str, end: str) -> Path:
    """기간별 캐시 파일 경로"""
    return CACHE_DIR / f"backtest_{start}_{end}.json"


@st.cache_data(show_spinner=False)
def load_backtest_results(start: str = None, end: str = None):
    """백테스트 결과 로딩. start/end 지정 시 해당 기간으로 계산."""
    default_start = BACKTEST_CONFIG["start"]
    default_end = BACKTEST_CONFIG["end"]
    use_start = start or default_start
    use_end = end or default_end

    def _filter(results):
        return {k: v for k, v in results.items() if k not in _REMOVED_STRATEGIES}

    # 기본 기간이면 기존 캐시 사용
    if use_start == default_start and use_end == default_end:
        if _BACKTEST_CACHE.exists():
            return _filter(json.loads(_BACKTEST_CACHE.read_text())["results"])

    # 기간별 캐시 확인
    period_cache = _period_cache_path(use_start, use_end)
    if period_cache.exists():
        return _filter(json.loads(period_cache.read_text())["results"])

    # 캐시 없으면 실시간 계산
    from step7_backtest import run_all_backtests, save_backtest_cache
    from contextlib import contextmanager

    original_start = BACKTEST_CONFIG["start"]
    original_end = BACKTEST_CONFIG["end"]
    try:
        BACKTEST_CONFIG["start"] = use_start
        BACKTEST_CONFIG["end"] = use_end
        results = run_all_backtests()
        # 기간별 캐시 저장
        CACHE_DIR.mkdir(exist_ok=True)
        payload = {
            "created_at": __import__("datetime").datetime.now().isoformat(),
            "config": {"start": use_start, "end": use_end},
            "results": results,
        }
        # numpy 변환
        from step7_backtest import _numpy_to_python
        period_cache.write_text(json.dumps(_numpy_to_python(payload), ensure_ascii=False))
        return results
    finally:
        BACKTEST_CONFIG["start"] = original_start
        BACKTEST_CONFIG["end"] = original_end


def load_all_results(start: str = None, end: str = None) -> dict:
    """기본 백테스트(A0, ATT2, KOSPI) + 저장된 커스텀 전략 결과를 병합하여 반환.

    저장된 전략 중 백테스트 결과가 있는 것만 포함한다.
    병합 후 STRATEGY_KEYS/ALL_KEYS/LABELS/COLORS를 동적으로 갱신한다.
    """
    results = dict(load_backtest_results(start, end))  # 캐시 결과를 변경하지 않도록 복사

    # 저장된 커스텀 전략에서 백테스트 결과 병합
    for strat in list_strategies():
        name = strat["name"]
        if name in BASE_STRATEGY_KEYS or name == "KOSPI":
            continue  # 기본 전략과 이름 충돌 방지
        data = load_strategy(name)
        if data and data.get("results"):
            results[name] = data["results"]

    _update_strategy_registry(results)
    return results


@st.cache_data(show_spinner=False)
def load_robustness_results(start: str = None, end: str = None,
                            is_end: str = None, oos_start: str = None):
    """강건성 검증 결과. 기간 지정 시 백테스트 결과에서 실시간 계산."""
    default_start = BACKTEST_CONFIG["start"]
    default_end = BACKTEST_CONFIG["end"]
    use_start = start or default_start
    use_end = end or default_end
    use_is_end = is_end or BACKTEST_CONFIG.get("insample_end", "2024-06-30")
    use_oos_start = oos_start or BACKTEST_CONFIG.get("oos_start", "2024-07-01")

    # 기본 기간이면 step8 캐시 사용
    if use_start == default_start and use_end == default_end:
        if _ROBUSTNESS_CACHE.exists():
            data = json.loads(_ROBUSTNESS_CACHE.read_text())
            for sig in data["stat"].get("bm_significance", {}).values():
                if "boot_means" in sig:
                    sig["boot_means"] = np.array(sig["boot_means"])
            return data["is_oos"], data["stat"], data["rolling"]

    # 커스텀 기간 → 백테스트 결과에서 실시간 계산
    return _compute_robustness(use_start, use_end, use_is_end, use_oos_start)


def _compute_robustness(start, end, is_end, oos_start):
    """백테스트 결과에서 IS/OOS, 통계 유의성, 롤링 윈도우를 계산."""
    from scipy import stats as sp_stats
    from step7_backtest import run_backtest, get_db, calc_etf_return, \
        calc_etf_monthly_returns, get_monthly_rebalance_dates, STRATEGIES, \
        _numpy_to_python

    # IS/OOS 분할
    is_results, oos_results = {}, {}
    original_start = BACKTEST_CONFIG["start"]
    original_end = BACKTEST_CONFIG["end"]

    for key, strat, _ in STRATEGIES:
        try:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = start, is_end
            r = run_backtest(strat)
            if r:
                r["strategy"] = key
                is_results[key] = r
        finally:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = original_start, original_end

        try:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = oos_start, end
            r = run_backtest(strat)
            if r:
                r["strategy"] = key
                oos_results[key] = r
        finally:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = original_start, original_end

    # 벤치마크 IS/OOS
    conn = sqlite3.connect(DB_PATH)
    is_ret = calc_etf_return(conn, "KS200", start, is_end)
    oos_ret = calc_etf_return(conn, "KS200", oos_start, end)
    conn.close()

    baseline = next((k for k in ["ATT2", "A0"] if k in is_results), None)
    is_months = len(is_results.get(baseline, {}).get("monthly_returns", [])) if baseline else 1
    oos_months = len(oos_results.get(baseline, {}).get("monthly_returns", [])) if baseline else 1

    bm_results = {"is": {}, "oos": {}}
    if is_ret is not None:
        bm_results["is"]["KOSPI"] = {
            "total_return": is_ret,
            "cagr": ((1 + is_ret) ** (12.0 / max(is_months, 1)) - 1.0),
            "name": "KOSPI 200",
        }
    if oos_ret is not None:
        bm_results["oos"]["KOSPI"] = {
            "total_return": oos_ret,
            "cagr": ((1 + oos_ret) ** (12.0 / max(oos_months, 1)) - 1.0),
            "name": "KOSPI 200",
        }

    is_oos_data = {"is_results": is_results, "oos_results": oos_results, "benchmarks": bm_results}

    # 전체 기간 백테스트 + 통계 유의성
    full_results = {}
    for key, strat, _ in STRATEGIES:
        try:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = start, end
            r = run_backtest(strat)
            if r:
                r["strategy"] = key
                full_results[key] = r
        finally:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = original_start, original_end

    if not baseline or baseline not in full_results:
        baseline = next(iter(full_results), None)
    if not baseline:
        return is_oos_data, {"full_results": {}, "bm_significance": {}}, {}

    conn = sqlite3.connect(DB_PATH)
    rb_dates = full_results[baseline]["rebalance_dates"]
    bm_monthly = np.array(calc_etf_monthly_returns(conn, "KS200", rb_dates))
    conn.close()

    rng = np.random.default_rng(42)
    bm_significance = {}
    for key, _, _ in STRATEGIES:
        if key not in full_results:
            continue
        strat_rets = np.array(full_results[key]["monthly_returns"])
        n = min(len(strat_rets), len(bm_monthly))
        diff = strat_rets[:n] - bm_monthly[:n]
        t_stat, p_value = sp_stats.ttest_rel(strat_rets[:n], bm_monthly[:n])
        boot_means = np.array([
            rng.choice(diff, size=len(diff), replace=True).mean()
            for _ in range(10_000)
        ])
        bm_significance[key] = {
            "n_months": n, "mean_diff": float(diff.mean()),
            "t_stat": float(t_stat), "p_value": float(p_value),
            "ci_lower": float(np.percentile(boot_means, 2.5)),
            "ci_upper": float(np.percentile(boot_means, 97.5)),
            "significant": float(np.percentile(boot_means, 2.5)) > 0,
            "boot_means": boot_means,
            "win_rate": float((boot_means > 0).mean()),
        }

    stat_data = {"full_results": full_results, "bm_significance": bm_significance}

    # 롤링 윈도우
    rolling_window = 24
    rolling_all = {}
    for key, _, _ in STRATEGIES:
        if key not in full_results:
            continue
        strat_monthly = np.array(full_results[key]["monthly_returns"])
        n = min(len(strat_monthly), len(bm_monthly))
        if n < rolling_window:
            continue
        rolling_results = []
        for i in range(n - rolling_window + 1):
            w_s = strat_monthly[i:i + rolling_window]
            w_b = bm_monthly[i:i + rolling_window]
            excess = float(np.prod(1 + w_s) - 1 - (np.prod(1 + w_b) - 1))
            s_date = rb_dates[i] if i < len(rb_dates) else ""
            e_idx = i + rolling_window
            e_date = rb_dates[e_idx] if e_idx < len(rb_dates) else ""
            rolling_results.append({"start_date": s_date, "end_date": e_date, "excess_return": excess})
        pos = sum(1 for r in rolling_results if r["excess_return"] > 0)
        rolling_all[key] = {
            "total_windows": len(rolling_results),
            "positive_windows": pos,
            "win_rate": pos / len(rolling_results),
            "rolling_data": rolling_results,
        }

    return is_oos_data, stat_data, rolling_all


def load_all_robustness_results(start: str = None, end: str = None,
                                 is_end: str = None, oos_start: str = None):
    """기본 전략 강건성 + 커스텀 전략 강건성 (저장된 백테스트 결과 기반)."""
    is_oos_data, stat_data, rolling_all = load_robustness_results(
        start, end, is_end, oos_start,
    )

    # 캐시 결과 변경 방지를 위한 복사
    is_oos_data = {
        "is_results": dict(is_oos_data.get("is_results", {})),
        "oos_results": dict(is_oos_data.get("oos_results", {})),
        "benchmarks": is_oos_data.get("benchmarks", {}),
    }
    stat_data = {
        "full_results": dict(stat_data.get("full_results", {})),
        "bm_significance": dict(stat_data.get("bm_significance", {})),
    }
    rolling_all = dict(rolling_all)

    # BM 월별 수익률 (부트스트랩/롤링 비교용)
    use_start = start or BACKTEST_CONFIG["start"]
    use_end = end or BACKTEST_CONFIG["end"]
    use_oos_start = oos_start or BACKTEST_CONFIG.get("oos_start", "2024-07-01")

    base_results = load_backtest_results(use_start, use_end)
    baseline_key = next((k for k in ["ATT2", "A0"] if k in base_results), None)
    if not baseline_key:
        return is_oos_data, stat_data, rolling_all

    rb_dates = base_results[baseline_key].get("rebalance_dates", [])
    if not rb_dates:
        return is_oos_data, stat_data, rolling_all

    from step7_backtest import calc_etf_monthly_returns
    conn = sqlite3.connect(str(DB_PATH))
    bm_monthly = np.array(calc_etf_monthly_returns(conn, "KS200", rb_dates))
    conn.close()

    from scipy import stats as sp_stats

    for strat_info in list_strategies():
        name = strat_info["name"]
        if name in BASE_STRATEGY_KEYS or name == "KOSPI":
            continue
        if name in stat_data["bm_significance"]:
            continue

        data = load_strategy(name)
        if not data or not data.get("results"):
            continue

        r = data["results"]
        monthly_rets = r.get("monthly_returns", [])
        strat_rb_dates = r.get("rebalance_dates", [])
        if not monthly_rets or not strat_rb_dates:
            continue

        strat_rets = np.array(monthly_rets)

        # IS/OOS 분할 (저장된 월별 수익률 기반)
        split_idx = next(
            (i for i, d in enumerate(strat_rb_dates) if d >= use_oos_start),
            len(strat_rb_dates),
        )
        is_rets = strat_rets[:split_idx]
        oos_rets = strat_rets[split_idx:]

        if len(is_rets) > 0:
            is_cum = float(np.prod(1 + is_rets) - 1)
            n_is = len(is_rets)
            cum_arr = np.cumprod(1 + is_rets)
            peak = np.maximum.accumulate(cum_arr)
            dd = (cum_arr - peak) / peak
            is_oos_data["is_results"][name] = {
                "strategy": name,
                "total_return": is_cum,
                "cagr": float((1 + is_cum) ** (12.0 / max(n_is, 1)) - 1),
                "sharpe": float(is_rets.mean() / is_rets.std() * np.sqrt(12))
                    if n_is > 1 and is_rets.std() > 0 else 0,
                "mdd": float(dd.min()),
                "monthly_returns": is_rets.tolist(),
            }

        if len(oos_rets) > 0:
            oos_cum = float(np.prod(1 + oos_rets) - 1)
            n_oos = len(oos_rets)
            cum_arr = np.cumprod(1 + oos_rets)
            peak = np.maximum.accumulate(cum_arr)
            dd = (cum_arr - peak) / peak
            is_oos_data["oos_results"][name] = {
                "strategy": name,
                "total_return": oos_cum,
                "cagr": float((1 + oos_cum) ** (12.0 / max(n_oos, 1)) - 1),
                "sharpe": float(oos_rets.mean() / oos_rets.std() * np.sqrt(12))
                    if n_oos > 1 and oos_rets.std() > 0 else 0,
                "mdd": float(dd.min()),
                "monthly_returns": oos_rets.tolist(),
            }

        # 전체 기간 결과
        stat_data["full_results"][name] = r

        # 부트스트랩 유의성
        n = min(len(strat_rets), len(bm_monthly))
        if n > 1:
            diff = strat_rets[:n] - bm_monthly[:n]
            t_stat, p_value = sp_stats.ttest_rel(strat_rets[:n], bm_monthly[:n])
            rng = np.random.default_rng(42)
            boot_means = np.array([
                rng.choice(diff, size=len(diff), replace=True).mean()
                for _ in range(10_000)
            ])
            stat_data["bm_significance"][name] = {
                "n_months": n, "mean_diff": float(diff.mean()),
                "t_stat": float(t_stat), "p_value": float(p_value),
                "ci_lower": float(np.percentile(boot_means, 2.5)),
                "ci_upper": float(np.percentile(boot_means, 97.5)),
                "significant": float(np.percentile(boot_means, 2.5)) > 0,
                "boot_means": boot_means,
                "win_rate": float((boot_means > 0).mean()),
            }

        # 롤링 윈도우
        rolling_window = 24
        if n >= rolling_window:
            rolling_results = []
            for i in range(n - rolling_window + 1):
                w_s = strat_rets[i:i + rolling_window]
                w_b = bm_monthly[i:i + rolling_window]
                excess = float(np.prod(1 + w_s) - 1 - (np.prod(1 + w_b) - 1))
                s_date = strat_rb_dates[i] if i < len(strat_rb_dates) else ""
                e_idx = i + rolling_window
                e_date = strat_rb_dates[e_idx] if e_idx < len(strat_rb_dates) else ""
                rolling_results.append({
                    "start_date": s_date, "end_date": e_date,
                    "excess_return": excess,
                })
            pos = sum(1 for r in rolling_results if r["excess_return"] > 0)
            rolling_all[name] = {
                "total_windows": len(rolling_results),
                "positive_windows": pos,
                "win_rate": pos / len(rolling_results),
                "rolling_data": rolling_results,
            }

    return is_oos_data, stat_data, rolling_all


# ═══════════════════════════════════════════════════════
# Tier 2: DB queries (1-hour cache)
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_holdings(strategy: str, calc_date: str, top_n: int = 30) -> pd.DataFrame:
    """Get portfolio stocks with weights and metadata."""
    from step7_backtest import _apply_mcap_cap

    conn = _get_conn()
    stocks = _get_strategy_stocks(conn, strategy, calc_date, top_n)

    if not stocks:
        conn.close()
        return pd.DataFrame()

    # mcap weights
    raw_mcaps = []
    for code, _ in stocks:
        row = conn.execute(
            "SELECT market_cap FROM daily_price "
            "WHERE stock_code=? AND trade_date>=? ORDER BY trade_date ASC LIMIT 1",
            (code, calc_date),
        ).fetchone()
        raw_mcaps.append(row[0] if row and row[0] else 0)

    cap = BACKTEST_CONFIG.get("weight_cap_pct", 15) / 100
    weights = _apply_mcap_cap(raw_mcaps, cap=cap)

    holdings = []
    for i, (code, score) in enumerate(stocks):
        meta = conn.execute(
            "SELECT sm.stock_name, COALESCE(fm.sec_cd_nm, sm.sector, '기타') "
            "FROM stock_master sm "
            "LEFT JOIN fnspace_master fm ON 'A'||sm.stock_code = fm.stock_code "
            "WHERE sm.stock_code=?",
            (code,),
        ).fetchone()

        vf = conn.execute(
            "SELECT value_score, value_score_orig, per_rank, pbr_rank, per, pbr, ev_ebitda "
            "FROM valuation_factors WHERE stock_code=? AND calc_date=?",
            (code, calc_date),
        ).fetchone()

        holdings.append({
            "종목코드": code,
            "종목명": meta[0] if meta else code,
            "섹터": meta[1] if meta else "기타",
            "비중(%)": round(weights[i] * 100, 2),
            "점수": round(score, 1),
            "value_score": vf[0] if vf else None,
            "PER": round(vf[4], 1) if vf and vf[4] else None,
            "PBR": round(vf[5], 2) if vf and vf[5] else None,
            "EV/EBITDA": round(vf[6], 1) if vf and vf[6] else None,
            "시가총액": raw_mcaps[i],
        })

    conn.close()
    return pd.DataFrame(holdings)


@st.cache_data(ttl=3600)
def get_portfolio_characteristics(strategy: str, calc_date: str) -> dict:
    """전략 포트폴리오의 가중평균 밸류에이션 지표 계산."""
    df = get_holdings(strategy, calc_date)
    if df.empty:
        return {}
    result = {}
    for metric in ["PER", "PBR", "EV/EBITDA"]:
        valid = df.dropna(subset=[metric, "비중(%)"])
        if valid.empty:
            result[metric] = None
            continue
        w = valid["비중(%)"].values
        v = valid[metric].values
        w_sum = w.sum()
        result[metric] = round(float((w * v).sum() / w_sum), 2) if w_sum > 0 else None
    return result


@st.cache_data(ttl=3600)
def get_portfolio_turnover(strategy: str, current_date: str, prev_date: str) -> dict:
    """이전 리밸런싱 대비 편입/편출/유지 종목 비교."""
    curr_df = get_holdings(strategy, current_date)
    prev_df = get_holdings(strategy, prev_date)

    curr_codes = set(curr_df["종목코드"]) if not curr_df.empty else set()
    prev_codes = set(prev_df["종목코드"]) if not prev_df.empty else set()

    added_codes = curr_codes - prev_codes
    removed_codes = prev_codes - curr_codes
    retained_codes = curr_codes & prev_codes

    cols = ["종목코드", "종목명", "섹터", "비중(%)"]
    added = curr_df[curr_df["종목코드"].isin(added_codes)][cols].copy() if added_codes and not curr_df.empty else pd.DataFrame()
    removed = prev_df[prev_df["종목코드"].isin(removed_codes)][cols].copy() if removed_codes and not prev_df.empty else pd.DataFrame()

    total = max(len(curr_codes), 1)
    turnover_rate = (len(added_codes) + len(removed_codes)) / (2 * total)

    return {
        "added": added, "removed": removed,
        "added_count": len(added_codes),
        "removed_count": len(removed_codes),
        "retained_count": len(retained_codes),
        "turnover_rate": turnover_rate,
    }


@st.cache_data(ttl=3600)
def get_monthly_attribution(strategy: str, start_date: str, end_date: str) -> pd.DataFrame:
    """특정 월의 종목별 수익률 기여도를 계산한다.

    Parameters:
        strategy: 전략 키 ("A0", "ATT2" 등)
        start_date: 리밸런싱 시작일 (해당 월 첫 거래일)
        end_date: 리밸런싱 종료일 (다음 월 첫 거래일)

    Returns:
        DataFrame with columns: 종목명, 섹터, 비중(%), 종목수익률(%), 기여도(%), 기여방향
    """
    from step7_backtest import _apply_mcap_cap

    conn = _get_conn()
    stocks = _get_strategy_stocks(conn, strategy, start_date,
                                   BACKTEST_CONFIG.get("top_n_stocks", 30))
    if not stocks:
        conn.close()
        return pd.DataFrame()

    # 시총 비중 계산
    raw_mcaps = []
    for code, _ in stocks:
        row = conn.execute(
            "SELECT market_cap FROM daily_price "
            "WHERE stock_code=? AND trade_date>=? ORDER BY trade_date ASC LIMIT 1",
            (code, start_date),
        ).fetchone()
        raw_mcaps.append(row[0] if row and row[0] else 0)

    cap = BACKTEST_CONFIG.get("weight_cap_pct", 15) / 100
    weights = _apply_mcap_cap(raw_mcaps, cap=cap)

    rows = []
    for i, (code, _) in enumerate(stocks):
        sp = conn.execute(
            "SELECT close FROM daily_price "
            "WHERE stock_code=? AND trade_date>=? ORDER BY trade_date ASC LIMIT 1",
            (code, start_date),
        ).fetchone()
        ep = conn.execute(
            "SELECT close FROM daily_price "
            "WHERE stock_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
            (code, end_date),
        ).fetchone()

        if not sp or sp[0] <= 0:
            continue
        ret = ((ep[0] - sp[0]) / sp[0]) if (ep and ep[0] > 0) else -1.0
        contribution = ret * weights[i] * 100  # 기여도(%)

        meta = conn.execute(
            "SELECT sm.stock_name, COALESCE(fm.sec_cd_nm, sm.sector, '기타') "
            "FROM stock_master sm "
            "LEFT JOIN fnspace_master fm ON 'A'||sm.stock_code = fm.stock_code "
            "WHERE sm.stock_code=?",
            (code,),
        ).fetchone()

        rows.append({
            "종목명": meta[0] if meta else code,
            "섹터": (meta[1] if meta else "기타").replace("코스피 ", ""),
            "비중(%)": round(weights[i] * 100, 1),
            "종목수익률(%)": round(ret * 100, 1),
            "기여도(%)": round(contribution, 2),
        })

    conn.close()
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("기여도(%)", ascending=True)
    return df


@st.cache_data(ttl=3600)
def get_overlap_matrix(calc_date: str, top_n: int = 30) -> pd.DataFrame:
    """Compute pairwise stock overlap between strategies."""
    from step7_backtest import get_portfolio_stocks

    conn = _get_conn()
    sets = {}
    for label, code in _STRAT_CODE.items():
        stocks = get_portfolio_stocks(conn, calc_date, code, top_n)
        sets[label] = set(c for c, _ in stocks)
    conn.close()

    labels = list(_STRAT_CODE.keys())
    matrix = []
    for a in labels:
        row = [len(sets[a] & sets[b]) for b in labels]
        matrix.append(row)

    return pd.DataFrame(matrix, index=labels, columns=labels)


@st.cache_data(ttl=3600)
def get_stock_comparison(calc_date: str, top_n: int = 30):
    """기존전략 vs 회귀only 종목 비교: 공통/기존전략단독/회귀only단독 반환."""
    a0_label = STRATEGY_LABELS["A0"]
    att2_label = STRATEGY_LABELS["ATT2"]

    a0_df = get_holdings("A0", calc_date, top_n)
    att2_df = get_holdings("ATT2", calc_date, top_n)

    if a0_df.empty or att2_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    a0_codes = set(a0_df["종목코드"])
    att2_codes = set(att2_df["종목코드"])
    common_codes = a0_codes & att2_codes
    a0_only_codes = a0_codes - att2_codes
    att2_only_codes = att2_codes - a0_codes

    # 공통 종목: 양쪽 비중 모두 표시
    if common_codes:
        common = a0_df[a0_df["종목코드"].isin(common_codes)].copy()
        att2_weights = att2_df.set_index("종목코드")["비중(%)"]
        att2_scores = att2_df.set_index("종목코드")["점수"]
        common = common.rename(columns={"비중(%)": f"{a0_label} 비중(%)", "점수": f"{a0_label} 점수"})
        common[f"{att2_label} 비중(%)"] = common["종목코드"].map(att2_weights)
        common[f"{att2_label} 점수"] = common["종목코드"].map(att2_scores)
        common = common[["종목코드", "종목명", "섹터", f"{a0_label} 비중(%)", f"{a0_label} 점수",
                          f"{att2_label} 비중(%)", f"{att2_label} 점수", "PER", "PBR", "EV/EBITDA"]]
    else:
        common = pd.DataFrame()

    a0_only = a0_df[a0_df["종목코드"].isin(a0_only_codes)].copy() if a0_only_codes else pd.DataFrame()
    att2_only = att2_df[att2_df["종목코드"].isin(att2_only_codes)].copy() if att2_only_codes else pd.DataFrame()

    return common, a0_only, att2_only


# ═══════════════════════════════════════════════════════
# Custom backtest (parameter changes only)
# ═══════════════════════════════════════════════════════

def run_custom_backtest(top_n: int = 30, tx_cost_bp: int = 30, weight_cap: int = 15):
    """커스텀 파라미터로 A0 전략 백테스트 실행."""
    from step7_backtest import (
        run_backtest, calc_all_benchmarks, get_monthly_rebalance_dates,
        get_db, _numpy_to_python,
    )

    orig = {
        "top_n_stocks": BACKTEST_CONFIG["top_n_stocks"],
        "transaction_cost_bp": BACKTEST_CONFIG["transaction_cost_bp"],
        "weight_cap_pct": BACKTEST_CONFIG.get("weight_cap_pct", 15),
    }
    try:
        BACKTEST_CONFIG["top_n_stocks"] = top_n
        BACKTEST_CONFIG["transaction_cost_bp"] = tx_cost_bp
        BACKTEST_CONFIG["weight_cap_pct"] = weight_cap

        result = run_backtest("A0")
        results = {}
        if result:
            result["strategy"] = "A0"
            results["A0"] = result

        conn = get_db()
        rb_dates = get_monthly_rebalance_dates(conn)
        if len(rb_dates) >= 2:
            bm = calc_all_benchmarks(conn, rb_dates)
            results.update(bm)
        conn.close()

        return _numpy_to_python(results)
    finally:
        BACKTEST_CONFIG["top_n_stocks"] = orig["top_n_stocks"]
        BACKTEST_CONFIG["transaction_cost_bp"] = orig["transaction_cost_bp"]
        BACKTEST_CONFIG["weight_cap_pct"] = orig["weight_cap_pct"]


# ═══════════════════════════════════════════════════════
# Strategy save / load  (strategy.py + meta.json 기반)
# ═══════════════════════════════════════════════════════

_STRATEGIES_DIR = CACHE_DIR / "strategies"


def save_strategy(
    name: str,
    code: str,
    description: str = "",
    results: dict = None,
):
    """
    커스텀 전략 저장.
    cache/strategies/{name}/strategy.py + meta.json 형태.
    """
    strat_dir = _STRATEGIES_DIR / name
    strat_dir.mkdir(parents=True, exist_ok=True)

    # strategy.py 저장
    (strat_dir / "strategy.py").write_text(code, encoding="utf-8")

    # meta.json 저장 (기존 메타 있으면 병합)
    meta_path = strat_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass

    meta.update({
        "name": name,
        "description": description,
        "updated_at": datetime.now().isoformat(),
    })
    if "created_at" not in meta:
        meta["created_at"] = meta["updated_at"]
    if results is not None:
        # numpy 타입 직렬화
        from step7_backtest import _numpy_to_python
        meta["results"] = _numpy_to_python(results)

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def load_strategy(name: str) -> dict:
    """
    저장된 전략 불러오기.
    Returns: {"name", "code", "description", "created_at", "results", ...}
    """
    strat_dir = _STRATEGIES_DIR / name
    code_path = strat_dir / "strategy.py"
    meta_path = strat_dir / "meta.json"

    if not code_path.exists():
        return {}

    result = {"name": name, "code": code_path.read_text(encoding="utf-8")}

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            result.update(meta)
        except Exception:
            pass

    return result


def list_strategies() -> list:
    """저장된 전략 목록 반환."""
    if not _STRATEGIES_DIR.exists():
        return []
    items = []
    for d in sorted(_STRATEGIES_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        code_path = d / "strategy.py"
        if not code_path.exists():
            continue

        meta = {"name": d.name}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass

        summary = {}
        r = meta.get("results", {})
        if r:
            summary = {
                "CAGR": f"{r.get('cagr', 0):+.1%}",
                "Sharpe": f"{r.get('sharpe', 0):.2f}",
            }

        items.append({
            "name": meta.get("name", d.name),
            "created_at": meta.get("created_at", ""),
            "description": meta.get("description", ""),
            "summary": summary,
        })
    return items


def delete_strategy(name: str):
    """저장된 전략 삭제."""
    strat_dir = _STRATEGIES_DIR / name
    if strat_dir.exists():
        shutil.rmtree(strat_dir)


# ═══════════════════════════════════════════════════════
# Strategy backtest (factor_engine 기반)
# ═══════════════════════════════════════════════════════

def run_strategy_backtest(strategy_code: str, progress_callback=None) -> dict | None:
    """
    커스텀 전략 코드로 백테스트를 실행한다.

    1. 코드 검증
    2. 코드를 모듈로 변환
    3. score_stocks_from_strategy를 stock_selector로 사용
    4. step7 run_backtest 호출
    5. 벤치마크(KOSPI) 결과 포함하여 반환
    """
    from step7_backtest import (
        run_backtest, calc_all_benchmarks,
        get_monthly_rebalance_dates, get_db, _numpy_to_python,
    )

    # 1. 검증
    is_valid, err = validate_strategy_code(strategy_code)
    if not is_valid:
        return {"error": err}

    # 2. 모듈 변환
    strategy_module = code_to_module(strategy_code)

    # 3. 전략에서 파라미터 추출
    params = getattr(strategy_module, "PARAMS", {})
    top_n = params.get("top_n", 30)
    tx_cost_bp = params.get("tx_cost_bp", 30)
    weight_cap_pct = params.get("weight_cap_pct", 15)

    # stock_selector 콜백 생성 (get_portfolio_stocks와 동일한 유동성+생존 필터 적용)
    def stock_selector(conn, calc_date, _top_n):
        candidates = score_stocks_from_strategy(conn, calc_date, strategy_module)
        filtered = []
        for code, score in candidates:
            if len(filtered) >= _top_n:
                break
            price_exists = conn.execute("""
                SELECT COUNT(*) FROM daily_price
                WHERE stock_code = ? AND trade_date >= date(?, '-5 days')
                  AND trade_date <= ?
            """, (code, calc_date, calc_date)).fetchone()[0]
            if price_exists == 0:
                continue
            vol_data = conn.execute("""
                SELECT AVG(close * volume) as avg_trade_amount
                FROM daily_price
                WHERE stock_code = ? AND trade_date <= ?
                  AND trade_date >= date(?, '-30 days')
            """, (code, calc_date, calc_date)).fetchone()
            if vol_data and vol_data[0] and vol_data[0] >= 100_000_000:
                filtered.append((code, score))
        return filtered

    # 4. BACKTEST_CONFIG 임시 변경 후 실행
    orig = {
        "top_n_stocks": BACKTEST_CONFIG["top_n_stocks"],
        "transaction_cost_bp": BACKTEST_CONFIG["transaction_cost_bp"],
        "weight_cap_pct": BACKTEST_CONFIG.get("weight_cap_pct", 15),
    }
    try:
        BACKTEST_CONFIG["top_n_stocks"] = top_n
        BACKTEST_CONFIG["transaction_cost_bp"] = tx_cost_bp
        BACKTEST_CONFIG["weight_cap_pct"] = weight_cap_pct

        result = run_backtest(
            "custom",
            stock_selector=stock_selector,
            progress_callback=progress_callback,
        )

        results = {}
        if result:
            result["strategy"] = "CUSTOM"
            results["CUSTOM"] = result

        # 벤치마크
        conn = get_db()
        rb_dates = get_monthly_rebalance_dates(conn)
        if len(rb_dates) >= 2:
            bm = calc_all_benchmarks(conn, rb_dates)
            results.update(bm)
        conn.close()

        # factor_engine 캐시 정리 (메모리 관리)
        clear_factor_cache()

        return _numpy_to_python(results)
    finally:
        BACKTEST_CONFIG["top_n_stocks"] = orig["top_n_stocks"]
        BACKTEST_CONFIG["transaction_cost_bp"] = orig["transaction_cost_bp"]
        BACKTEST_CONFIG["weight_cap_pct"] = orig["weight_cap_pct"]
