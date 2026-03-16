"""
Dashboard data layer: cached loaders + constants.
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ─── Path setup ───
ALPHA_LAB_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ALPHA_LAB_DIR))
sys.path.insert(0, str(ALPHA_LAB_DIR / "scripts"))

from config.settings import BACKTEST_CONFIG, CACHE_DIR
from lib.db import get_conn as _get_conn_raw


def _get_conn():
    """DB 커넥션 반환. 연결 실패 시 None 대신 예외를 st.error로 표시."""
    try:
        return _get_conn_raw()
    except Exception as e:
        st.error(f"DB 연결 실패: {e}. 이 탭은 DB 연결이 필요합니다.")
        st.stop()
from lib.factor_engine import (
    validate_strategy_code, code_to_module, score_stocks_from_strategy,
    DEFAULT_STRATEGY_CODE, clear_factor_cache,
)

# ─── Strategy constants (기본 전략) ───
BASE_STRATEGY_KEYS = ["A0"]
STRATEGY_KEYS = ["A0"]  # 동적으로 갱신됨
ALL_KEYS = ["A0", "KOSPI"]  # 동적으로 갱신됨

STRATEGY_LABELS = {
    "A0":    "기존전략",
    "KOSPI": "KODEX 200",
}

STRATEGY_COLORS = {
    "A0":    "#42A5F5",   # 밝은 파랑
    "KOSPI": "#90A4AE",   # 회색
}

# 삭제된 전략 — 캐시에 남아있을 수 있으므로 로딩 시 필터링
_REMOVED_STRATEGIES = {"A", "A+M", "VM", "ATT2"}

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

    # 벤치마크 라벨: 결과의 strategy 필드에서 직접 읽기 (KRX 300 / KOSPI 200)
    bm_result = results.get("KOSPI", {})
    if bm_result:
        STRATEGY_LABELS["KOSPI"] = bm_result.get("strategy", "KODEX 200")

    for i, key in enumerate(custom_keys):
        if key not in STRATEGY_LABELS:
            STRATEGY_LABELS[key] = key
        if key not in STRATEGY_COLORS:
            STRATEGY_COLORS[key] = _CUSTOM_PALETTE[i % len(_CUSTOM_PALETTE)]

# step7 uses these internal codes
_STRAT_CODE = {"A0": "A0"}

# 기본 전략 코드 (factor_engine 파이프라인용)
_BASE_STRATEGY_CODES = {
    "A0": DEFAULT_STRATEGY_CODE,
}

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
        "weights_small": {},
        "regression_models": ["pbr_roe", "evic_roic", "fper_epsg", "fevebit_ebitg"],
        "scoring": {"large": "quartile"},
        "large_only": [],
        "small_only": [],
    },
}


# ─── DB helper (lib.db에서 import) ───
# _get_conn = lib.db.get_conn (상단에서 import 완료)


@st.cache_data(ttl=3600)
def get_latest_price_date() -> str | None:
    """최신 거래일 추출. PG 우선 → JSON fallback."""
    # 1) PG에서 조회
    try:
        conn = _get_conn_raw()
        row = conn.execute("""
            SELECT results_json FROM backtest_cache
            WHERE name = 'A0' AND universe = 'KOSPI' AND rebal_type = 'monthly'
        """, ()).fetchone()
        conn.close()
        if row and row[0]:
            dates = row[0].get("rebalance_dates", [])
            if dates:
                return dates[-1]
    except Exception:
        pass

    # 2) JSON fallback
    for path in [_combo_backtest_path("KOSPI", "monthly"), _BACKTEST_CACHE]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for v in data.get("results", {}).values():
                    dates = v.get("rebalance_dates", [])
                    if dates:
                        return dates[-1]
            except Exception:
                pass
    return None


def _get_strategy_stocks(conn, strategy: str, calc_date: str, top_n: int = 30,
                         rebal_type: str = "monthly", min_market_cap: float = 0):
    """모든 전략을 factor_engine 파이프라인으로 종목 선정.

    universe 테이블과 교집합하여 유동성/생존 필터를 대체한다.
    """
    from step7_backtest import get_universe_stocks

    # 전략 코드 결정
    if strategy in _BASE_STRATEGY_CODES:
        code = _BASE_STRATEGY_CODES[strategy]
    else:
        data = load_strategy(strategy)
        if not data or "code" not in data:
            return []
        code = data["code"]

    strategy_module = code_to_module(code)
    all_stocks = score_stocks_from_strategy(conn, calc_date, strategy_module)

    # universe 테이블과 교집합 (유동성/생존 필터 대체)
    universe_set = get_universe_stocks(conn, calc_date, rebal_type, min_market_cap)
    filtered = [(code_str, score) for code_str, score in all_stocks
                if code_str in universe_set][:top_n]

    return filtered


# ═══════════════════════════════════════════════════════
# Tier 1: Pre-computed cache (instant) → fallback to computation
# ═══════════════════════════════════════════════════════

_BACKTEST_CACHE = CACHE_DIR / "backtest_results.json"
_ROBUSTNESS_CACHE = CACHE_DIR / "robustness_results.json"


def _combo_cache_key(universe: str = None, rebal_type: str = None) -> str:
    u = (universe or "KOSPI").replace("+", "_")
    r = rebal_type or "monthly"
    return f"{u}_{r}"


def _combo_backtest_path(universe: str = None, rebal_type: str = None) -> Path:
    return CACHE_DIR / f"backtest_{_combo_cache_key(universe, rebal_type)}.json"


def _combo_holdings_path(universe: str = None, rebal_type: str = None) -> Path:
    return CACHE_DIR / f"holdings_{_combo_cache_key(universe, rebal_type)}.json"


def _combo_attribution_path(universe: str = None, rebal_type: str = None) -> Path:
    return CACHE_DIR / f"attribution_{_combo_cache_key(universe, rebal_type)}.json"


def _period_cache_path(start: str, end: str) -> Path:
    """기간별 캐시 파일 경로"""
    return CACHE_DIR / f"backtest_{start}_{end}.json"


def _is_default_params(weight_cap_pct: int = None, universe: str = None) -> bool:
    """파라미터가 기본값인지 확인"""
    default_cap = BACKTEST_CONFIG.get("weight_cap_pct", 10)
    return (weight_cap_pct is None or weight_cap_pct == default_cap) and \
           (universe is None or universe == "KOSPI")


def _slice_period_single(val: dict, s: str, e: str) -> dict | None:
    """단일 전략 결과를 지정 기간으로 슬라이싱.
    데이터 구조: rebalance_dates(list), portfolio_values(list, 1:1),
                 monthly_returns(list, len=dates-1)
    """
    if not isinstance(val, dict):
        return None
    rb = val.get("rebalance_dates", [])
    if not rb:
        return dict(val)

    indices = [i for i, d in enumerate(rb) if s <= d <= e]
    if not indices:
        return None

    v = dict(val)
    pv = val.get("portfolio_values", [])
    mr = val.get("monthly_returns", [])

    v["rebalance_dates"] = [rb[i] for i in indices]

    if pv and len(pv) == len(rb):
        sliced_pv = [pv[i] for i in indices]
        base = sliced_pv[0] if sliced_pv[0] != 0 else 1.0
        v["portfolio_values"] = [p / base for p in sliced_pv]

    if mr and len(mr) == len(rb) - 1:
        mr_indices = [i for i in indices if i < len(mr)]
        v["monthly_returns"] = [mr[i] for i in mr_indices]

    ps = val.get("portfolio_sizes", [])
    if ps and len(ps) == len(rb):
        v["portfolio_sizes"] = [ps[i] for i in indices]

    # 통계 재계산
    new_pv = v.get("portfolio_values", [])
    new_mr = v.get("monthly_returns", [])
    if new_pv and len(new_pv) >= 2:
        v["total_return"] = new_pv[-1] / new_pv[0] - 1
        n_years = max(len(v["rebalance_dates"]) / 12, 0.5)
        tr = v["total_return"]
        v["cagr"] = (1 + tr) ** (1 / n_years) - 1 if tr > -1 else 0
    if new_mr:
        mr_arr = np.array(new_mr)
        v["sharpe"] = float(np.mean(mr_arr) / np.std(mr_arr) * np.sqrt(12)) if np.std(mr_arr) > 1e-8 else 0
        v["months"] = len(new_mr)
        v["avg_monthly_return"] = float(np.mean(mr_arr))
        v["monthly_std"] = float(np.std(mr_arr))
        cum_arr = np.cumprod(1 + mr_arr)
        peak = np.maximum.accumulate(cum_arr)
        dd = (cum_arr - peak) / peak
        v["mdd"] = float(np.min(dd))

    return v


def _slice_period_multi(results: dict, s: str, e: str) -> dict:
    """여러 전략 결과를 지정 기간으로 슬라이싱."""
    if s == BACKTEST_CONFIG["start"] and e == BACKTEST_CONFIG["end"]:
        return results
    sliced = {}
    for key, val in results.items():
        r = _slice_period_single(val, s, e)
        if r is not None:
            sliced[key] = r
    return sliced


@st.cache_data(show_spinner=False, ttl=3600)
def _load_backtest_cached(start: str, end: str, universe: str = None, rebal_type: str = None):
    """기본 백테스트 캐시 로더. PG backtest_cache 테이블 → JSON 파일 fallback."""
    _uni = universe or "KOSPI"
    _rt = rebal_type or "monthly"

    def _filter(results):
        return {k: v for k, v in results.items() if k not in _REMOVED_STRATEGIES}

    # 1) PG에서 results_json만 조회 (holdings_json 제외 → 속도 개선)
    try:
        conn = _get_conn_raw()
        rows = conn.execute("""
            SELECT name, results_json FROM backtest_cache
            WHERE universe = %s AND rebal_type = %s
              AND results_json IS NOT NULL
              AND name NOT IN ('__ROBUSTNESS__')
        """, (_uni, _rt)).fetchall()
        conn.close()
        if rows:
            full_results = {}
            for name, rj in rows:
                if isinstance(rj, dict):
                    rj.pop("holdings", None)
                    rj.pop("attribution", None)
                    full_results[name] = rj
            if full_results:
                return _slice_period_multi(_filter(full_results), start, end)
    except Exception:
        pass

    # 2) fallback: JSON 캐시 파일
    combo_path = _combo_backtest_path(universe, rebal_type)
    if combo_path.exists():
        full_results = _filter(json.loads(combo_path.read_text())["results"])
        return _slice_period_multi(full_results, start, end)
    if (_uni == "KOSPI") and (_rt == "monthly") and _BACKTEST_CACHE.exists():
        full_results = _filter(json.loads(_BACKTEST_CACHE.read_text())["results"])
        return _slice_period_multi(full_results, start, end)

    st.warning("백테스트 캐시가 없습니다. 파이프라인을 실행해 캐시를 생성하세요.")
    return {}


def _run_backtest_with_params(start: str, end: str, weight_cap_pct: int = None, universe: str = None):
    """비기본 파라미터용 실시간 계산 (캐시 없음)."""
    from step7_backtest import run_all_backtests
    from lib.factor_engine import clear_factor_cache

    # 유니버스/파라미터가 바뀌므로 팩터 캐시를 먼저 비움
    clear_factor_cache()

    original = {
        "start": BACKTEST_CONFIG["start"],
        "end": BACKTEST_CONFIG["end"],
        "weight_cap_pct": BACKTEST_CONFIG.get("weight_cap_pct", 10),
        "universe": BACKTEST_CONFIG.get("universe", "KOSPI"),
    }
    try:
        BACKTEST_CONFIG["start"] = start
        BACKTEST_CONFIG["end"] = end
        if weight_cap_pct is not None:
            BACKTEST_CONFIG["weight_cap_pct"] = weight_cap_pct
        if universe is not None:
            BACKTEST_CONFIG["universe"] = universe
        return run_all_backtests()
    finally:
        BACKTEST_CONFIG["start"] = original["start"]
        BACKTEST_CONFIG["end"] = original["end"]
        BACKTEST_CONFIG["weight_cap_pct"] = original["weight_cap_pct"]
        BACKTEST_CONFIG["universe"] = original["universe"]


def load_backtest_results(start: str = None, end: str = None,
                          weight_cap_pct: int = None, universe: str = None,
                          rebal_type: str = None):
    """백테스트 결과 로딩. 캐시가 있으면 캐시, 아니면 실시간 계산."""
    use_start = start or BACKTEST_CONFIG["start"]
    use_end = end or BACKTEST_CONFIG["end"]
    use_universe = universe or "KOSPI"
    use_rebal = rebal_type or "monthly"

    # 캐시 있으면 캐시 사용 (weight_cap_pct가 기본값이면)
    default_cap = BACKTEST_CONFIG.get("weight_cap_pct", 10)
    if weight_cap_pct is None or weight_cap_pct == default_cap:
        return _load_backtest_cached(use_start, use_end, use_universe, use_rebal)
    else:
        return _run_backtest_with_params(use_start, use_end, weight_cap_pct, universe)


def load_all_results(start: str = None, end: str = None,
                     weight_cap_pct: int = None, universe: str = None,
                     rebal_type: str = None) -> dict:
    """기본 백테스트(A0, KOSPI) + 저장된 커스텀 전략 결과를 병합하여 반환.

    저장된 전략 중 백테스트 결과가 있는 것만 포함한다.
    병합 후 STRATEGY_KEYS/ALL_KEYS/LABELS/COLORS를 동적으로 갱신한다.
    """
    results = dict(load_backtest_results(start, end, weight_cap_pct, universe, rebal_type))
    # _load_backtest_cached already loads ALL strategies from backtest_cache
    # (including custom ones), so no need to loop list_strategies separately.
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

    # 기본 기간이면 캐시 사용 (PG 우선 → JSON fallback)
    if use_start == default_start and use_end == default_end:
        # 1) PG
        try:
            conn = _get_conn_raw()
            row = conn.execute("""
                SELECT results_json FROM backtest_cache
                WHERE name = '__ROBUSTNESS__' AND universe = 'ALL' AND rebal_type = 'ALL'
            """, ()).fetchone()
            conn.close()
            if row and row[0]:
                data = row[0]
                for sig in data.get("stat", {}).get("bm_significance", {}).values():
                    if "boot_means" in sig:
                        sig["boot_means"] = np.array(sig["boot_means"])
                return data["is_oos"], data["stat"], data["rolling"]
        except Exception:
            pass
        # 2) JSON fallback
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
        make_engine_selector, _numpy_to_python

    # IS/OOS 분할
    is_results, oos_results = {}, {}
    original_start = BACKTEST_CONFIG["start"]
    original_end = BACKTEST_CONFIG["end"]

    for key, strat, _ in STRATEGIES:
        selector = make_engine_selector(key)
        try:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = start, is_end
            r = run_backtest(strat, stock_selector=selector)
            clear_factor_cache()
            if r:
                r["strategy"] = key
                is_results[key] = r
        finally:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = original_start, original_end

        try:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = oos_start, end
            r = run_backtest(strat, stock_selector=selector)
            clear_factor_cache()
            if r:
                r["strategy"] = key
                oos_results[key] = r
        finally:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = original_start, original_end

    # 벤치마크 IS/OOS — DB 없이 계산 불가, 빈 값 사용
    _bm_name = "KODEX 200"
    is_ret = None
    oos_ret = None

    baseline = next((k for k in ["A0"] if k in is_results), None)
    is_months = len(is_results.get(baseline, {}).get("monthly_returns", [])) if baseline else 1
    oos_months = len(oos_results.get(baseline, {}).get("monthly_returns", [])) if baseline else 1

    bm_results = {"is": {}, "oos": {}}
    if is_ret is not None:
        bm_results["is"]["KOSPI"] = {
            "total_return": is_ret,
            "cagr": ((1 + is_ret) ** (12.0 / max(is_months, 1)) - 1.0),
            "name": _bm_name,
        }
    if oos_ret is not None:
        bm_results["oos"]["KOSPI"] = {
            "total_return": oos_ret,
            "cagr": ((1 + oos_ret) ** (12.0 / max(oos_months, 1)) - 1.0),
            "name": _bm_name,
        }

    is_oos_data = {"is_results": is_results, "oos_results": oos_results, "benchmarks": bm_results}

    # 전체 기간 백테스트 + 통계 유의성
    full_results = {}
    for key, strat, _ in STRATEGIES:
        selector = make_engine_selector(key)
        try:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = start, end
            r = run_backtest(strat, stock_selector=selector)
            clear_factor_cache()
            if r:
                r["strategy"] = key
                full_results[key] = r
        finally:
            BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"] = original_start, original_end

    if not baseline or baseline not in full_results:
        baseline = next(iter(full_results), None)
    if not baseline:
        return is_oos_data, {"full_results": {}, "bm_significance": {}}, {}

    rb_dates = full_results[baseline]["rebalance_dates"]
    # 벤치마크 월별 수익률: 캐시에서 가져옴
    _cached_results = load_backtest_results(start, end)
    _bm_cached = _cached_results.get("KOSPI", {}).get("monthly_returns", [])
    bm_monthly = np.array(_bm_cached) if _bm_cached else np.zeros(len(rb_dates) - 1)

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
                                 is_end: str = None, oos_start: str = None,
                                 universe: str = None):
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
    baseline_key = next((k for k in ["A0"] if k in base_results), None)
    if not baseline_key:
        return is_oos_data, stat_data, rolling_all

    rb_dates = base_results[baseline_key].get("rebalance_dates", [])
    if not rb_dates:
        return is_oos_data, stat_data, rolling_all

    # 벤치마크 월별 수익률: 캐시의 KOSPI 결과에서 가져옴
    bm_result = base_results.get("KOSPI", {})
    bm_monthly_list = bm_result.get("monthly_returns", [])
    if not bm_monthly_list:
        return is_oos_data, stat_data, rolling_all
    bm_monthly = np.array(bm_monthly_list)

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
# Tier 2: 캐시 기반 포트폴리오 데이터
# ═══════════════════════════════════════════════════════

_HOLDINGS_CACHE = CACHE_DIR / "holdings_cache.json"
_ATTRIBUTION_CACHE = CACHE_DIR / "attribution_cache.json"


def _load_holdings_cache(universe: str = None, rebal_type: str = None) -> dict:
    _uni = universe or "KOSPI"
    _rt = rebal_type or "monthly"

    # 1) PG backtest_cache에서 holdings_json 조회
    try:
        conn = _get_conn_raw()
        rows = conn.execute("""
            SELECT name, holdings_json FROM backtest_cache
            WHERE universe = %s AND rebal_type = %s AND holdings_json IS NOT NULL
              AND name NOT IN ('KOSPI')
        """, (_uni, _rt)).fetchall()
        conn.close()
        if rows:
            result = {}
            for name, hj in rows:
                if isinstance(hj, dict) and "holdings" in hj:
                    result[name] = hj["holdings"]
                elif isinstance(hj, dict):
                    result[name] = hj
            if result:
                return result
    except Exception:
        pass

    # 2) fallback: JSON 캐시
    combo_path = _combo_holdings_path(universe, rebal_type)
    if combo_path.exists():
        return json.loads(combo_path.read_text()).get("data", {})
    if _uni == "KOSPI" and _rt == "monthly" and _HOLDINGS_CACHE.exists():
        return json.loads(_HOLDINGS_CACHE.read_text()).get("data", {})
    return {}


def _load_attribution_cache(universe: str = None, rebal_type: str = None) -> dict:
    _uni = universe or "KOSPI"
    _rt = rebal_type or "monthly"

    # 1) PG backtest_cache에서 attribution 조회
    try:
        conn = _get_conn_raw()
        rows = conn.execute("""
            SELECT name, holdings_json FROM backtest_cache
            WHERE universe = %s AND rebal_type = %s AND holdings_json IS NOT NULL
              AND name NOT IN ('KOSPI')
        """, (_uni, _rt)).fetchall()
        conn.close()
        if rows:
            result = {}
            for name, hj in rows:
                if isinstance(hj, dict) and "attribution" in hj:
                    result[name] = hj["attribution"]
            if result:
                return result
    except Exception:
        pass

    # 2) fallback: JSON 캐시
    combo_path = _combo_attribution_path(universe, rebal_type)
    if combo_path.exists():
        return json.loads(combo_path.read_text()).get("data", {})
    if _uni == "KOSPI" and _rt == "monthly" and _ATTRIBUTION_CACHE.exists():
        return json.loads(_ATTRIBUTION_CACHE.read_text()).get("data", {})
    return {}


@st.cache_data(ttl=3600)
def get_holdings(strategy: str, calc_date: str, top_n: int = 30,
                 universe: str = None, rebal_type: str = None) -> pd.DataFrame:
    """캐시에서 보유종목 데이터를 읽음. 저장 전략은 meta.json에서 로드."""
    # 기본 전략: holdings 캐시에서
    cache = _load_holdings_cache(universe, rebal_type)
    rows = cache.get(strategy, {}).get(calc_date, [])
    if rows:
        return pd.DataFrame(rows)
    # 저장 전략: meta.json의 holdings에서
    data = load_strategy(strategy)
    if data:
        rows = data.get("results", {}).get("holdings", {}).get(calc_date, [])
        if rows:
            return pd.DataFrame(rows)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_portfolio_characteristics(strategy: str, calc_date: str,
                                  universe: str = None, rebal_type: str = None) -> dict:
    """전략 포트폴리오의 가중평균 + 단순평균 밸류에이션 지표 계산."""
    df = get_holdings(strategy, calc_date, universe=universe, rebal_type=rebal_type)
    if df.empty:
        return {}
    result = {}
    for metric in ["PER", "PBR", "EV/EBITDA"]:
        valid = df.dropna(subset=[metric, "비중(%)"])
        if valid.empty:
            result[metric] = None
            result[f"{metric}_simple"] = None
            continue
        w = valid["비중(%)"].values
        v = valid[metric].values
        w_sum = w.sum()
        result[metric] = round(float((w * v).sum() / w_sum), 2) if w_sum > 0 else None
        result[f"{metric}_simple"] = round(float(v.mean()), 2)
    return result


@st.cache_data(ttl=3600)
def get_portfolio_turnover(strategy: str, current_date: str, prev_date: str,
                           universe: str = None, rebal_type: str = None) -> dict:
    """이전 리밸런싱 대비 편입/편출/유지 종목 비교."""
    curr_df = get_holdings(strategy, current_date, universe=universe, rebal_type=rebal_type)
    prev_df = get_holdings(strategy, prev_date, universe=universe, rebal_type=rebal_type)

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
def get_monthly_attribution(strategy: str, start_date: str, end_date: str,
                            universe: str = None, rebal_type: str = None) -> pd.DataFrame:
    """캐시에서 월별 기여도 데이터를 읽음."""
    cache = _load_attribution_cache(universe, rebal_type)
    key = f"{start_date}_{end_date}"
    rows = cache.get(strategy, {}).get(key, [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("기여도(%)", ascending=True)
    return df


@st.cache_data(ttl=3600)
def get_overlap_matrix(calc_date: str, top_n: int = 30,
                       universe: str = None, rebal_type: str = None) -> pd.DataFrame:
    """캐시에서 종목 오버랩 계산."""
    cache = _load_holdings_cache(universe, rebal_type)
    sets = {}
    for label in _STRAT_CODE:
        rows = cache.get(label, {}).get(calc_date, [])
        sets[label] = set(r["종목코드"] for r in rows)

    labels = list(_STRAT_CODE.keys())
    matrix = []
    for a in labels:
        row = [len(sets.get(a, set()) & sets.get(b, set())) for b in labels]
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
        get_db, make_engine_selector, _numpy_to_python,
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

        selector = make_engine_selector("A0")
        result = run_backtest("A0", stock_selector=selector)
        clear_factor_cache()
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
# Strategy save / load  (PG backtest_cache 테이블 기반)
# ═══════════════════════════════════════════════════════


def _pg_json(data):
    """psycopg2 Json 어댑터. PG JSONB 컬럼에 dict 저장용."""
    from psycopg2.extras import Json
    return Json(data)


def save_strategy(
    name: str,
    code: str,
    description: str = "",
    results: dict = None,
    universe: str = None,
    rebal_type: str = None,
):
    """커스텀 전략을 PG backtest_cache 테이블에 저장 (UPSERT)."""
    _uni = universe or "KOSPI"
    _rt = rebal_type or "monthly"
    conn = _get_conn_raw()

    results_json = None
    holdings_json = None
    if results is not None:
        from step7_backtest import _numpy_to_python
        clean = _numpy_to_python(results)
        holdings_json = clean.pop("holdings", None)
        results_json = clean

    try:
        conn.execute("""
            INSERT INTO backtest_cache (name, universe, rebal_type, strategy_code, description, results_json, holdings_json, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (name, universe, rebal_type)
            DO UPDATE SET
                strategy_code = EXCLUDED.strategy_code,
                description = EXCLUDED.description,
                results_json = COALESCE(EXCLUDED.results_json, backtest_cache.results_json),
                holdings_json = COALESCE(EXCLUDED.holdings_json, backtest_cache.holdings_json),
                updated_at = NOW()
        """, (name, _uni, _rt, code, description,
              _pg_json(results_json) if results_json else None,
              _pg_json(holdings_json) if holdings_json else None))
        conn.commit()
    finally:
        conn.close()


def load_strategy(name: str, rebal_type: str = None, universe: str = None) -> dict:
    """PG에서 저장된 전략 불러오기."""
    _rt = rebal_type or "monthly"
    _uni = universe or "KOSPI"
    conn = _get_conn_raw()
    try:
        row = conn.execute("""
            SELECT name, strategy_code, description, results_json, holdings_json,
                   universe, rebal_type, created_at
            FROM backtest_cache
            WHERE name = %s AND universe = %s AND rebal_type = %s
        """, (name, _uni, _rt)).fetchone()
        if not row:
            return {}
        result = {
            "name": row[0],
            "code": row[1] or "",
            "description": row[2] or "",
            "results": row[3] or {},
            "holdings": row[4],
            "universe": row[5],
            "rebal_type": row[6],
            "created_at": str(row[7]) if row[7] else "",
        }
        # holdings를 results 안에 넣기 (get_holdings 호환)
        if result["holdings"] and isinstance(result["results"], dict):
            result["results"]["holdings"] = result["holdings"]
        return result
    finally:
        conn.close()


def list_strategies(universe: str = None, rebal_type: str = None) -> list:
    """PG에서 저장된 전략 목록 반환."""
    conn = _get_conn_raw()
    try:
        conditions = ["name NOT IN ('A0', 'KOSPI', '__ROBUSTNESS__')"]
        params = []
        if universe:
            conditions.append("universe = %s")
            params.append(universe)
        if rebal_type:
            conditions.append("rebal_type = %s")
            params.append(rebal_type)

        where = " AND ".join(conditions)
        rows = conn.execute(f"""
            SELECT name, description, created_at, results_json
            FROM backtest_cache
            WHERE {where}
            ORDER BY updated_at DESC
        """, tuple(params)).fetchall()

        items = []
        for row in rows:
            r = row[3] or {}
            summary = {}
            if r:
                summary = {
                    "CAGR": f"{r.get('cagr', 0):+.1%}",
                    "Sharpe": f"{r.get('sharpe', 0):.2f}",
                }
            items.append({
                "name": row[0],
                "description": row[1] or "",
                "created_at": str(row[2]) if row[2] else "",
                "summary": summary,
            })
        return items
    finally:
        conn.close()


def delete_strategy(name: str):
    """PG에서 전략 삭제."""
    conn = _get_conn_raw()
    try:
        conn.execute("DELETE FROM backtest_cache WHERE name = %s", (name,))
        conn.commit()
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════
# Strategy backtest (factor_engine 기반)
# ═══════════════════════════════════════════════════════

def run_strategy_backtest(strategy_code: str, progress_callback=None, universe: str = None,
                          weight_cap_pct_override: int = None, tx_cost_bp_override: int = None,
                          rebal_type: str = None) -> dict | None:
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
    if weight_cap_pct_override is not None:
        weight_cap_pct = weight_cap_pct_override

    # stock_selector 콜백 생성 (universe 테이블 기반)
    from step7_backtest import get_universe_stocks

    def stock_selector(conn, calc_date, _top_n):
        universe_set = get_universe_stocks(conn, calc_date)
        candidates = score_stocks_from_strategy(conn, calc_date, strategy_module)
        filtered = [(c, s) for c, s in candidates if c in universe_set][:_top_n]
        return filtered

    # 4. BACKTEST_CONFIG 임시 변경 후 실행
    if tx_cost_bp_override is not None:
        tx_cost_bp = tx_cost_bp_override
    _rebal = rebal_type or BACKTEST_CONFIG.get("rebal_type", "monthly")
    orig = {
        "top_n_stocks": BACKTEST_CONFIG["top_n_stocks"],
        "transaction_cost_bp": BACKTEST_CONFIG["transaction_cost_bp"],
        "weight_cap_pct": BACKTEST_CONFIG.get("weight_cap_pct", 15),
        "universe": BACKTEST_CONFIG.get("universe", "KOSPI"),
        "rebal_type": BACKTEST_CONFIG.get("rebal_type", "monthly"),
    }
    try:
        BACKTEST_CONFIG["top_n_stocks"] = top_n
        BACKTEST_CONFIG["transaction_cost_bp"] = tx_cost_bp
        BACKTEST_CONFIG["weight_cap_pct"] = weight_cap_pct
        BACKTEST_CONFIG["rebal_type"] = _rebal
        if universe:
            BACKTEST_CONFIG["universe"] = universe

        result = run_backtest(
            "custom",
            stock_selector=stock_selector,
            rebal_type=_rebal,
            progress_callback=progress_callback,
        )

        results = {}
        if result:
            # holdings_by_date → 포트폴리오 데이터 변환
            hbd = result.pop("holdings_by_date", {})
            if hbd:
                conn2 = get_db()
                from lib.db import read_sql
                master_all = read_sql(
                    "SELECT stock_code, stock_name, COALESCE(sec_cd_nm, '기타') as sector, "
                    "snapshot_date FROM fnspace_master", conn2,
                )
                _master_by_snap = {}
                for r in master_all.itertuples():
                    _master_by_snap.setdefault(r.snapshot_date, {})[r.stock_code] = (r.stock_name, r.sector)

                holdings_dict = {}
                for date, items in hbd.items():
                    _snap = date[:7]
                    _snaps = sorted(s for s in _master_by_snap if s <= _snap)
                    name_map = _master_by_snap.get(_snaps[-1], {}) if _snaps else {}

                    # 밸류에이션 조회
                    codes = [c for c, _, _, _ in items]
                    if codes:
                        fin_rows = conn2.execute(f"""
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
                    else:
                        fin_map = {}

                    h_rows = []
                    for code, score, weight, mcap in items:
                        acode = f"A{code}"
                        nm = name_map.get(acode, (code, "기타"))
                        fin = fin_map.get(acode, (None, None, None))
                        h_rows.append({
                            "종목코드": code, "종목명": nm[0], "섹터": nm[1],
                            "비중(%)": round(weight * 100, 2),
                            "점수": round(score, 1), "value_score": round(score, 1),
                            "PER": round(fin[0], 1) if fin[0] else None,
                            "PBR": round(fin[1], 2) if fin[1] else None,
                            "EV/EBITDA": round(fin[2], 1) if fin[2] else None,
                            "시가총액": mcap,
                        })
                    holdings_dict[date] = h_rows

                result["holdings"] = holdings_dict
                conn2.close()

            result["strategy"] = "CUSTOM"
            results["CUSTOM"] = result

        # 벤치마크
        from step7_backtest import get_rebalance_dates
        conn = get_db()
        rb_dates = get_rebalance_dates(conn, _rebal)
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
        BACKTEST_CONFIG["universe"] = orig["universe"]
        BACKTEST_CONFIG["rebal_type"] = orig["rebal_type"]
