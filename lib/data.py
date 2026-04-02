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

# 커스텀 전략용 팔레트 (색상 대비 최대화)
_CUSTOM_PALETTE = [
    "#FF6B35",  # 진한 오렌지
    "#2ECC71",  # 초록
    "#9B59B6",  # 퍼플
    "#F1C40F",  # 노랑
    "#E74C3C",  # 빨강
    "#1ABC9C",  # 청록
    "#E67E22",  # 다크 오렌지
    "#3498DB",  # 파랑
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
    if ps and len(ps) == len(rb) - 1:
        ps_indices = [i for i in indices if i < len(ps)]
        v["portfolio_sizes"] = [ps[i] for i in ps_indices]

    # 통계 재계산 (날짜 기반 연환산 — 월간/격주 모두 정확)
    new_pv = v.get("portfolio_values", [])
    new_mr = v.get("monthly_returns", [])
    rb_dates = v.get("rebalance_dates", [])
    if new_pv and len(new_pv) >= 2 and len(rb_dates) >= 2:
        from datetime import datetime as _dt
        _d0 = _dt.strptime(rb_dates[0], "%Y-%m-%d")
        _d1 = _dt.strptime(rb_dates[-1], "%Y-%m-%d")
        n_years = max((_d1 - _d0).days / 365.25, 0.5)
        periods_per_year = len(new_mr) / n_years if n_years > 0 else 12

        v["total_return"] = new_pv[-1] / new_pv[0] - 1
        tr = v["total_return"]
        v["cagr"] = (1 + tr) ** (1 / n_years) - 1 if tr > -1 else 0
    else:
        periods_per_year = 12
    if new_mr:
        mr_arr = np.array(new_mr)
        v["sharpe"] = float(np.mean(mr_arr) / np.std(mr_arr) * np.sqrt(periods_per_year)) if np.std(mr_arr) > 1e-8 else 0
        v["months"] = len(new_mr)
        v["avg_monthly_return"] = float(np.mean(mr_arr))
        v["monthly_std"] = float(np.std(mr_arr))
        cum_arr = np.concatenate([[1.0], np.cumprod(1 + mr_arr)])
        peak = np.maximum.accumulate(cum_arr)
        dd = (cum_arr - peak) / peak
        v["mdd"] = float(abs(np.min(dd)))  # 양수로 통일 (step7과 일치)

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


def _load_backtest_cached(start: str, end: str, universe: str = None, rebal_type: str = None):
    """기본 백테스트 캐시 로더. PG backtest_cache 테이블 → JSON 파일 fallback.
    NOTE: st.cache_data 제거 — 커스텀 전략 추가/삭제가 즉시 반영되어야 하므로."""
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
                    # Custom strategies store results as {"KOSPI": {...}, "CUSTOM": {...}}
                    # Extract the CUSTOM sub-result if present
                    if "CUSTOM" in rj and "rebalance_dates" not in rj:
                        rj = rj["CUSTOM"]
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
    # Filter out strategies with incomplete results (missing rebalance_dates)
    results = {k: v for k, v in results.items()
               if isinstance(v, dict) and v.get("rebalance_dates")}
    _update_strategy_registry(results)
    return results


@st.cache_data(show_spinner=False)
def load_robustness_results(start: str = None, end: str = None,
                            is_end: str = None, oos_start: str = None,
                            rebal_type: str = None, universe: str = None):
    """강건성 검증 결과. 기간 지정 시 백테스트 결과에서 실시간 계산."""
    default_start = BACKTEST_CONFIG["start"]
    default_end = BACKTEST_CONFIG["end"]
    use_start = start or default_start
    use_end = end or default_end
    use_is_end = is_end or BACKTEST_CONFIG.get("insample_end", "2024-06-30")
    use_oos_start = oos_start or BACKTEST_CONFIG.get("oos_start", "2024-07-01")
    use_rebal = rebal_type or BACKTEST_CONFIG.get("rebal_type", "monthly")
    use_universe = universe or "KOSPI"

    # 기본 기간+기본 rebal_type+기본 universe이면 캐시 사용
    if use_start == default_start and use_end == default_end and use_rebal == "monthly" and use_universe == "KOSPI":
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

    # 커스텀 기간/rebal_type/universe → 캐시된 백테스트 결과에서 IS/OOS 슬라이싱
    return _compute_robustness_from_cache(use_start, use_end, use_is_end, use_oos_start, use_rebal, use_universe)


def _compute_robustness_from_cache(start, end, is_end, oos_start, rebal_type="monthly", universe="KOSPI"):
    """캐시된 백테스트 결과를 IS/OOS로 슬라이싱해서 강건성 계산 (빠름)."""
    from scipy import stats as sp_stats

    # 전체 기간 백테스트 결과 로드
    full_results = load_backtest_results(start, end, rebal_type=rebal_type, universe=universe)
    is_results = _slice_period_multi(full_results, start, is_end) if full_results else {}
    oos_results = _slice_period_multi(full_results, oos_start, end) if full_results else {}

    # IS/OOS 데이터 구성
    is_oos_data = {"is_results": {}, "oos_results": {}, "benchmarks": {}}
    for key, r in is_results.items():
        if isinstance(r, dict) and "cagr" in r:
            is_oos_data["is_results"][key] = r
    for key, r in oos_results.items():
        if isinstance(r, dict) and "cagr" in r:
            is_oos_data["oos_results"][key] = r

    # 통계 데이터
    stat_data = {"full_results": {}, "bm_significance": {}}
    for key, r in full_results.items():
        if isinstance(r, dict) and "cagr" in r:
            stat_data["full_results"][key] = r

    # BM 대비 유의성 (부트스트랩)
    baseline_key = next((k for k in ["A0"] if k in full_results), None)
    bm_key = next((k for k in ["KOSPI", "BM"] if k in full_results), None)
    if baseline_key and bm_key:
        bl = full_results[baseline_key]
        bm = full_results[bm_key]
        bl_mr = np.array(bl.get("monthly_returns", []))
        bm_mr = np.array(bm.get("monthly_returns", []))
        if len(bl_mr) > 0 and len(bm_mr) > 0:
            min_len = min(len(bl_mr), len(bm_mr))
            excess = bl_mr[:min_len] - bm_mr[:min_len]
            n_boot = 5000
            boot_means = np.array([
                np.mean(np.random.choice(excess, size=len(excess), replace=True))
                for _ in range(n_boot)
            ])
            stat_data["bm_significance"][baseline_key] = {
                "mean_excess": float(np.mean(excess)),
                "p_value": float((boot_means < 0).sum() / n_boot),
                "ci_lower": float(np.percentile(boot_means, 2.5)),
                "ci_upper": float(np.percentile(boot_means, 97.5)),
            }

    # 롤링 윈도우
    rolling_all = {}
    if baseline_key:
        bl = full_results[baseline_key]
        bl_mr = np.array(bl.get("monthly_returns", []))
        bm_mr = np.array(full_results.get(bm_key, {}).get("monthly_returns", [])) if bm_key else np.array([])
        min_len = min(len(bl_mr), len(bm_mr)) if len(bm_mr) > 0 else len(bl_mr)
        if min_len >= 24:
            dates = bl.get("rebalance_dates", [])[1:min_len+1]
            rolling_excess = []
            for i in range(min_len - 23):
                window = bl_mr[i:i+24] - bm_mr[i:i+24] if len(bm_mr) >= min_len else bl_mr[i:i+24]
                cum = float(np.prod(1 + window) - 1)
                rolling_excess.append(cum)
            rolling_all[baseline_key] = {
                "dates": dates[23:] if len(dates) > 23 else [],
                "rolling_24m_excess": rolling_excess,
            }

    return is_oos_data, stat_data, rolling_all


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
    # 날짜 기반 연환산 (is_ret/oos_ret은 현재 None이지만 추후 사용 대비)
    from datetime import datetime as _dt
    _is_nyears = max((_dt.strptime(is_end, "%Y-%m-%d") - _dt.strptime(start, "%Y-%m-%d")).days / 365.25, 0.5)
    _oos_nyears = max((_dt.strptime(end, "%Y-%m-%d") - _dt.strptime(oos_start, "%Y-%m-%d")).days / 365.25, 0.5)

    bm_results = {"is": {}, "oos": {}}
    if is_ret is not None:
        bm_results["is"]["KOSPI"] = {
            "total_return": is_ret,
            "cagr": ((1 + is_ret) ** (1.0 / _is_nyears) - 1.0),
            "name": _bm_name,
        }
    if oos_ret is not None:
        bm_results["oos"]["KOSPI"] = {
            "total_return": oos_ret,
            "cagr": ((1 + oos_ret) ** (1.0 / _oos_nyears) - 1.0),
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
                                 universe: str = None, rebal_type: str = None):
    """기본 전략 강건성 + 커스텀 전략 강건성 (저장된 백테스트 결과 기반)."""
    is_oos_data, stat_data, rolling_all = load_robustness_results(
        start, end, is_end, oos_start, rebal_type=rebal_type, universe=universe,
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

    _rebal = rebal_type or BACKTEST_CONFIG.get("rebal_type", "monthly")
    _uni = universe or "KOSPI"
    base_results = load_backtest_results(use_start, use_end, rebal_type=_rebal, universe=_uni)
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

    _uni_filter = universe or "KOSPI"
    for strat_info in list_strategies(universe=_uni_filter, rebal_type=_rebal):
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

        # 날짜 기반 연환산 계수 계산
        from datetime import datetime as _dt
        _all_dates = strat_rb_dates
        if len(_all_dates) >= 2:
            _total_days = (_dt.strptime(_all_dates[-1], "%Y-%m-%d") - _dt.strptime(_all_dates[0], "%Y-%m-%d")).days
            _ppy = len(monthly_rets) / max(_total_days / 365.25, 0.5)  # periods per year
        else:
            _ppy = 12

        if len(is_rets) > 0:
            is_cum = float(np.prod(1 + is_rets) - 1)
            n_is = len(is_rets)
            _is_years = max(n_is / _ppy, 0.5)
            cum_arr = np.cumprod(1 + is_rets)
            peak = np.maximum.accumulate(cum_arr)
            dd = (cum_arr - peak) / peak
            is_oos_data["is_results"][name] = {
                "strategy": name,
                "total_return": is_cum,
                "cagr": float((1 + is_cum) ** (1.0 / _is_years) - 1),
                "sharpe": float(is_rets.mean() / is_rets.std() * np.sqrt(_ppy))
                    if n_is > 1 and is_rets.std() > 0 else 0,
                "mdd": float(dd.min()),
                "monthly_returns": is_rets.tolist(),
            }

        if len(oos_rets) > 0:
            oos_cum = float(np.prod(1 + oos_rets) - 1)
            n_oos = len(oos_rets)
            _oos_years = max(n_oos / _ppy, 0.5)
            cum_arr = np.cumprod(1 + oos_rets)
            peak = np.maximum.accumulate(cum_arr)
            dd = (cum_arr - peak) / peak
            is_oos_data["oos_results"][name] = {
                "strategy": name,
                "total_return": oos_cum,
                "cagr": float((1 + oos_cum) ** (1.0 / _oos_years) - 1),
                "sharpe": float(oos_rets.mean() / oos_rets.std() * np.sqrt(_ppy))
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
                    # 튜플 형식이든 dict 형식이든 그대로 저장
                    # (변환은 get_holdings에서 선택된 날짜만 수행)
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
def _resolve_stock_names(codes: list[str]) -> dict[str, str]:
    """PG fnspace_master에서 종목코드 → 종목명 매핑 (A 접두사 자동 처리)."""
    if not codes:
        return {}
    conn = _get_conn_raw()
    try:
        a_codes = [f"A{c}" if not c.startswith("A") else c for c in codes]
        ph = ",".join(["%s"] * len(a_codes))
        rows = conn.execute(
            f"SELECT DISTINCT ON (stock_code) stock_code, stock_name "
            f"FROM fnspace_master WHERE stock_code IN ({ph}) ORDER BY stock_code",
            tuple(a_codes),
        ).fetchall()
        # A 접두사 제거해서 반환
        return {r[0].lstrip("A"): r[1] for r in rows}
    finally:
        conn.close()


def _convert_raw_holdings_bulk(hbd: dict) -> dict:
    """holdings_by_date 전체를 일괄 변환. 종목명/섹터/밸류에이션 한 번만 조회."""
    # 전체 종목코드 수집
    all_codes = set()
    for items in hbd.values():
        for r in items:
            all_codes.add(r[0])
    codes_list = list(all_codes)
    if not codes_list:
        return {}

    names = _resolve_stock_names(codes_list)
    a_codes = [f"A{c}" if not c.startswith("A") else c for c in codes_list]
    ph = ",".join(["%s"] * len(a_codes))

    # 섹터 조회 (PG fnspace_master)
    sector_map: dict[str, str] = {}
    conn = _get_conn_raw()
    try:
        sec_rows = conn.execute(
            f"SELECT DISTINCT ON (stock_code) stock_code, COALESCE(sec_cd_nm, '기타') "
            f"FROM fnspace_master WHERE stock_code IN ({ph}) ORDER BY stock_code",
            tuple(a_codes),
        ).fetchall()
        sector_map = {r[0].lstrip("A"): r[1] for r in sec_rows}
    finally:
        conn.close()

    # 밸류에이션 조회 - 최신 날짜 기준 1회만 (PG fnspace_finance Annual)
    from datetime import datetime as _dt
    latest_dt = max(hbd.keys())
    val_map: dict[str, dict] = {}
    conn2 = _get_conn_raw()
    try:
        _d = _dt.strptime(latest_dt[:10], "%Y-%m-%d")
        _max_fy = _d.year - 1 if _d.month >= 4 else _d.year - 2
        val_rows = conn2.execute(
            f"SELECT ff.stock_code, ff.per, ff.pbr, ff.ev_ebitda "
            f"FROM fnspace_finance ff "
            f"INNER JOIN ("
            f"  SELECT stock_code, MAX(fiscal_year) as my FROM fnspace_finance "
            f"  WHERE fiscal_quarter='Annual' AND fiscal_year <= %s AND stock_code IN ({ph}) "
            f"  GROUP BY stock_code"
            f") t ON ff.stock_code = t.stock_code AND ff.fiscal_year = t.my "
            f"AND ff.fiscal_quarter = 'Annual'",
            (_max_fy, *tuple(a_codes)),
        ).fetchall()
        for vr in val_rows:
            val_map[vr[0].lstrip("A")] = {
                "PER": round(vr[1], 1) if vr[1] else None,
                "PBR": round(vr[2], 2) if vr[2] else None,
                "EV/EBITDA": round(vr[3], 1) if vr[3] else None,
            }
    except Exception:
        pass
    finally:
        conn2.close()

    converted = {}
    for dt, items in hbd.items():
        vm = val_map
        converted[dt] = [
            {"종목코드": r[0], "종목명": names.get(r[0], r[0]), "섹터": sector_map.get(r[0], ""),
             "점수": r[1], "비중(%)": round(r[2] * 100, 2), "시가총액": r[3],
             "PER": vm.get(r[0], {}).get("PER"),
             "PBR": vm.get(r[0], {}).get("PBR"),
             "EV/EBITDA": vm.get(r[0], {}).get("EV/EBITDA")}
            for r in items
        ]
    return converted


def _convert_raw_holdings(rows: list, calc_date: str = None) -> list[dict]:
    """[code, score, weight, mcap] 튜플 리스트 → dict 리스트 변환 (PG에서 종목명/섹터/밸류 조회)."""
    codes = [r[0] for r in rows]
    names = _resolve_stock_names(codes)

    # 밸류에이션 조회 (PG fnspace_finance Annual, look-ahead bias 방지)
    val_map: dict[str, dict] = {}
    if calc_date:
        conn = _get_conn_raw()
        try:
            from datetime import datetime as _dt
            _d = _dt.strptime(calc_date[:10], "%Y-%m-%d")
            _max_fy = _d.year - 1 if _d.month >= 4 else _d.year - 2
            a_codes = [f"A{c}" if not c.startswith("A") else c for c in codes]
            ph = ",".join(["%s"] * len(a_codes))
            val_rows = conn.execute(
                f"SELECT ff.stock_code, ff.per, ff.pbr, ff.ev_ebitda "
                f"FROM fnspace_finance ff "
                f"INNER JOIN ("
                f"  SELECT stock_code, MAX(fiscal_year) as my FROM fnspace_finance "
                f"  WHERE fiscal_quarter='Annual' AND fiscal_year <= %s AND stock_code IN ({ph}) "
                f"  GROUP BY stock_code"
                f") t ON ff.stock_code = t.stock_code AND ff.fiscal_year = t.my "
                f"AND ff.fiscal_quarter = 'Annual'",
                (_max_fy, *tuple(a_codes)),
            ).fetchall()
            for vr in val_rows:
                val_map[vr[0].lstrip("A")] = {
                    "PER": round(vr[1], 1) if vr[1] else None,
                    "PBR": round(vr[2], 2) if vr[2] else None,
                    "EV/EBITDA": round(vr[3], 1) if vr[3] else None,
                }
        except Exception:
            pass
        finally:
            conn.close()

    # 섹터 조회
    sector_map: dict[str, str] = {}
    conn2 = _get_conn_raw()
    try:
        a_codes = [f"A{c}" if not c.startswith("A") else c for c in codes]
        ph2 = ",".join(["%s"] * len(a_codes))
        sec_rows = conn2.execute(
            f"SELECT DISTINCT ON (stock_code) stock_code, COALESCE(sec_cd_nm, '기타') "
            f"FROM fnspace_master WHERE stock_code IN ({ph2}) ORDER BY stock_code",
            tuple(a_codes),
        ).fetchall()
        sector_map = {r[0].lstrip("A"): r[1] for r in sec_rows}
    finally:
        conn2.close()

    return [
        {
            "종목코드": r[0],
            "종목명": names.get(r[0], r[0]),
            "섹터": sector_map.get(r[0], ""),
            "점수": r[1],
            "비중(%)": round(r[2] * 100, 2),
            "시가총액": r[3],
            "PER": val_map.get(r[0], {}).get("PER"),
            "PBR": val_map.get(r[0], {}).get("PBR"),
            "EV/EBITDA": val_map.get(r[0], {}).get("EV/EBITDA"),
        }
        for r in rows
    ]


def get_holdings(strategy: str, calc_date: str, top_n: int = 30,
                 universe: str = None, rebal_type: str = None) -> pd.DataFrame:
    """캐시에서 보유종목 데이터를 읽음. 저장 전략은 PG에서 로드."""
    # 기본 전략: holdings 캐시에서
    cache = _load_holdings_cache(universe, rebal_type)
    rows = cache.get(strategy, {}).get(calc_date, [])
    if rows:
        if isinstance(rows[0], (list, tuple)) and len(rows[0]) >= 4:
            rows = _convert_raw_holdings(rows, calc_date)
        return pd.DataFrame(rows)
    # 저장 전략: holdings_json (DB별도 컬럼) 또는 results 내 holdings
    data = load_strategy(strategy)
    if data:
        holdings_src = data.get("results", {}).get("holdings", {})
        if not holdings_src:
            holdings_src = data.get("results", {}).get("holdings_by_date", {})
        rows = holdings_src.get(calc_date, [])
        if rows:
            if isinstance(rows[0], (list, tuple)) and len(rows[0]) >= 4:
                rows = _convert_raw_holdings(rows, calc_date)
            return pd.DataFrame(rows)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_first_entry_dates(strategy: str, universe: str = None, rebal_type: str = None) -> dict:
    """전략 전체 기간에서 각 종목코드의 최초 편입일 반환. {종목코드: 날짜문자열}"""
    cache = _load_holdings_cache(universe, rebal_type)
    strat_cache = cache.get(strategy, {})
    first_dates = {}
    for date in sorted(strat_cache.keys()):
        for row in strat_cache[date]:
            code = row[0] if isinstance(row, (list, tuple)) else row.get("종목코드")
            if code and code not in first_dates:
                first_dates[code] = date
    return first_dates


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
        # Custom backtest results come as {"KOSPI": {...}, "CUSTOM": {...}}
        # Extract the CUSTOM sub-result for storage
        if "CUSTOM" in clean and "rebalance_dates" not in clean:
            custom = clean["CUSTOM"]
            holdings_json = custom.pop("holdings", None)
            results_json = custom
        else:
            holdings_json = clean.pop("holdings", None)
            # 레짐 조합은 holdings_by_date 키를 사용
            if not holdings_json and "holdings_by_date" in clean:
                holdings_json = clean.pop("holdings_by_date", None)
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
            # 사전 정의 전략 fallback
            if name in _BASE_STRATEGY_CODES:
                return {"name": name, "code": _BASE_STRATEGY_CODES[name], "description": "", "results": {}, "holdings": None}
            return {}
        result = {
            "name": row[0],
            "code": row[1] or _BASE_STRATEGY_CODES.get(row[0], ""),
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
            SELECT DISTINCT ON (name) name, description, created_at, results_json
            FROM backtest_cache
            WHERE {where}
            ORDER BY name, updated_at DESC
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
                          rebal_type: str = None,
                          stop_loss_enabled: bool = None, stop_loss_pct: int = None,
                          stop_loss_mode: str = None, stop_loss_basis: str = None) -> dict | None:
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
    from lib.factor_engine import prefetch_all_data, clear_prefetch_cache

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

    # stop loss 파라미터
    _sl_enabled = params.get("stop_loss_enabled", False)
    _sl_pct = params.get("stop_loss_pct", 15)
    _sl_mode = params.get("stop_loss_mode", "sell")
    _sl_basis = params.get("stop_loss_basis", "entry")
    if stop_loss_enabled is not None: _sl_enabled = stop_loss_enabled
    if stop_loss_pct is not None: _sl_pct = stop_loss_pct
    if stop_loss_mode is not None: _sl_mode = stop_loss_mode
    if stop_loss_basis is not None: _sl_basis = stop_loss_basis

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
        "stop_loss_enabled": BACKTEST_CONFIG.get("stop_loss_enabled", False),
        "stop_loss_pct": BACKTEST_CONFIG.get("stop_loss_pct", 15),
        "stop_loss_mode": BACKTEST_CONFIG.get("stop_loss_mode", "sell"),
        "stop_loss_basis": BACKTEST_CONFIG.get("stop_loss_basis", "entry"),
    }
    try:
        BACKTEST_CONFIG["top_n_stocks"] = top_n
        BACKTEST_CONFIG["transaction_cost_bp"] = tx_cost_bp
        BACKTEST_CONFIG["weight_cap_pct"] = weight_cap_pct
        BACKTEST_CONFIG["rebal_type"] = _rebal
        BACKTEST_CONFIG["stop_loss_enabled"] = _sl_enabled
        BACKTEST_CONFIG["stop_loss_pct"] = _sl_pct
        BACKTEST_CONFIG["stop_loss_mode"] = _sl_mode
        BACKTEST_CONFIG["stop_loss_basis"] = _sl_basis
        if universe:
            BACKTEST_CONFIG["universe"] = universe

        # 프리페치: 전체 데이터를 한 번에 메모리로 로드
        _pf_conn = get_db()
        prefetch_all_data(_pf_conn)
        _pf_conn.close()

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

                    # 밸류에이션 조회 (look-ahead bias 방지)
                    codes = [c for c, _, _, _ in items]
                    if codes:
                        _dt = datetime.strptime(date, "%Y-%m-%d")
                        _max_fy = _dt.year - 1 if _dt.month >= 4 else _dt.year - 2
                        fin_rows = conn2.execute(f"""
                            SELECT ff.stock_code, ff.per, ff.pbr, ff.ev_ebitda
                            FROM fnspace_finance ff
                            INNER JOIN (
                                SELECT stock_code, MAX(fiscal_year) as my
                                FROM fnspace_finance
                                WHERE fiscal_quarter='Annual'
                                  AND fiscal_year <= ?
                                  AND stock_code IN ({','.join(['?']*len(codes))})
                                GROUP BY stock_code
                            ) t ON ff.stock_code = t.stock_code AND ff.fiscal_year = t.my
                                AND ff.fiscal_quarter = 'Annual'
                        """, (_max_fy, *tuple(f"A{c}" for c in codes))).fetchall()
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
        clear_prefetch_cache()
        BACKTEST_CONFIG["top_n_stocks"] = orig["top_n_stocks"]
        BACKTEST_CONFIG["transaction_cost_bp"] = orig["transaction_cost_bp"]
        BACKTEST_CONFIG["weight_cap_pct"] = orig["weight_cap_pct"]
        BACKTEST_CONFIG["universe"] = orig["universe"]
        BACKTEST_CONFIG["rebal_type"] = orig["rebal_type"]
        BACKTEST_CONFIG["stop_loss_enabled"] = orig["stop_loss_enabled"]
        BACKTEST_CONFIG["stop_loss_pct"] = orig["stop_loss_pct"]
        BACKTEST_CONFIG["stop_loss_mode"] = orig["stop_loss_mode"]


# 사이클 기준 약세장 기간 (2018년 이후, 데이터 시작 기준)
_CYCLE_BEAR_PERIODS = [
    ("2018-01-01", "2019-01-01"),  # 미중 무역전쟁
    ("2020-01-01", "2020-03-31"),  # 코로나 쇼크
    ("2021-06-01", "2022-10-31"),  # 글로벌 금리 인상
    ("2024-07-01", "2024-10-31"),  # AI 랠리 단기 조정
]


def _get_regime_by_cycle(calc_date: str) -> str:
    """사이클 기준 레짐 판정 (하드코딩된 약세장 기간)"""
    for start, end in _CYCLE_BEAR_PERIODS:
        if start <= calc_date <= end:
            return "Bear"
    return "Bull"


def run_regime_combo_backtest(
    bull_key: str,
    bear_key: str,
    universe: str = None,
    rebal_type: str = None,
    ma_window: int = 50,
    regime_mode: str = "ma",  # "ma" or "cycle"
) -> dict | None:
    """
    레짐 조합 백테스트: 장세마다 다른 전략을 실제로 적용해 완전 재실행.

    - Bull (KOSPI 200 > 50일 MA +3%) → bull_key 전략 종목 선택
    - Bear (KOSPI 200 < 50일 MA -3%) → bear_key 전략 종목 선택
    - Sideways (±3% 이내)            → bear_key 전략 종목 선택

    turnover 및 거래비용은 step7_backtest 루프가 자동으로 정확히 계산.
    """
    from step7_backtest import (
        run_backtest, calc_all_benchmarks,
        get_db, get_rebalance_dates, _numpy_to_python,
    )
    from lib.factor_engine import prefetch_all_data, clear_prefetch_cache, score_stocks_from_strategy
    from step7_backtest import get_universe_stocks
    from lib.db import get_conn as _get_conn

    # 1. 두 전략 코드 로드
    bull_data = load_strategy(bull_key, rebal_type=rebal_type, universe=universe)
    bear_data = load_strategy(bear_key, rebal_type=rebal_type, universe=universe)
    if not bull_data.get("code") or not bear_data.get("code"):
        return {"error": "전략 코드를 찾을 수 없습니다."}

    # 2. 코드 → 모듈 변환 및 검증
    for code_str, label in [(bull_data["code"], "Bull"), (bear_data["code"], "Bear")]:
        ok, err = validate_strategy_code(code_str)
        if not ok:
            return {"error": f"{label} 전략 오류: {err}"}

    bull_module = code_to_module(bull_data["code"])
    bear_module = code_to_module(bear_data["code"])

    # 3. 파라미터 (tx_cost는 두 전략 평균, cap은 레짐별 분리)
    bull_params = getattr(bull_module, "PARAMS", {})
    bear_params = getattr(bear_module, "PARAMS", {})
    top_n = bull_params.get("top_n", 30)
    tx_cost_bp = int((bull_params.get("tx_cost_bp", 30) + bear_params.get("tx_cost_bp", 30)) / 2)
    bull_cap_pct = bull_params.get("weight_cap_pct", 15)
    bear_cap_pct = bear_params.get("weight_cap_pct", 10)
    bull_sl = bull_params.get("stop_loss_enabled", False)
    bear_sl = bear_params.get("stop_loss_enabled", False)
    bull_sl_pct = bull_params.get("stop_loss_pct", 15)
    bear_sl_pct = bear_params.get("stop_loss_pct", 15)
    bull_sl_mode = bull_params.get("stop_loss_mode", "sell")
    bear_sl_mode = bear_params.get("stop_loss_mode", "sell")
    _rebal = rebal_type or BACKTEST_CONFIG.get("rebal_type", "monthly")
    _universe = universe or BACKTEST_CONFIG.get("universe", "KOSPI")

    # 4. 레짐 신호 사전 계산 (전체 날짜 범위)
    _conn_regime = _get_conn()
    regime_cache: dict[str, str] = {}

    def _get_regime(calc_date: str) -> str:
        if calc_date in regime_cache:
            return regime_cache[calc_date]
        if regime_mode == "cycle":
            result_regime = _get_regime_by_cycle(calc_date)
        else:
            rows = _conn_regime.execute(
                "SELECT close FROM daily_price WHERE stock_code = '069500' "
                f"AND trade_date <= ? ORDER BY trade_date DESC LIMIT {ma_window + 1}",
                (calc_date,)
            ).fetchall()
            prices = [r[0] for r in rows if r[0]]
            if len(prices) < ma_window + 1:
                result_regime = "Bull"
            else:
                current = prices[0]
                ma50 = float(np.mean(prices[1:ma_window + 1]))
                result_regime = "Bull" if current >= ma50 else "Bear"
        regime_cache[calc_date] = result_regime
        return result_regime

    # 5. 레짐 기반 stock_selector (cap도 레짐별 동적 적용)
    def regime_stock_selector(conn, calc_date, _top_n):
        regime = _get_regime(calc_date)
        module = bull_module if regime == "Bull" else bear_module
        # 레짐별 cap 및 stop loss 동적 적용
        BACKTEST_CONFIG["weight_cap_pct"] = bull_cap_pct if regime == "Bull" else bear_cap_pct
        if regime == "Bull":
            BACKTEST_CONFIG["stop_loss_enabled"] = bull_sl
            BACKTEST_CONFIG["stop_loss_pct"] = bull_sl_pct
            BACKTEST_CONFIG["stop_loss_mode"] = bull_sl_mode
        else:
            BACKTEST_CONFIG["stop_loss_enabled"] = bear_sl
            BACKTEST_CONFIG["stop_loss_pct"] = bear_sl_pct
            BACKTEST_CONFIG["stop_loss_mode"] = bear_sl_mode
        universe_set = get_universe_stocks(conn, calc_date)
        candidates = score_stocks_from_strategy(conn, calc_date, module)
        return [(c, s) for c, s in candidates if c in universe_set][:_top_n]

    orig = {
        "top_n_stocks": BACKTEST_CONFIG["top_n_stocks"],
        "transaction_cost_bp": BACKTEST_CONFIG["transaction_cost_bp"],
        "weight_cap_pct": BACKTEST_CONFIG.get("weight_cap_pct", 15),
        "universe": BACKTEST_CONFIG.get("universe", "KOSPI"),
        "rebal_type": BACKTEST_CONFIG.get("rebal_type", "monthly"),
        "stop_loss_enabled": BACKTEST_CONFIG.get("stop_loss_enabled", False),
        "stop_loss_pct": BACKTEST_CONFIG.get("stop_loss_pct", 15),
        "stop_loss_mode": BACKTEST_CONFIG.get("stop_loss_mode", "sell"),
    }
    try:
        BACKTEST_CONFIG["top_n_stocks"] = top_n
        BACKTEST_CONFIG["transaction_cost_bp"] = tx_cost_bp
        BACKTEST_CONFIG["weight_cap_pct"] = bull_cap_pct  # 초기값은 bull (첫 기간용)
        BACKTEST_CONFIG["stop_loss_enabled"] = bull_sl  # 초기값은 bull
        BACKTEST_CONFIG["stop_loss_pct"] = bull_sl_pct
        BACKTEST_CONFIG["stop_loss_mode"] = bull_sl_mode
        BACKTEST_CONFIG["rebal_type"] = _rebal
        BACKTEST_CONFIG["universe"] = _universe

        _pf_conn = get_db()
        prefetch_all_data(_pf_conn)
        _pf_conn.close()

        result = run_backtest(
            "regime_combo",
            stock_selector=regime_stock_selector,
            rebal_type=_rebal,
        )

        results = {}
        if result:
            result["strategy"] = "레짐 조합"
            result["bull_key"] = bull_key
            result["bear_key"] = bear_key
            results["REGIME_COMBO"] = result

        # 벤치마크 + 두 원전략 결과 포함
        conn_bm = get_db()
        rb_dates = get_rebalance_dates(conn_bm, _rebal)
        if len(rb_dates) >= 2:
            bm = calc_all_benchmarks(conn_bm, rb_dates)
            results.update(bm)
        conn_bm.close()

        # 두 원전략 결과도 함께 반환 (비교용)
        all_cached = load_all_results(universe=_universe, rebal_type=_rebal)
        for key in [bull_key, bear_key]:
            if key in all_cached:
                results[key] = all_cached[key]

        clear_factor_cache()
        return _numpy_to_python(results)

    finally:
        _conn_regime.close()
        clear_prefetch_cache()
        BACKTEST_CONFIG["top_n_stocks"] = orig["top_n_stocks"]
        BACKTEST_CONFIG["transaction_cost_bp"] = orig["transaction_cost_bp"]
        BACKTEST_CONFIG["weight_cap_pct"] = orig["weight_cap_pct"]
        BACKTEST_CONFIG["universe"] = orig["universe"]
        BACKTEST_CONFIG["rebal_type"] = orig["rebal_type"]
        BACKTEST_CONFIG["stop_loss_enabled"] = orig["stop_loss_enabled"]
        BACKTEST_CONFIG["stop_loss_pct"] = orig["stop_loss_pct"]
        BACKTEST_CONFIG["stop_loss_mode"] = orig["stop_loss_mode"]


def compute_regime_analysis(
    start: str = None,
    end: str = None,
    universe: str = None,
    rebal_type: str = None,
    ma_window: int = 50,
) -> dict:
    """KOSPI 200 50일 이동평균 기준으로 시장 국면(Bull/Bear)을 분류하고,
    각 전략의 국면별 성과 통계를 반환한다.

    국면 기준 (리밸런싱 시점):
        Bull : KOSPI 200 >= 50일 MA
        Bear : KOSPI 200 < 50일 MA

    Returns:
        {
            "regimes": {date_str: regime_str, ...},          # 국면 분류 (날짜별)
            "summary": {strategy_key: {regime: stats}, ...}, # 전략×국면 성과
            "regime_counts": {regime: int, ...},             # 전체 국면 월 수
        }
    """
    from lib.db import get_conn as _get_conn
    results = load_all_results(start, end, universe=universe, rebal_type=rebal_type)

    bm = results.get("KOSPI", {})
    bm_dates = bm.get("rebalance_dates", [])
    bm_returns = bm.get("monthly_returns", [])

    if len(bm_dates) < 2 or not bm_returns:
        return {"regimes": {}, "summary": {}, "regime_counts": {}}

    # 날짜 기반 periods_per_year (월간/격주 자동 대응)
    from datetime import datetime as _dt
    _total_days = (_dt.strptime(bm_dates[-1], "%Y-%m-%d") - _dt.strptime(bm_dates[0], "%Y-%m-%d")).days
    _ppy_regime = len(bm_returns) / max(_total_days / 365.25, 0.5) if _total_days > 0 else 12

    # monthly_returns[i] → 기간: bm_dates[i] ~ bm_dates[i+1]
    n = len(bm_returns)
    ret_dates = bm_dates[1 : n + 1]   # len == n (각 기간의 종료일)
    start_dates = bm_dates[:n]         # len == n (각 기간의 시작일, 신호 판단 기준)

    bm_arr = np.array(bm_returns, dtype=float)

    # KOSPI 200 50일 MA 신호: 각 리밸런싱 시작일 기준
    _conn = _get_conn()
    regimes_by_date = {}
    for i, (start_d, end_d) in enumerate(zip(start_dates, ret_dates)):
        rows = _conn.execute(
            "SELECT close FROM daily_price WHERE stock_code = '069500' "
            "AND trade_date <= ? ORDER BY trade_date DESC LIMIT 51",
            (start_d,)
        ).fetchall()
        prices = [r[0] for r in rows if r[0]]
        if len(prices) < 51:
            regime = "Bull"
        else:
            current = prices[0]
            ma50 = float(np.mean(prices[1:51]))
            regime = "Bull" if current >= ma50 else "Bear"
        regimes_by_date[str(end_d)] = regime
    _conn.close()

    regime_counts = {"Bull": 0, "Bear": 0}
    for r in regimes_by_date.values():
        regime_counts[r] = regime_counts.get(r, 0) + 1

    # 전략별 국면 성과 계산
    REGIMES = ["Bull", "Bear"]
    summary = {}

    for key, val in results.items():
        if not isinstance(val, dict):
            continue
        strat_dates = val.get("rebalance_dates", [])
        strat_returns = val.get("monthly_returns", [])
        if not strat_returns or len(strat_dates) < 2:
            continue

        ns = len(strat_returns)
        strat_ret_dates = strat_dates[1 : ns + 1]
        strat_arr = np.array(strat_returns, dtype=float)

        # 날짜 → 인덱스 맵 (bm 기준)
        bm_date_idx = {str(d): i for i, d in enumerate(ret_dates)}

        per_regime: dict[str, dict] = {r: {"rets": [], "bm_rets": []} for r in ["Bull", "Bear"]}

        for j, sd in enumerate(strat_ret_dates):
            regime = regimes_by_date.get(str(sd))
            if regime is None:
                continue
            per_regime[regime]["rets"].append(float(strat_arr[j]))
            bi = bm_date_idx.get(str(sd))
            if bi is not None:
                per_regime[regime]["bm_rets"].append(float(bm_arr[bi]))

        strat_summary = {}
        for regime, data in per_regime.items():
            rets = np.array(data["rets"], dtype=float)
            bm_rets = np.array(data["bm_rets"], dtype=float)
            cnt = len(rets)
            if cnt == 0:
                strat_summary[regime] = {
                    "count": 0,
                    "avg_monthly_return": None,
                    "total_return": None,
                    "sharpe": None,
                    "win_rate": None,
                    "avg_excess": None,
                }
                continue

            avg_ret = float(rets.mean())
            total_ret = float(np.prod(1 + rets) - 1)
            win_rate = float((rets > 0).mean())

            if cnt >= 3 and rets.std() > 0:
                sharpe = float(rets.mean() / rets.std() * np.sqrt(_ppy_regime))
            else:
                sharpe = None

            if len(bm_rets) == cnt:
                avg_excess = float((rets - bm_rets).mean())
            else:
                avg_excess = None

            strat_summary[regime] = {
                "count": cnt,
                "avg_monthly_return": avg_ret,
                "total_return": total_ret,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "avg_excess": avg_excess,
            }

        summary[key] = strat_summary

    return {
        "regimes": regimes_by_date,
        "summary": summary,
        "regime_counts": regime_counts,
    }
