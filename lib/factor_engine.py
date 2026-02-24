"""
Factor Engine: step3에서 추출한 재사용 가능한 팩터 계산 엔진.

전략 설정(strategy config)을 받아 팩터 데이터 로딩 → 회귀분석 → 십분위 채점 →
가중합 계산 → 퀄리티 필터 → 최종 종목 선정까지의 전체 파이프라인을 실행한다.
"""
import importlib.util
import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy import stats

# ─── Path setup ───
ALPHA_LAB_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ALPHA_LAB_DIR))
from config.settings import DB_PATH

LARGE_CAP_CUTOFF = 200
FINANCE_TYPES = ["금융업", "은행업", "보험업", "증권업", "여신전문금융업", "기타금융업"]

# 날짜별 factor data 캐시 (전략과 무관한 범용 데이터)
_factor_data_cache: dict[str, pd.DataFrame] = {}


# ═══════════════════════════════════════════════════════
# 십분위/사분위 점수 함수 (step3에서 그대로)
# ═══════════════════════════════════════════════════════

def _pcts(series, only_positive=False):
    valid = series[(series > 0) & series.notna()] if only_positive else series[series.notna()]
    if len(valid) < 20:
        return None
    return {p: valid.quantile(p / 100) for p in range(10, 100, 10)}


def decile_rule1(series):
    """낮을수록 좋음 (밸류멀티플). <=0/NaN->0, P10이하->10 ... P90이상->1"""
    P = _pcts(series, only_positive=True)
    if P is None:
        return pd.Series(0, index=series.index)

    def s(v):
        if pd.isna(v) or v <= 0:
            return 0
        for i, p in enumerate([90, 80, 70, 60, 50, 40, 30, 20, 10]):
            if v >= P[p]:
                return i + 1
        return 10
    return series.apply(s)


def decile_rule2(series):
    """높을수록 좋음 (회귀/성장/CURRENT). P90이상->10 ... P10이하->1, NaN->0"""
    P = _pcts(series)
    if P is None:
        return pd.Series(0, index=series.index)

    def s(v):
        if pd.isna(v):
            return 0
        for i, p in enumerate([90, 80, 70, 60, 50, 40, 30, 20, 10]):
            if v >= P[p]:
                return 10 - i
        return 1
    return series.apply(s)


def decile_rule3(series):
    """낮을수록 좋음 (PRICE_M/NDEBT_EBITDA). NaN->0, P90이상->1 ... P10이하->10"""
    P = _pcts(series)
    if P is None:
        return pd.Series(0, index=series.index)

    def s(v):
        if pd.isna(v):
            return 0
        for i, p in enumerate([90, 80, 70, 60, 50, 40, 30, 20, 10]):
            if v >= P[p]:
                return i + 1
        return 10
    return series.apply(s)


def quartile_rule1(series):
    """사분위: 낮을수록 좋음 (0~4). NaN/<=0→0, Q3+→1, Q2-Q3→2, Q1-Q2→3, <Q1→4"""
    flt = series[(series > 0) & series.notna()]
    if len(flt) < 20:
        return pd.Series(0, index=series.index)
    Q1, Q2, Q3 = flt.quantile(0.25), flt.quantile(0.5), flt.quantile(0.75)

    def s(v):
        if pd.isna(v) or v <= 0:
            return 0
        if v >= Q3:
            return 1
        if v >= Q2:
            return 2
        if v >= Q1:
            return 3
        return 4
    return series.apply(s)


def quartile_rule2(series):
    """사분위: 높을수록 좋음 (0~4). NaN→0, <Q1→1, Q1-Q2→2, Q2-Q3→3, Q3+→4"""
    flt = series[series.notna()]
    if len(flt) < 20:
        return pd.Series(0, index=series.index)
    Q1, Q2, Q3 = flt.quantile(0.25), flt.quantile(0.5), flt.quantile(0.75)

    def s(v):
        if pd.isna(v):
            return 0
        if v >= Q3:
            return 4
        if v >= Q2:
            return 3
        if v >= Q1:
            return 2
        return 1
    return series.apply(s)


SCORING_FUNCS = {
    "rule1": decile_rule1,
    "rule2": decile_rule2,
    "rule3": decile_rule3,
}

# 사분위 매핑: rule → quartile 함수
_QUARTILE_MAP = {
    "rule1": quartile_rule1,   # 낮을수록 좋음
    "rule2": quartile_rule2,   # 높을수록 좋음
    "rule3": quartile_rule1,   # 낮을수록 좋음 (rule1과 동일 방향)
}


# ═══════════════════════════════════════════════════════
# 1. 범용 팩터 데이터 로딩 (전략과 무관)
# ═══════════════════════════════════════════════════════

def load_factor_data(conn, calc_date: str) -> pd.DataFrame | None:
    """
    DB에서 재무/주가/포워드 데이터를 로딩하고 모든 파생 지표를 계산한다.
    전략과 무관한 범용 데이터이므로 날짜별로 캐시 가능.
    """
    if calc_date in _factor_data_cache:
        return _factor_data_cache[calc_date].copy()

    dt = datetime.strptime(calc_date, "%Y-%m-%d")
    max_usable_year = dt.year - 1 if dt.month >= 4 else dt.year - 2

    # ─── 1. Trailing 재무 ───
    fin_df = pd.read_sql_query("""
        SELECT ff.stock_code, ff.fiscal_year,
               ff.pbr, ff.roe, ff.roic,
               ff.ev, ff.ic, ff.ev_ebit, ff.ebit, ff.ebitda,
               ff.net_debt, ff.interest_debt, ff.total_equity,
               ff.eps, ff.bps, ff.per, ff.psr, ff.ev_ebitda,
               ff.revenue, ff.operating_income, ff.net_income,
               ff.oi_margin, ff.div_yield, ff.pcf
        FROM fnspace_finance ff
        INNER JOIN (
            SELECT stock_code, MAX(fiscal_year) as max_year
            FROM fnspace_finance
            WHERE fiscal_quarter = 'Annual' AND fiscal_year <= ? AND roe IS NOT NULL
            GROUP BY stock_code
        ) latest ON ff.stock_code = latest.stock_code
            AND ff.fiscal_year = latest.max_year AND ff.fiscal_quarter = 'Annual'
    """, conn, params=(max_usable_year,))

    if fin_df.empty:
        return None

    prev_rev_df = pd.read_sql_query("""
        SELECT stock_code, revenue as prev_revenue
        FROM fnspace_finance WHERE fiscal_quarter='Annual' AND fiscal_year=?
    """, conn, params=(max_usable_year - 1,))

    # ─── 2. Forward ───
    fwd_date = conn.execute(
        "SELECT MAX(trade_date) FROM fnspace_forward WHERE trade_date < ?",
        (calc_date,),
    ).fetchone()[0]
    if not fwd_date:
        fwd_date = conn.execute(
            "SELECT MIN(trade_date) FROM fnspace_forward"
        ).fetchone()[0]

    fwd_df = pd.DataFrame()
    if fwd_date:
        fwd_df = pd.read_sql_query("""
            SELECT stock_code, fwd_eps, fwd_per, fwd_ebit, fwd_ebitda,
                   fwd_ev_ebitda, fwd_revenue, fwd_oi, fwd_ni, fwd_roe, fwd_bps
            FROM fnspace_forward WHERE trade_date = ?
        """, conn, params=(fwd_date,))

    # 3개월 전 Forward EPS
    three_m_ago = (dt - timedelta(days=90)).strftime("%Y-%m-%d")
    fwd_date_3m = conn.execute(
        "SELECT MAX(trade_date) FROM fnspace_forward WHERE trade_date <= ?",
        (three_m_ago,),
    ).fetchone()[0]
    fwd_3m_df = pd.DataFrame()
    if fwd_date_3m:
        fwd_3m_df = pd.read_sql_query(
            "SELECT stock_code, fwd_eps as fwd_eps_3m FROM fnspace_forward WHERE trade_date=?",
            conn, params=(fwd_date_3m,),
        )

    # ─── 3. 주가 ───
    price_date = conn.execute(
        "SELECT MAX(trade_date) FROM daily_price WHERE trade_date < ?",
        (calc_date,),
    ).fetchone()[0]
    price_df = pd.read_sql_query("""
        SELECT 'A' || dp.stock_code as stock_code, dp.close, dp.market_cap, dp.trade_amount
        FROM daily_price dp WHERE dp.trade_date = ?
    """, conn, params=(price_date,))

    # 3개월 전 주가
    price_date_3m = conn.execute(
        "SELECT MAX(trade_date) FROM daily_price WHERE trade_date <= ?",
        (three_m_ago,),
    ).fetchone()[0]
    price_3m_df = pd.DataFrame()
    if price_date_3m:
        price_3m_df = pd.read_sql_query(
            "SELECT 'A'||stock_code as stock_code, close as close_3m FROM daily_price WHERE trade_date=?",
            conn, params=(price_date_3m,),
        )

    master_df = pd.read_sql_query(
        "SELECT stock_code, stock_name, market, sec_cd_nm, finacc_typ FROM fnspace_master", conn,
    )

    # ─── 병합 ───
    merged = fin_df.merge(price_df, on="stock_code", how="inner")
    merged = merged.merge(master_df, on="stock_code", how="inner")
    for df_extra in [fwd_df, prev_rev_df, fwd_3m_df, price_3m_df]:
        if not df_extra.empty:
            merged = merged.merge(df_extra, on="stock_code", how="left")
    merged = merged[(merged["market_cap"] > 0) & (merged["close"] > 0)].copy()
    if len(merged) == 0:
        return None

    # 단위 통일 (천원 -> 원)
    for c in ["ev", "ic", "ebit", "ebitda", "net_debt", "interest_debt",
              "total_equity", "revenue", "operating_income", "net_income"]:
        if c in merged.columns:
            merged[c] = merged[c] * 1000
    for c in ["fwd_ebit", "fwd_ebitda", "fwd_revenue", "fwd_oi", "fwd_ni"]:
        if c in merged.columns:
            merged[c] = merged[c] * 1000
    if "prev_revenue" in merged.columns:
        merged["prev_revenue"] *= 1000

    # 금융업 제외
    merged = merged[~merged["finacc_typ"].isin(FINANCE_TYPES)].copy()

    # ─── 밸류에이션 재계산 (현재 주가 기준) ───
    m = merged["bps"].notna() & (merged["bps"] > 0)
    merged["pbr"] = np.where(m, merged["close"] / merged["bps"], np.nan)

    m = merged["eps"].notna() & (merged["eps"] > 0)
    merged["t_per"] = np.where(m, merged["close"] / merged["eps"], np.nan)

    m = merged["net_debt"].notna()
    merged["ev"] = np.where(m, merged["market_cap"] + merged["net_debt"], np.nan)

    # T_EV/EBITDA
    if "ebitda" in merged.columns and merged["ebitda"].notna().sum() > 10:
        m = (merged["ev"] > 0) & merged["ebitda"].notna() & (merged["ebitda"] > 0)
        merged["t_ev_ebitda"] = np.where(m, merged["ev"] / merged["ebitda"], np.nan)
    else:
        m = (merged["ev"] > 0) & merged["ebit"].notna() & (merged["ebit"] > 0)
        merged["t_ev_ebitda"] = np.where(m, merged["ev"] / merged["ebit"], np.nan)

    m = (merged["ev"] > 0) & merged["ic"].notna() & (merged["ic"] > 0)
    merged["ev_ic"] = np.where(m, merged["ev"] / merged["ic"], np.nan)

    if "fwd_eps" in merged.columns:
        m = merged["fwd_eps"].notna() & (merged["fwd_eps"] > 0)
        merged["f_per"] = np.where(m, merged["close"] / merged["fwd_eps"], np.nan)
    else:
        merged["f_per"] = np.nan

    # F_EV/EBITDA
    if "fwd_ebitda" in merged.columns and merged["fwd_ebitda"].notna().sum() > 10:
        m = (merged["ev"] > 0) & merged["fwd_ebitda"].notna() & (merged["fwd_ebitda"] > 0)
        merged["f_ev_ebitda"] = np.where(m, merged["ev"] / merged["fwd_ebitda"], np.nan)
    elif "fwd_ebit" in merged.columns:
        m = (merged["ev"] > 0) & merged["fwd_ebit"].notna() & (merged["fwd_ebit"] > 0)
        merged["f_ev_ebitda"] = np.where(m, merged["ev"] / merged["fwd_ebit"], np.nan)
    else:
        merged["f_ev_ebitda"] = np.nan

    # F_EV/EBIT
    if "fwd_ebit" in merged.columns:
        m = (merged["ev"] > 0) & merged["fwd_ebit"].notna() & (merged["fwd_ebit"] > 0)
        merged["f_ev_ebit"] = np.where(m, merged["ev"] / merged["fwd_ebit"], np.nan)
    else:
        merged["f_ev_ebit"] = np.nan

    # F_PBR
    if "fwd_bps" in merged.columns:
        m = merged["fwd_bps"].notna() & (merged["fwd_bps"] > 0)
        merged["f_pbr"] = np.where(m, merged["close"] / merged["fwd_bps"], np.nan)
    else:
        merged["f_pbr"] = np.nan

    # T_PCF
    if "pcf" in merged.columns:
        m = merged["pcf"].notna() & (merged["pcf"] > 0)
        merged["t_pcf"] = np.where(m, merged["pcf"], np.nan)
    else:
        merged["t_pcf"] = np.nan

    # ─── 파생 변수 ───
    if "fwd_eps" in merged.columns:
        m = merged["eps"].notna() & (merged["eps"].abs() > 0) & merged["fwd_eps"].notna()
        merged["f_epsg"] = np.where(
            m, (merged["fwd_eps"] - merged["eps"]) / merged["eps"].abs() * 100, np.nan,
        )
    else:
        merged["f_epsg"] = np.nan

    if "fwd_ebit" in merged.columns:
        m = merged["ebit"].notna() & (merged["ebit"].abs() > 0) & merged["fwd_ebit"].notna()
        merged["f_ebitg"] = np.where(
            m, (merged["fwd_ebit"] - merged["ebit"]) / merged["ebit"].abs() * 100, np.nan,
        )
    else:
        merged["f_ebitg"] = np.nan

    if "prev_revenue" in merged.columns:
        m = merged["revenue"].notna() & merged["prev_revenue"].notna() & (merged["prev_revenue"].abs() > 0)
        merged["t_spsg"] = np.where(
            m, (merged["revenue"] - merged["prev_revenue"]) / merged["prev_revenue"].abs() * 100, np.nan,
        )
    else:
        merged["t_spsg"] = np.nan

    if "fwd_revenue" in merged.columns:
        m = merged["revenue"].notna() & (merged["revenue"].abs() > 0) & merged["fwd_revenue"].notna()
        merged["f_spsg"] = np.where(
            m, (merged["fwd_revenue"] - merged["revenue"]) / merged["revenue"].abs() * 100, np.nan,
        )
    else:
        merged["f_spsg"] = np.nan

    # F_EPS_M
    if "fwd_eps_3m" in merged.columns and "fwd_eps" in merged.columns:
        m = merged["fwd_eps_3m"].notna() & (merged["fwd_eps_3m"].abs() > 0) & merged["fwd_eps"].notna()
        merged["f_eps_m"] = np.where(
            m, (merged["fwd_eps"] - merged["fwd_eps_3m"]) / merged["fwd_eps_3m"].abs() * 100, np.nan,
        )
    else:
        merged["f_eps_m"] = np.nan

    # PRICE_M
    if "close_3m" in merged.columns:
        m = merged["close_3m"].notna() & (merged["close_3m"] > 0)
        merged["price_m"] = np.where(
            m, (merged["close"] - merged["close_3m"]) / merged["close_3m"] * 100, np.nan,
        )
    else:
        merged["price_m"] = np.nan

    # NDEBT_EBITDA
    ebitda_col = "ebitda" if ("ebitda" in merged.columns and merged["ebitda"].notna().sum() > 10) else "ebit"
    m = merged[ebitda_col].notna() & (merged[ebitda_col] > 0) & merged["net_debt"].notna()
    merged["ndebt_ebitda"] = np.where(m, merged["net_debt"] / merged[ebitda_col], np.nan)

    merged["current_ratio"] = np.nan

    # ─── 대형/중소형 분리 ───
    merged = merged.sort_values("market_cap", ascending=False).reset_index(drop=True)
    cutoff = min(LARGE_CAP_CUTOFF, len(merged) // 3)
    merged["size_group"] = "mid_small"
    merged.loc[:cutoff - 1, "size_group"] = "large"

    # 캐시 저장
    _factor_data_cache[calc_date] = merged.copy()
    return merged


def clear_factor_cache():
    """팩터 데이터 캐시 초기화."""
    _factor_data_cache.clear()


# ═══════════════════════════════════════════════════════
# 2. 회귀분석 (전략 설정에 따라)
# ═══════════════════════════════════════════════════════

def _run_single_regression(df, x_col, y_col, model_name, formula_type, outlier_filter):
    """단일 회귀분석 실행. step3의 run_regression과 동일 로직."""
    col_attr = f"{model_name}_attractiveness"
    df[col_attr] = np.nan

    x_min = outlier_filter.get("x_min", 0)
    x_max = outlier_filter.get("x_max", 9999)
    y_min = outlier_filter.get("y_min", 0)
    y_max = outlier_filter.get("y_max", 9999)

    if x_col not in df.columns or y_col not in df.columns:
        df[col_attr] = 0.0
        return df, {"model": model_name, "n": 0, "r2": 0, "status": "missing_column"}

    vm = (df[x_col].notna() & df[y_col].notna() &
          (df[x_col] > x_min) & (df[x_col] < x_max) &
          (df[y_col] > y_min) & (df[y_col] < y_max))
    valid = df[vm].copy()

    if len(valid) < 20:
        df[col_attr] = 0.0
        return df, {"model": model_name, "n": len(valid), "r2": 0, "status": "insufficient"}

    slope, intercept, r_value, _, _ = stats.linregress(valid[x_col].values, valid[y_col].values)
    r2 = r_value ** 2
    has_x = df[x_col].notna() & (df[x_col] > x_min)
    fitted = slope * df[x_col] + intercept

    if formula_type == "ratio":
        # fitted / actual - 1 (pbr_roe, fper_epsg 형태)
        mask = has_x & (df[y_col] > 0)
        df.loc[mask, col_attr] = (fitted[mask] / df.loc[mask, y_col]) - 1
        no_x = df[x_col].isna() & (df[y_col] > 0)
        df.loc[no_x, col_attr] = (intercept / df.loc[no_x, y_col]) - 1

    elif formula_type == "ev_equity":
        # (fitted * IC - net_debt) / market_cap - 1 (evic_roic 형태)
        if "ic" in df.columns:
            fitted_ev = fitted * df["ic"]
            mask = has_x & (df["market_cap"] > 0)
            df.loc[mask, col_attr] = (
                (fitted_ev[mask] - df.loc[mask, "net_debt"]) / df.loc[mask, "market_cap"] - 1
            )

    elif formula_type == "ev_equity_ebit":
        # (fitted * |fwd_ebit| - net_debt) / market_cap - 1 (fevebit_ebitg 형태)
        if "fwd_ebit" in df.columns:
            fitted_ev = fitted * df["fwd_ebit"].abs()
            mask = has_x & (df["market_cap"] > 0)
            df.loc[mask, col_attr] = (
                (fitted_ev[mask] - df.loc[mask, "net_debt"]) / df.loc[mask, "market_cap"] - 1
            )

    elif formula_type == "simple":
        # 잔차 기반: (fitted - actual) / |actual| (커스텀용)
        mask = has_x & df[y_col].notna() & (df[y_col].abs() > 0)
        df.loc[mask, col_attr] = (fitted[mask] - df.loc[mask, y_col]) / df.loc[mask, y_col].abs()

    df[col_attr] = df[col_attr].clip(-1.0, 5.0).fillna(0.0)
    return df, {
        "model": model_name, "n": len(valid), "r2": round(r2, 4),
        "slope": round(slope, 4), "intercept": round(intercept, 4), "status": "ok",
    }


def run_regressions(df, regression_models, outlier_filters=None):
    """
    전략 설정에 따라 회귀분석을 실행한다.

    regression_models: [(name, x_col, y_col, formula_type), ...]
    outlier_filters: {model_name: {"x_min":, "x_max":, "y_min":, "y_max":}}
    """
    if outlier_filters is None:
        outlier_filters = {}

    all_info = []
    default_filter = {"x_min": 0, "x_max": 9999, "y_min": 0, "y_max": 9999}

    for model_cfg in regression_models:
        name, x_col, y_col, formula_type = model_cfg
        filt = outlier_filters.get(name, default_filter)

        for sg in ["large", "mid_small"]:
            mask = df["size_group"] == sg
            gdf = df[mask].copy()
            idx = df[mask].index

            gdf, info = _run_single_regression(gdf, x_col, y_col, name, formula_type, filt)
            info["size"] = sg
            all_info.append(info)
            df.loc[idx, f"{name}_attractiveness"] = gdf[f"{name}_attractiveness"].values

    return df, all_info


# ═══════════════════════════════════════════════════════
# 3. 십분위 채점 (전략 설정에 따라)
# ═══════════════════════════════════════════════════════

def apply_scoring(df, scoring_rules, scoring_mode=None):
    """
    scoring_rules: {column_name: "rule1"|"rule2"|"rule3", ...}
    scoring_mode: {"large": "quartile"|"decile", "small": "quartile"|"decile"}
      - 대형주: 사분위(0~4) or 십분위(0~10)
      - 중소형주: 사분위(0~4) or 십분위(0~10)
    결과 컬럼: {column_name}_score
    """
    if scoring_mode is None:
        scoring_mode = {"large": "quartile", "small": "decile"}

    for col, rule in scoring_rules.items():
        score_col = f"{col}_score"

        for sg in ["large", "mid_small"]:
            idx = df["size_group"] == sg
            mode_key = "large" if sg == "large" else "small"
            mode = scoring_mode.get(mode_key, "decile")

            if mode == "quartile":
                func = _QUARTILE_MAP.get(rule)
            else:
                func = SCORING_FUNCS.get(rule)

            if func is None:
                df.loc[idx, score_col] = 0
                continue

            if col in df.columns:
                df.loc[idx, score_col] = func(df.loc[idx, col]).values
            else:
                df.loc[idx, score_col] = 0

    return df


# ═══════════════════════════════════════════════════════
# 4. 가중합 계산
# ═══════════════════════════════════════════════════════

def calc_weighted_scores(df, weights_large, weights_small, score_map, scoring_mode=None):
    """
    weights_large/small: {FACTOR_KEY: weight, ...}
    score_map: {FACTOR_KEY: score_column_name, ...}
    scoring_mode: {"large": "quartile"|"decile", "small": ...}
    """
    if scoring_mode is None:
        scoring_mode = {"large": "quartile", "small": "decile"}

    # 점수 컬럼 없으면 0으로 초기화
    for col in score_map.values():
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    for sg, W in [("large", weights_large), ("mid_small", weights_small)]:
        mask = df["size_group"] == sg
        total = sum(
            df.loc[mask, score_map[k]] * w
            for k, w in W.items()
            if k in score_map and w > 0
        )
        df.loc[mask, "total_raw"] = total

    # 0~100 정규화: 사분위(max=4), 십분위(max=10)
    large_max = 4.0 if scoring_mode.get("large") == "quartile" else 10.0
    small_max = 4.0 if scoring_mode.get("small") == "quartile" else 10.0

    df["value_score"] = 0.0
    large_mask = df["size_group"] == "large"
    small_mask = df["size_group"] == "mid_small"
    df.loc[large_mask, "value_score"] = (
        df.loc[large_mask, "total_raw"] / large_max * 100
    ).round(1).clip(0, 100)
    df.loc[small_mask, "value_score"] = (
        df.loc[small_mask, "total_raw"] / small_max * 100
    ).round(1).clip(0, 100)
    return df


# ═══════════════════════════════════════════════════════
# 5. 퀄리티 필터
# ═══════════════════════════════════════════════════════

def apply_quality_filter(df, filter_config):
    """
    filter_config: {
        "exclude_spac_etf_reit": True,
        "require_positive_oi": True,
        "require_positive_roe": True,
        "min_avg_volume": 500_000_000,
    }
    """
    df["quality_pass"] = 1
    df["quality_fail_reason"] = ""

    if filter_config.get("exclude_spac_etf_reit", True):
        for kw in ["스팩", "SPAC", "ETF", "ETN", "리츠", "REIT"]:
            m = df["stock_name"].str.contains(kw, case=False, na=False)
            df.loc[m, "quality_pass"] = 0
            df.loc[m, "quality_fail_reason"] += "스팩/ETF/리츠; "

    if filter_config.get("require_positive_oi", True):
        m = df["operating_income"].notna() & (df["operating_income"] <= 0)
        df.loc[m, "quality_pass"] = 0
        df.loc[m, "quality_fail_reason"] += "영업적자; "

    if filter_config.get("require_positive_roe", True):
        m = df["roe"].notna() & (df["roe"] <= 0)
        df.loc[m, "quality_pass"] = 0
        df.loc[m, "quality_fail_reason"] += "ROE<=0; "

    min_vol = filter_config.get("min_avg_volume", 500_000_000)
    if min_vol > 0:
        m = df["trade_amount"] < min_vol
        df.loc[m, "quality_pass"] = 0
        df.loc[m, "quality_fail_reason"] += "거래대금부족; "

    return df


# ═══════════════════════════════════════════════════════
# 6. 종합 파이프라인
# ═══════════════════════════════════════════════════════

def score_stocks_from_strategy(conn, calc_date, strategy) -> list[tuple[str, float]]:
    """
    전략 모듈/설정에 따라 팩터 데이터 -> 회귀분석 -> 채점 -> 가중합 -> 퀄리티 필터
    전체 파이프라인을 실행하고 (stock_code, score) 리스트를 반환.

    strategy: ModuleType 또는 dict-like (WEIGHTS_LARGE, WEIGHTS_SMALL, ...)
    """
    df = load_factor_data(conn, calc_date)
    if df is None or df.empty:
        return []

    # 전략 설정 읽기
    weights_large = getattr(strategy, "WEIGHTS_LARGE", {})
    weights_small = getattr(strategy, "WEIGHTS_SMALL", {})
    regression_models = getattr(strategy, "REGRESSION_MODELS", [])
    outlier_filters = getattr(strategy, "OUTLIER_FILTERS", {})
    score_map = getattr(strategy, "SCORE_MAP", {})
    scoring_rules = getattr(strategy, "SCORING_RULES", {})
    scoring_mode = getattr(strategy, "SCORING_MODE", {"large": "quartile", "small": "decile"})
    quality_filter = getattr(strategy, "QUALITY_FILTER", {
        "exclude_spac_etf_reit": True,
        "require_positive_oi": True,
        "require_positive_roe": True,
        "min_avg_volume": 500_000_000,
    })
    top_n = getattr(strategy, "PARAMS", {}).get("top_n", 30)

    # 파이프라인 실행
    df, _reg_info = run_regressions(df, regression_models, outlier_filters)
    df = apply_scoring(df, scoring_rules, scoring_mode)
    df = calc_weighted_scores(df, weights_large, weights_small, score_map, scoring_mode)
    df = apply_quality_filter(df, quality_filter)

    # 퀄리티 통과 종목만, 점수 순 정렬
    passed = df[df["quality_pass"] == 1].nlargest(top_n * 2, "value_score")

    # stock_code에서 'A' 접두사 제거
    result = []
    for _, row in passed.iterrows():
        code = row["stock_code"]
        if code.startswith("A"):
            code = code[1:]
        result.append((code, float(row["value_score"])))

    return result[:top_n * 2]  # 유동성 필터용으로 여유분 반환


# ═══════════════════════════════════════════════════════
# 전략 파일 로딩/검증
# ═══════════════════════════════════════════════════════

# 보안: 허용되지 않는 패턴
_FORBIDDEN_PATTERNS = [
    r'\bimport\b', r'\bfrom\b', r'\bdef\b', r'\bclass\b',
    r'\bexec\b', r'\beval\b', r'__\w+__', r'\bopen\b',
    r'\bos\b', r'\bsys\b', r'\bsubprocess\b',
]


def validate_strategy_code(code: str) -> tuple[bool, str]:
    """
    전략 코드를 검증한다.
    Returns: (is_valid, error_message)
    """
    # 1. 보안 스캔 (docstring/주석 제외한 실제 코드 라인만 검사)
    lines = code.split("\n")
    in_docstring = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # docstring 토글
        if '"""' in stripped or "'''" in stripped:
            count = stripped.count('"""') + stripped.count("'''")
            if count == 1:
                in_docstring = not in_docstring
                continue
            # 같은 줄에 열고 닫는 경우
            continue

        if in_docstring:
            continue

        # 주석 라인 스킵
        if stripped.startswith("#"):
            continue

        # 실제 코드에서 금지 패턴 검사
        for pattern in _FORBIDDEN_PATTERNS:
            if re.search(pattern, stripped):
                return False, f"Line {i}: 금지된 패턴 '{pattern}' 발견: {stripped}"

    # 2. 문법 검사
    try:
        compile(code, "<strategy>", "exec")
    except SyntaxError as e:
        return False, f"문법 오류 (line {e.lineno}): {e.msg}"

    # 3. 실행하여 필수 변수 확인
    namespace = {}
    try:
        exec(compile(code, "<strategy>", "exec"), {"__builtins__": {}}, namespace)
    except Exception as e:
        return False, f"실행 오류: {e}"

    # 4. 필수 변수 존재 확인
    required = ["WEIGHTS_LARGE", "WEIGHTS_SMALL", "REGRESSION_MODELS",
                "SCORE_MAP", "SCORING_RULES", "PARAMS"]
    for var in required:
        if var not in namespace:
            return False, f"필수 변수 '{var}'가 없습니다."

    # 5. 타입 확인
    if not isinstance(namespace["WEIGHTS_LARGE"], dict):
        return False, "WEIGHTS_LARGE는 dict여야 합니다."
    if not isinstance(namespace["WEIGHTS_SMALL"], dict):
        return False, "WEIGHTS_SMALL는 dict여야 합니다."
    if not isinstance(namespace["REGRESSION_MODELS"], list):
        return False, "REGRESSION_MODELS는 list여야 합니다."

    # 6. 가중치 합 검증
    for name, weights in [("WEIGHTS_LARGE", namespace["WEIGHTS_LARGE"]),
                          ("WEIGHTS_SMALL", namespace["WEIGHTS_SMALL"])]:
        total = sum(v for v in weights.values() if v > 0)
        if total < 0.95 or total > 1.05:
            return False, f"{name} 가중치 합이 {total:.2f}입니다. 1.0에 가까워야 합니다."

    # 7. SCORE_MAP과 WEIGHTS 키 매칭
    score_map = namespace["SCORE_MAP"]
    for weights_name in ["WEIGHTS_LARGE", "WEIGHTS_SMALL"]:
        for k, v in namespace[weights_name].items():
            if v > 0 and k not in score_map:
                return False, f"{weights_name}의 '{k}' (가중치 {v})에 대응하는 SCORE_MAP 항목이 없습니다."

    # 8. PARAMS 필수 키
    params = namespace["PARAMS"]
    for key in ["top_n", "tx_cost_bp", "weight_cap_pct"]:
        if key not in params:
            return False, f"PARAMS에 '{key}'가 없습니다."

    # 9. SCORING_MODE 검증 (선택사항)
    scoring_mode = namespace.get("SCORING_MODE")
    if scoring_mode is not None:
        if not isinstance(scoring_mode, dict):
            return False, "SCORING_MODE는 dict여야 합니다."
        for k in ["large", "small"]:
            if k in scoring_mode and scoring_mode[k] not in ("quartile", "decile"):
                return False, f"SCORING_MODE['{k}']는 'quartile' 또는 'decile'이어야 합니다."

    return True, ""


def load_strategy_module(strategy_path: str) -> ModuleType:
    """전략 .py 파일을 모듈로 로딩."""
    spec = importlib.util.spec_from_file_location("strategy", strategy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_to_module(code: str) -> ModuleType:
    """전략 코드 문자열을 모듈 객체로 변환."""
    module = ModuleType("strategy")
    exec(compile(code, "<strategy>", "exec"), module.__dict__)
    return module


# ═══════════════════════════════════════════════════════
# 디폴트 전략 코드 (현재 A0와 동일)
# ═══════════════════════════════════════════════════════

DEFAULT_STRATEGY_CODE = '''"""
Strategy: A0 기본 전략
Created: 2026-02-23
Description: 원본 사분위 밸류 전략 (밸류 35% + 회귀 30% + 성장 20% + 차별화 15%)

설계 원리:
  - 대형주(시총 상위 200): 사분위(Quartile, 0~4점) 채점. 종목 수가 적어 십분위 구간이 촘촘해짐을 방지.
  - 중소형주(나머지): 십분위(Decile, 0~10점) 채점. 종목 수 충분하여 세밀한 분류 가능.
  - 대형주와 중소형주에 서로 다른 가중치 적용. 중소형주에는 주가모멘텀·부채비율·유동비율 추가.
  - 4개 회귀 모델(PBR-ROE, EV/IC-ROIC, F.PER-이익성장, F.EV/EBIT-EBIT성장)로 내재가치 괴리도 측정.
"""

# ─── 채점 방식 ───
SCORING_MODE = {
    "large": "quartile",   # 대형주: 사분위 (0~4점)
    "small": "decile",     # 중소형주: 십분위 (0~10점)
}

# ─── 팩터 가중치 (합계 1.0) ───
WEIGHTS_LARGE = {
    "T_PER": .05, "F_PER": .05, "T_EVEBITDA": .05, "F_EVEBITDA": .05,
    "T_PBR": .05, "F_PBR": .05, "T_PCF": .05,
    "ATT_PBR": .05, "ATT_EVIC": .05, "ATT_PER": .10, "ATT_EVEBIT": .10,
    "T_SPSG": .10, "F_SPSG": .10,
    "F_EPS_M": .15,
}

WEIGHTS_SMALL = {
    "T_PER": .05, "F_PER": .05, "T_EVEBITDA": .05, "F_EVEBITDA": .05,
    "T_PBR": .05, "F_PBR": .05, "T_PCF": .05,
    "ATT_PBR": .05, "ATT_EVIC": .05, "ATT_PER": .10, "ATT_EVEBIT": .10,
    "T_SPSG": .10, "F_SPSG": .10,
    "PRICE_M": .05, "NDEBT_EBITDA": .05, "CURRENT": .05,
}

# ─── 회귀 모델 (name, x_col, y_col, formula_type) ───
# formula_type: "ratio" | "ev_equity" | "ev_equity_ebit" | "simple"
REGRESSION_MODELS = [
    ("pbr_roe", "roe", "pbr", "ratio"),
    ("evic_roic", "roic", "ev_ic", "ev_equity"),
    ("fper_epsg", "f_epsg", "f_per", "ratio"),
    ("fevebit_ebitg", "f_ebitg", "f_ev_ebit", "ev_equity_ebit"),
]

# ─── 회귀 이상치 필터 ───
OUTLIER_FILTERS = {
    "pbr_roe": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 20},
    "evic_roic": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 51},
    "fper_epsg": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60},
    "fevebit_ebitg": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60},
}

# ─── 가중치 키 -> 점수 컬럼 매핑 ───
SCORE_MAP = {
    "T_PER": "t_per_score", "F_PER": "f_per_score",
    "T_EVEBITDA": "t_ev_ebitda_score", "F_EVEBITDA": "f_ev_ebitda_score",
    "T_PBR": "pbr_score", "F_PBR": "f_pbr_score", "T_PCF": "t_pcf_score",
    "ATT_PBR": "pbr_roe_attractiveness_score", "ATT_EVIC": "evic_roic_attractiveness_score",
    "ATT_PER": "fper_epsg_attractiveness_score", "ATT_EVEBIT": "fevebit_ebitg_attractiveness_score",
    "T_SPSG": "t_spsg_score", "F_SPSG": "f_spsg_score",
    "F_EPS_M": "f_eps_m_score",
    "PRICE_M": "price_m_score", "NDEBT_EBITDA": "ndebt_ebitda_score",
    "CURRENT": "current_ratio_score",
}

# ─── 스코어링 규칙 ───
# "rule1": 낮을수록 좋음 (밸류 멀티플: PER, PBR, EV/EBITDA, PCF)
# "rule2": 높을수록 좋음 (회귀 매력도, 성장률, EPS 모멘텀, 유동비율)
# "rule3": 낮을수록 좋음 (주가 모멘텀 역방향, 부채비율)
SCORING_RULES = {
    "t_per": "rule1", "f_per": "rule1",
    "t_ev_ebitda": "rule1", "f_ev_ebitda": "rule1",
    "pbr": "rule1", "f_pbr": "rule1", "t_pcf": "rule1",
    "pbr_roe_attractiveness": "rule2",
    "evic_roic_attractiveness": "rule2",
    "fper_epsg_attractiveness": "rule2",
    "fevebit_ebitg_attractiveness": "rule2",
    "t_spsg": "rule2", "f_spsg": "rule2",
    "f_eps_m": "rule2",
    "price_m": "rule3", "ndebt_ebitda": "rule3",
    "current_ratio": "rule2",
}

# ─── 운용 파라미터 ───
PARAMS = {
    "top_n": 30,
    "tx_cost_bp": 30,
    "weight_cap_pct": 15,
}

# ─── 퀄리티 필터 ───
QUALITY_FILTER = {
    "exclude_spac_etf_reit": True,
    "require_positive_oi": True,
    "require_positive_roe": True,
    "min_avg_volume": 500_000_000,
}
'''
