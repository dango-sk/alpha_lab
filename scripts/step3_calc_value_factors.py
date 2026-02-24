"""
Step 3: 종합 value_score 계산 (원본 구조 완전 복원)
실행: python scripts/step3_calc_value_factors.py

v3 변경사항:
  - 십분위 점수 (0~10) — 원본 FINAL_CODE 완전 복원
  - 대형/중소형 가중치 분리 (원본 그대로)
  - EBITDA 원시값으로 EV/EBITDA 재계산
  - 중소형: PRICE_M + NDEBT_EBITDA + CURRENT 추가
  - 회귀모델④: F_EV/EBIT vs F_EBITG (EBIT 기준, 원본 동일)

점수 구성 (원본 FINAL_CODE_LARGE_02 / SMALL_02):
  ─────────────────────────────────────────────
  공통 (대형/중소형 동일):
    밸류: T_PER 5%, F_PER 5%, T_EVEBITDA 5%, F_EVEBITDA 5%,
          T_PBR 5%, F_PBR 5%, T_PCF 5%           = 35%
    회귀: ATT_PBR 5%, ATT_EVIC 5%,
          ATT_PER 10%, ATT_EVEBIT 10%             = 30%
    성장: T_SPSG 10%, F_SPSG 10%                  = 20%

  대형만:  F_EPS_M 15%                            = 15%
  중소만:  PRICE_M 5%, NDEBT_EBITDA 5%, CURRENT 5% = 15%
  ─────────────────────────────────────────────

십분위 규칙 (원본 동일):
  Rule1 (낮을수록 좋음): 밸류 멀티플 → ≤0/NaN=0, P10↓=10 ... P90↑=1
  Rule2 (높을수록 좋음): 회귀/성장/CURRENT → P90↑=10 ... P10↓=1, NaN=0
  Rule3 (낮을수록 좋음): PRICE_M/NDEBT_EBITDA → P90↑=1 ... P10↓=10, NaN=0
"""
import sqlite3, sys, numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DB_PATH, QUALITY_FILTER

LARGE_CAP_CUTOFF = 200

# ─── 가중치 (합계 1.0) ───
# 밸류35 / 회귀30 / 성장20 / 차별화15 (원래)
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

OUTLIER_FILTERS = {
    "pbr_roe":       {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 20},
    "evic_roic":     {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 51},
    "fper_epsg":     {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60},
    "fevebit_ebitg": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60},
}

FINANCE_TYPES = ["금융업", "은행업", "보험업", "증권업", "여신전문금융업", "기타금융업"]

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

# ═══════════════════════════════════════════════════════
# 십분위 점수 (원본 FINAL_CODE 동일)
# ═══════════════════════════════════════════════════════

def _pcts(series, only_positive=False):
    valid = series[(series > 0) & series.notna()] if only_positive else series[series.notna()]
    if len(valid) < 20:
        return None
    return {p: valid.quantile(p/100) for p in range(10, 100, 10)}

def decile_rule1(series):
    """낮을수록 좋음 (밸류멀티플). ≤0/NaN→0, P10↓→10 ... P90↑→1"""
    P = _pcts(series, only_positive=True)
    if P is None: return pd.Series(0, index=series.index)
    def s(v):
        if pd.isna(v) or v <= 0: return 0
        for i, p in enumerate([90,80,70,60,50,40,30,20,10]):
            if v >= P[p]: return i+1
        return 10
    return series.apply(s)

def decile_rule2(series):
    """높을수록 좋음 (회귀/성장/CURRENT). P90↑→10 ... P10↓→1, NaN→0"""
    P = _pcts(series)
    if P is None: return pd.Series(0, index=series.index)
    def s(v):
        if pd.isna(v): return 0
        for i, p in enumerate([90,80,70,60,50,40,30,20,10]):
            if v >= P[p]: return 10-i
        return 1
    return series.apply(s)

def decile_rule3(series):
    """낮을수록 좋음 (PRICE_M/NDEBT_EBITDA). NaN→0, P90↑→1 ... P10↓→10"""
    P = _pcts(series)
    if P is None: return pd.Series(0, index=series.index)
    def s(v):
        if pd.isna(v): return 0
        for i, p in enumerate([90,80,70,60,50,40,30,20,10]):
            if v >= P[p]: return i+1
        return 10
    return series.apply(s)


# ═══════════════════════════════════════════════════════
# 사분위 점수 (원본 FINAL_CODE_LARGE_02 동일)
# ═══════════════════════════════════════════════════════

def quartile_rule1(series):
    """원본 LARGE_02: 낮을수록 좋음 (사분위 0~4)
    NaN/<=0→0, Q3↑→1, Q2~Q3→2, Q1~Q2→3, <Q1→4"""
    flt = series[(series > 0) & series.notna()]
    if len(flt) < 20: return pd.Series(0, index=series.index)
    Q1, Q2, Q3 = flt.quantile(0.25), flt.quantile(0.5), flt.quantile(0.75)
    def s(v):
        if pd.isna(v) or v <= 0: return 0
        if v >= Q3: return 1
        if v >= Q2: return 2
        if v >= Q1: return 3
        return 4
    return series.apply(s)

def quartile_rule2(series):
    """원본 LARGE_02: 높을수록 좋음 (사분위 0~4)
    Q3↑→4, Q2~Q3→3, Q1~Q2→2, <Q1→1, NaN→0"""
    flt = series[series.notna()]
    if len(flt) < 20: return pd.Series(0, index=series.index)
    Q1, Q2, Q3 = flt.quantile(0.25), flt.quantile(0.5), flt.quantile(0.75)
    def s(v):
        if pd.isna(v): return 0
        if v >= Q3: return 4
        if v >= Q2: return 3
        if v >= Q1: return 2
        return 1
    return series.apply(s)


# ═══════════════════════════════════════════════════════
# 회귀분석
# ═══════════════════════════════════════════════════════

def run_regression(df, x_col, y_col, model_name, size_group):
    col_attr = f"{model_name}_attractiveness"
    df[col_attr] = np.nan

    f = OUTLIER_FILTERS.get(model_name, {})
    x_min, x_max = f.get("x_min", 0), f.get("x_max", 9999)
    y_min, y_max = f.get("y_min", 0), f.get("y_max", 9999)

    vm = (df[x_col].notna() & df[y_col].notna() &
          (df[x_col] > x_min) & (df[x_col] < x_max) &
          (df[y_col] > y_min) & (df[y_col] < y_max))
    valid = df[vm].copy()

    if len(valid) < 20:
        df[col_attr] = 0.0
        return df, {"model": model_name, "size": size_group,
                     "n": len(valid), "r2": 0, "slope": 0, "intercept": 0, "status": "insufficient"}

    slope, intercept, r_value, _, _ = stats.linregress(valid[x_col].values, valid[y_col].values)
    r2 = r_value ** 2
    has_x = df[x_col].notna() & (df[x_col] > x_min)
    fitted = slope * df[x_col] + intercept

    if model_name in ["pbr_roe", "fper_epsg"]:
        mask = has_x & (df[y_col] > 0)
        df.loc[mask, col_attr] = (fitted[mask] / df.loc[mask, y_col]) - 1
        no_x = df[x_col].isna() & (df[y_col] > 0)
        df.loc[no_x, col_attr] = (intercept / df.loc[no_x, y_col]) - 1
    elif model_name in ["evic_roic", "fevebit_ebitg"]:
        if model_name == "evic_roic":
            fitted_ev = fitted * df["ic"]
        else:
            fitted_ev = fitted * df["fwd_ebit"].abs()
        mask = has_x & (df["market_cap"] > 0)
        df.loc[mask, col_attr] = (fitted_ev[mask] - df.loc[mask, "net_debt"]) / df.loc[mask, "market_cap"] - 1

    df[col_attr] = df[col_attr].clip(-1.0, 5.0).fillna(0.0)
    return df, {"model": model_name, "size": size_group,
                "n": len(valid), "r2": round(r2,4), "slope": round(slope,4),
                "intercept": round(intercept,4), "status": "ok"}


# ═══════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════

def calc_valuation_for_date(target_date=None):
    conn = get_db()
    if target_date is None:
        target_date = conn.execute("SELECT MAX(trade_date) FROM daily_price").fetchone()[0]

    print(f"    [Step3] {target_date} 기준 종합 점수 계산...")

    dt = datetime.strptime(target_date, "%Y-%m-%d")
    max_usable_year = dt.year - 1 if dt.month >= 4 else dt.year - 2
    print(f"    공시 기준: {max_usable_year}년 Annual")

    # ─── 1. Trailing 재무 (ebitda 추가) ───
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
        print("    ⚠️ Trailing 재무 없음"); conn.close(); return None

    prev_rev_df = pd.read_sql_query("""
        SELECT stock_code, revenue as prev_revenue
        FROM fnspace_finance WHERE fiscal_quarter='Annual' AND fiscal_year=?
    """, conn, params=(max_usable_year - 1,))

    # ─── 2. Forward ───
    fwd_date = conn.execute("SELECT MAX(trade_date) FROM fnspace_forward WHERE trade_date < ?", (target_date,)).fetchone()[0]
    if not fwd_date:
        fwd_date = conn.execute("SELECT MIN(trade_date) FROM fnspace_forward").fetchone()[0]

    fwd_df = pd.DataFrame()
    if fwd_date:
        fwd_df = pd.read_sql_query("""
            SELECT stock_code, fwd_eps, fwd_per, fwd_ebit, fwd_ebitda,
                   fwd_ev_ebitda, fwd_revenue, fwd_oi, fwd_ni, fwd_roe, fwd_bps
            FROM fnspace_forward WHERE trade_date = ?
        """, conn, params=(fwd_date,))
        print(f"    Forward: {fwd_date}")

    # 3개월 전 Forward EPS
    three_m_ago = (dt - timedelta(days=90)).strftime("%Y-%m-%d")
    fwd_date_3m = conn.execute("SELECT MAX(trade_date) FROM fnspace_forward WHERE trade_date <= ?", (three_m_ago,)).fetchone()[0]
    fwd_3m_df = pd.DataFrame()
    if fwd_date_3m:
        fwd_3m_df = pd.read_sql_query("SELECT stock_code, fwd_eps as fwd_eps_3m FROM fnspace_forward WHERE trade_date=?", conn, params=(fwd_date_3m,))
        print(f"    3M전 Forward: {fwd_date_3m}")

    # ─── 3. 주가 ───
    price_date = conn.execute("SELECT MAX(trade_date) FROM daily_price WHERE trade_date < ?", (target_date,)).fetchone()[0]
    price_df = pd.read_sql_query("""
        SELECT 'A' || dp.stock_code as stock_code, dp.close, dp.market_cap, dp.trade_amount
        FROM daily_price dp WHERE dp.trade_date = ?
    """, conn, params=(price_date,))

    # 3개월 전 주가 (PRICE_M용)
    price_date_3m = conn.execute("SELECT MAX(trade_date) FROM daily_price WHERE trade_date <= ?", (three_m_ago,)).fetchone()[0]
    price_3m_df = pd.DataFrame()
    if price_date_3m:
        price_3m_df = pd.read_sql_query("SELECT 'A'||stock_code as stock_code, close as close_3m FROM daily_price WHERE trade_date=?", conn, params=(price_date_3m,))

    print(f"    주가: {price_date}")

    master_df = pd.read_sql_query("SELECT stock_code, stock_name, market, sec_cd_nm, finacc_typ FROM fnspace_master", conn)

    # ─── 병합 ───
    merged = fin_df.merge(price_df, on="stock_code", how="inner")
    merged = merged.merge(master_df, on="stock_code", how="inner")
    for df_extra in [fwd_df, prev_rev_df, fwd_3m_df, price_3m_df]:
        if not df_extra.empty:
            merged = merged.merge(df_extra, on="stock_code", how="left")
    merged = merged[(merged["market_cap"] > 0) & (merged["close"] > 0)].copy()
    if len(merged) == 0:
        print("    ⚠️ 병합 0건"); conn.close(); return None

    # 단위 통일 (천원→원)
    for c in ["ev","ic","ebit","ebitda","net_debt","interest_debt","total_equity","revenue","operating_income","net_income"]:
        if c in merged.columns: merged[c] = merged[c] * 1000
    for c in ["fwd_ebit","fwd_ebitda","fwd_revenue","fwd_oi","fwd_ni"]:
        if c in merged.columns: merged[c] = merged[c] * 1000
    if "prev_revenue" in merged.columns: merged["prev_revenue"] *= 1000

    merged = merged[~merged["finacc_typ"].isin(FINANCE_TYPES)].copy()
    print(f"    금융업 제외 후: {len(merged)}종목")

    # ═══════════════════════════════════════════════════
    # 밸류에이션 재계산 (현재 주가 기준)
    # ═══════════════════════════════════════════════════
    m = merged["bps"].notna() & (merged["bps"] > 0)
    merged["pbr"] = np.where(m, merged["close"] / merged["bps"], np.nan)

    m = merged["eps"].notna() & (merged["eps"] > 0)
    merged["t_per"] = np.where(m, merged["close"] / merged["eps"], np.nan)

    m = merged["net_debt"].notna()
    merged["ev"] = np.where(m, merged["market_cap"] + merged["net_debt"], np.nan)

    # T_EV/EBITDA — EBITDA 원시값 사용 (없으면 EBIT fallback)
    if "ebitda" in merged.columns and merged["ebitda"].notna().sum() > 10:
        m = (merged["ev"] > 0) & merged["ebitda"].notna() & (merged["ebitda"] > 0)
        merged["t_ev_ebitda"] = np.where(m, merged["ev"] / merged["ebitda"], np.nan)
    else:
        m = (merged["ev"] > 0) & merged["ebit"].notna() & (merged["ebit"] > 0)
        merged["t_ev_ebitda"] = np.where(m, merged["ev"] / merged["ebit"], np.nan)
        print("    ⚠️ ebitda 없음 → ebit fallback")

    m = (merged["ev"] > 0) & merged["ic"].notna() & (merged["ic"] > 0)
    merged["ev_ic"] = np.where(m, merged["ev"] / merged["ic"], np.nan)

    if "fwd_eps" in merged.columns:
        m = merged["fwd_eps"].notna() & (merged["fwd_eps"] > 0)
        merged["f_per"] = np.where(m, merged["close"] / merged["fwd_eps"], np.nan)
    else: merged["f_per"] = np.nan

    # F_EV/EBITDA — Forward EBITDA 원시값 (없으면 fwd_ebit fallback)
    if "fwd_ebitda" in merged.columns and merged["fwd_ebitda"].notna().sum() > 10:
        m = (merged["ev"] > 0) & merged["fwd_ebitda"].notna() & (merged["fwd_ebitda"] > 0)
        merged["f_ev_ebitda"] = np.where(m, merged["ev"] / merged["fwd_ebitda"], np.nan)
    elif "fwd_ebit" in merged.columns:
        m = (merged["ev"] > 0) & merged["fwd_ebit"].notna() & (merged["fwd_ebit"] > 0)
        merged["f_ev_ebitda"] = np.where(m, merged["ev"] / merged["fwd_ebit"], np.nan)
    else: merged["f_ev_ebitda"] = np.nan

    # F_EV/EBIT (모델④ — EBIT 기준, EV/EBITDA와 별개)
    if "fwd_ebit" in merged.columns:
        m = (merged["ev"] > 0) & merged["fwd_ebit"].notna() & (merged["fwd_ebit"] > 0)
        merged["f_ev_ebit"] = np.where(m, merged["ev"] / merged["fwd_ebit"], np.nan)
    else: merged["f_ev_ebit"] = np.nan

    # F_PBR = 주가 / Forward BPS
    if "fwd_bps" in merged.columns:
        m = merged["fwd_bps"].notna() & (merged["fwd_bps"] > 0)
        merged["f_pbr"] = np.where(m, merged["close"] / merged["fwd_bps"], np.nan)
    else:
        merged["f_pbr"] = np.nan

    # T_PCF = P/FCF2 (FnSpace에서 이미 배수로 제공, 양수만 사용)
    if "pcf" in merged.columns:
        m = merged["pcf"].notna() & (merged["pcf"] > 0)
        merged["t_pcf"] = np.where(m, merged["pcf"], np.nan)
    else:
        merged["t_pcf"] = np.nan

    # ─── 파생 변수 ───
    if "fwd_eps" in merged.columns:
        m = merged["eps"].notna() & (merged["eps"].abs() > 0) & merged["fwd_eps"].notna()
        merged["f_epsg"] = np.where(m, (merged["fwd_eps"]-merged["eps"])/merged["eps"].abs()*100, np.nan)
    else: merged["f_epsg"] = np.nan

    if "fwd_ebit" in merged.columns:
        m = merged["ebit"].notna() & (merged["ebit"].abs() > 0) & merged["fwd_ebit"].notna()
        merged["f_ebitg"] = np.where(m, (merged["fwd_ebit"]-merged["ebit"])/merged["ebit"].abs()*100, np.nan)
    else: merged["f_ebitg"] = np.nan

    if "prev_revenue" in merged.columns:
        m = merged["revenue"].notna() & merged["prev_revenue"].notna() & (merged["prev_revenue"].abs()>0)
        merged["t_spsg"] = np.where(m, (merged["revenue"]-merged["prev_revenue"])/merged["prev_revenue"].abs()*100, np.nan)
    else: merged["t_spsg"] = np.nan

    if "fwd_revenue" in merged.columns:
        m = merged["revenue"].notna() & (merged["revenue"].abs()>0) & merged["fwd_revenue"].notna()
        merged["f_spsg"] = np.where(m, (merged["fwd_revenue"]-merged["revenue"])/merged["revenue"].abs()*100, np.nan)
    else: merged["f_spsg"] = np.nan

    # 대형: F_EPS_M
    if "fwd_eps_3m" in merged.columns and "fwd_eps" in merged.columns:
        m = merged["fwd_eps_3m"].notna() & (merged["fwd_eps_3m"].abs()>0) & merged["fwd_eps"].notna()
        merged["f_eps_m"] = np.where(m, (merged["fwd_eps"]-merged["fwd_eps_3m"])/merged["fwd_eps_3m"].abs()*100, np.nan)
    else: merged["f_eps_m"] = np.nan

    # 중소: PRICE_M
    if "close_3m" in merged.columns:
        m = merged["close_3m"].notna() & (merged["close_3m"]>0)
        merged["price_m"] = np.where(m, (merged["close"]-merged["close_3m"])/merged["close_3m"]*100, np.nan)
    else: merged["price_m"] = np.nan

    # 중소: NDEBT_EBITDA
    ebitda_col = "ebitda" if ("ebitda" in merged.columns and merged["ebitda"].notna().sum()>10) else "ebit"
    m = merged[ebitda_col].notna() & (merged[ebitda_col]>0) & merged["net_debt"].notna()
    merged["ndebt_ebitda"] = np.where(m, merged["net_debt"]/merged[ebitda_col], np.nan)

    merged["current_ratio"] = np.nan  # 추후 FnSpace에서 추가 가능

    # ═══════════════════════════════════════════════════
    # 대형/중소형 분리
    # ═══════════════════════════════════════════════════
    merged = merged.sort_values("market_cap", ascending=False).reset_index(drop=True)
    cutoff = min(LARGE_CAP_CUTOFF, len(merged)//3)
    merged["size_group"] = "mid_small"
    merged.loc[:cutoff-1, "size_group"] = "large"
    print(f"    대형: {(merged['size_group']=='large').sum()}, 중소형: {(merged['size_group']=='mid_small').sum()}")

    # ═══════════════════════════════════════════════════
    # 회귀분석 (4모델 × 2그룹)
    # ═══════════════════════════════════════════════════
    all_info = []
    for sg in ["large", "mid_small"]:
        mask = merged["size_group"]==sg; gdf = merged[mask].copy(); idx = merged[mask].index
        for model, x, y in [("pbr_roe","roe","pbr"), ("evic_roic","roic","ev_ic"),
                             ("fper_epsg","f_epsg","f_per"), ("fevebit_ebitg","f_ebitg","f_ev_ebit")]:
            gdf, info = run_regression(gdf, x, y, model, sg)
            all_info.append(info)
            merged.loc[idx, f"{model}_attractiveness"] = gdf[f"{model}_attractiveness"].values

    print(f"\n    회귀분석:")
    print(f"    {'모델':<20} {'그룹':<10} {'N':>5} {'R2':>8} {'기울기':>10}")
    print(f"    {'─'*55}")
    for i in all_info:
        print(f"    {i['model']:<20} {i['size']:<10} {i['n']:>5} {i['r2']:>8.4f} {i['slope']:>10.4f}")

    # ═══════════════════════════════════════════════════
    # 십분위 점수 (그룹별)
    # ═══════════════════════════════════════════════════
    for sg in ["large", "mid_small"]:
        idx = merged["size_group"]==sg

        # Rule1: 밸류 멀티플 (낮을수록 좋음)
        for c in ["t_per","f_per","t_ev_ebitda","f_ev_ebitda","pbr","f_pbr","t_pcf"]:
            merged.loc[idx, f"{c}_score"] = decile_rule1(merged.loc[idx, c]).values

        # Rule2: 회귀 + 성장 + CURRENT + F_EPS_M (높을수록 좋음)
        for m in ["pbr_roe","evic_roic","fper_epsg","fevebit_ebitg"]:
            merged.loc[idx, f"{m}_score"] = decile_rule2(merged.loc[idx, f"{m}_attractiveness"]).values
        for c in ["t_spsg","f_spsg"]:
            merged.loc[idx, f"{c}_score"] = decile_rule2(merged.loc[idx, c]).values
        merged.loc[idx, "f_eps_m_score"] = decile_rule2(merged.loc[idx, "f_eps_m"]).values
        merged.loc[idx, "current_ratio_score"] = decile_rule2(merged.loc[idx, "current_ratio"]).values

        # Rule3: PRICE_M, NDEBT_EBITDA (낮을수록 좋음)
        merged.loc[idx, "price_m_score"] = decile_rule3(merged.loc[idx, "price_m"]).values
        merged.loc[idx, "ndebt_ebitda_score"] = decile_rule3(merged.loc[idx, "ndebt_ebitda"]).values

    # ═══════════════════════════════════════════════════
    # TOTAL_SCORE (대형/중소형 별도 가중치)
    # ═══════════════════════════════════════════════════
    SCM = {
        "T_PER":"t_per_score", "F_PER":"f_per_score",
        "T_EVEBITDA":"t_ev_ebitda_score", "F_EVEBITDA":"f_ev_ebitda_score",
        "T_PBR":"pbr_score", "F_PBR":"f_pbr_score", "T_PCF":"t_pcf_score",
        "ATT_PBR":"pbr_roe_score", "ATT_EVIC":"evic_roic_score",
        "ATT_PER":"fper_epsg_score", "ATT_EVEBIT":"fevebit_ebitg_score",
        "T_SPSG":"t_spsg_score", "F_SPSG":"f_spsg_score",
        "F_EPS_M":"f_eps_m_score",
        "PRICE_M":"price_m_score", "NDEBT_EBITDA":"ndebt_ebitda_score", "CURRENT":"current_ratio_score",
    }
    for c in SCM.values():
        if c not in merged.columns: merged[c] = 0
        merged[c] = merged[c].fillna(0)

    for sg, W in [("large", WEIGHTS_LARGE), ("mid_small", WEIGHTS_SMALL)]:
        mask = merged["size_group"]==sg
        merged.loc[mask, "total_raw"] = sum(merged.loc[mask, SCM[k]] * w for k, w in W.items() if k in SCM)

    # 0~100 (max=10*1.0=10.0)
    merged["value_score"] = (merged["total_raw"] / 10.0 * 100).round(1).clip(0, 100)

    # ═══════════════════════════════════════════════════
    # 원본 FINAL_CODE 사분위 점수 (LARGE만 사분위, SMALL은 십분위 그대로)
    # ═══════════════════════════════════════════════════
    for sg in ["large", "mid_small"]:
        idx = merged["size_group"] == sg
        if sg == "large":
            # Rule1: 밸류 멀티플 (사분위, 낮을수록 좋음)
            for c in ["t_per","f_per","t_ev_ebitda","f_ev_ebitda","pbr","f_pbr","t_pcf"]:
                merged.loc[idx, f"{c}_qscore"] = quartile_rule1(merged.loc[idx, c]).values
            # Rule2: 회귀+성장+F_EPS_M (사분위, 높을수록 좋음)
            for m in ["pbr_roe","evic_roic","fper_epsg","fevebit_ebitg"]:
                merged.loc[idx, f"{m}_qscore"] = quartile_rule2(merged.loc[idx, f"{m}_attractiveness"]).values
            for c in ["t_spsg","f_spsg"]:
                merged.loc[idx, f"{c}_qscore"] = quartile_rule2(merged.loc[idx, c]).values
            merged.loc[idx, "f_eps_m_qscore"] = quartile_rule2(merged.loc[idx, "f_eps_m"]).values
        else:
            # SMALL은 기존 십분위와 동일 → _qscore = _score 복사
            for c in SCM.values():
                col_q = c.replace("_score", "_qscore")
                if col_q not in merged.columns:
                    merged[col_q] = 0
                merged.loc[idx, col_q] = merged.loc[idx, c].values

    # 원본 가중합 계산
    QSCM = {k: v.replace("_score", "_qscore") for k, v in SCM.items()}
    for c in QSCM.values():
        if c not in merged.columns:
            merged[c] = 0
        merged[c] = merged[c].fillna(0)

    for sg, W in [("large", WEIGHTS_LARGE), ("mid_small", WEIGHTS_SMALL)]:
        mask = merged["size_group"] == sg
        merged.loc[mask, "total_raw_orig"] = sum(
            merged.loc[mask, QSCM[k]] * w for k, w in W.items() if k in QSCM)

    # 정규화: LARGE는 max=4 (사분위), SMALL은 max=10 (십분위)
    merged["value_score_orig"] = 0.0
    large_mask = merged["size_group"] == "large"
    small_mask = merged["size_group"] == "mid_small"
    merged.loc[large_mask, "value_score_orig"] = (
        merged.loc[large_mask, "total_raw_orig"] / 4.0 * 100).round(1).clip(0, 100)
    merged.loc[small_mask, "value_score_orig"] = (
        merged.loc[small_mask, "total_raw_orig"] / 10.0 * 100).round(1).clip(0, 100)

    # ═══════════════════════════════════════════════════
    # ATT2 점수 (ATT_PBR + ATT_EVIC, 50:50)
    # A0와 동일 체계: 대형=사분위/4, 중소=십분위/10
    # ═══════════════════════════════════════════════════
    merged["att2_score"] = 0.0
    merged.loc[large_mask, "att2_score"] = (
        (merged.loc[large_mask, "pbr_roe_qscore"] * 0.5 +
         merged.loc[large_mask, "evic_roic_qscore"] * 0.5) / 4.0 * 100
    ).round(1).clip(0, 100)
    merged.loc[small_mask, "att2_score"] = (
        (merged.loc[small_mask, "pbr_roe_qscore"] * 0.5 +
         merged.loc[small_mask, "evic_roic_qscore"] * 0.5) / 10.0 * 100
    ).round(1).clip(0, 100)

    # ═══════════════════════════════════════════════════
    # 퀄리티 필터
    # ═══════════════════════════════════════════════════
    merged["quality_pass"] = 1; merged["quality_fail_reason"] = ""
    for kw in ["스팩","SPAC","ETF","ETN","리츠","REIT"]:
        m = merged["stock_name"].str.contains(kw, case=False, na=False)
        merged.loc[m, "quality_pass"] = 0; merged.loc[m, "quality_fail_reason"] += "스팩/ETF/리츠; "
    m = merged["operating_income"].notna() & (merged["operating_income"]<=0)
    merged.loc[m, "quality_pass"] = 0; merged.loc[m, "quality_fail_reason"] += "영업적자; "
    m = merged["roe"].notna() & (merged["roe"]<=0)
    merged.loc[m, "quality_pass"] = 0; merged.loc[m, "quality_fail_reason"] += "ROE<=0; "
    min_vol = QUALITY_FILTER.get("min_avg_volume_20d", 100_000_000)
    m = merged["trade_amount"] < min_vol
    merged.loc[m, "quality_pass"] = 0; merged.loc[m, "quality_fail_reason"] += "거래대금부족; "

    # ─── DB 저장 ───
    merged["calc_date"] = target_date
    for _, row in merged.iterrows():
        code = row["stock_code"]
        if code.startswith("A"): code = code[1:]
        conn.execute("""
            INSERT OR REPLACE INTO valuation_factors
            (stock_code, calc_date, per, pbr, psr, ev_ebitda, dividend_yield,
             per_rank, pbr_rank, psr_rank, ev_ebitda_rank, div_yield_rank,
             value_score, value_score_orig, att2_score,
             quality_pass, quality_fail_reason, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now','localtime'))
        """, (code, target_date,
              row.get("t_per"), row.get("pbr"), row.get("psr"),
              row.get("t_ev_ebitda"), row.get("div_yield"),
              round(row.get("pbr_roe_attractiveness",0)*100,1),
              round(row.get("evic_roic_attractiveness",0)*100,1),
              round(row.get("fper_epsg_attractiveness",0)*100,1),
              round(row.get("fevebit_ebitg_attractiveness",0)*100,1),
              round(row.get("total_raw",0)*10,1),
              row["value_score"], row["value_score_orig"], row["att2_score"],
              int(row["quality_pass"]), row["quality_fail_reason"]))
    conn.commit(); conn.close()
    return merged


# ═══════════════════════════════════════════════════════
# 출력
# ═══════════════════════════════════════════════════════

def show_candidate_pool(df):
    print("\n" + "=" * 60)
    print("🎯 Step 3: 후보 풀 (십분위, 대형/중소형 분리)")
    print("=" * 60)

    qp = df[df["quality_pass"]==1]; qf = df[df["quality_pass"]==0]
    print(f"\n  전체: {len(df)}, 통과: {len(qp)}, 탈락: {len(qf)}")
    if len(qf)>0:
        reasons = qf["quality_fail_reason"].str.split("; ").explode()
        for r, c in reasons[reasons!=""].value_counts().items(): print(f"    {r}: {c}건")

    for sg, label, W in [("large","대형",WEIGHTS_LARGE), ("mid_small","중소형",WEIGHTS_SMALL)]:
        g = qp[qp["size_group"]==sg]
        if len(g)==0: continue
        print(f"\n  [{label}] {len(g)}종목, 평균 score: {g['value_score'].mean():.1f}")
        if sg=="large":
            has = g["f_eps_m"].notna().sum()
            print(f"    F_EPS_M 커버: {has}/{len(g)}")
        else:
            print(f"    PRICE_M 커버: {g['price_m'].notna().sum()}/{len(g)}")
            print(f"    NDEBT_EBITDA 커버: {g['ndebt_ebitda'].notna().sum()}/{len(g)}")
        # 공통
        print(f"    F_PBR 커버: {g['f_pbr'].notna().sum()}/{len(g)}")
        print(f"    T_PCF 커버: {g['t_pcf'].notna().sum()}/{len(g)}")
        print(f"    F_EBITDA 커버: {(g['fwd_ebitda'].notna().sum() if 'fwd_ebitda' in g.columns else 0)}/{len(g)}")

    top = qp.nlargest(20, "value_score")
    print(f"\n  📊 TOP 20:")
    print(f"  {'#':<3} {'종목명':<14} {'그룹':<5} {'종합':>5} {'밸류':>5} {'회귀':>5} {'성장':>5} {'D':>5} {'ROE':>6} {'PBR':>6}")
    print(f"  {'─'*72}")
    for i, (_,r) in enumerate(top.iterrows(), 1):
        name = r.get("stock_name","")[:12]
        sg = "대형" if r["size_group"]=="large" else "중소"
        mult = sum(r.get(c,0) for c in ["t_per_score","f_per_score","t_ev_ebitda_score","f_ev_ebitda_score","pbr_score","f_pbr_score","t_pcf_score"])
   
        reg = sum(r.get(c,0) for c in ["pbr_roe_score","evic_roic_score","fper_epsg_score","fevebit_ebitg_score"])
        grow = sum(r.get(c,0) for c in ["t_spsg_score","f_spsg_score"])
        d = r.get("f_eps_m_score",0) if r["size_group"]=="large" else sum(r.get(c,0) for c in ["price_m_score","ndebt_ebitda_score","current_ratio_score"])
        roe = f"{r['roe']:.1f}" if pd.notna(r.get('roe')) else "N/A"
        pbr = f"{r['pbr']:.2f}" if pd.notna(r.get('pbr')) else "N/A"
        print(f"  {i:<3} {name:<14} {sg:<5} {r['value_score']:>5.1f} {mult:>5.0f} {reg:>5.0f} {grow:>5.0f} {d:>5.0f} {roe:>6} {pbr:>6}")

    print(f"\n  ✅ Step 3 완료!")
    return qp


if __name__ == "__main__":
    print("🚀 Step 3 v3: 십분위 + 대형/중소형 분리 + EBITDA 통일")
    print(f"   DB: {DB_PATH}")
    print(f"   대형: 밸류35% + 회귀30% + 성장20% + F_EPS_M 15%")
    print(f"   중소: 밸류35% + 회귀30% + 성장20% + 건전성15%")
    df = calc_valuation_for_date()
    if df is not None:
        show_candidate_pool(df)
