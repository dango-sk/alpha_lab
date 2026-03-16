"""
Strategy: A0 기본 전략 (대형주)
Created: 2026-02-23
Description: 원본 사분위 밸류 전략 (밸류 35% + 회귀 30% + 성장 20% + 차별화 15%)

설계 원리:
  - 대형주(시총 상위 200) 유니버스 한정. KOSPI 200과 공정 비교 가능.
  - 사분위(Quartile, 0~4점) 채점. 종목 수가 적어 십분위 구간이 촘촘해짐을 방지.
  - 4개 회귀 모델(PBR-ROE, EV/IC-ROIC, F.PER-이익성장, F.EV/EBIT-EBIT성장)로 내재가치 괴리도 측정.
"""

# ─── 채점 방식 ───
SCORING_MODE = {
    "large": "quartile",   # 대형주: 사분위 (0~4점)
}

# ─── 팩터 가중치 (합계 1.0) ───
WEIGHTS_LARGE = {
    "T_PER": 0.05,
    "F_PER": 0.05,
    "T_EVEBITDA": 0.05,
    "F_EVEBITDA": 0.05,
    "T_PBR": 0.05,
    "F_PBR": 0.05,
    "T_PCF": 0.05,
    "ATT_PBR": 0.05,
    "ATT_EVIC": 0.05,
    "ATT_PER": 0.10,
    "ATT_EVEBIT": 0.10,
    "T_SPSG": 0.10,
    "F_SPSG": 0.10,
    "F_EPS_M": 0.15,
}

WEIGHTS_SMALL = {}

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
    "weight_cap_pct": 10,
}

# ─── 퀄리티 필터 ───
QUALITY_FILTER = {
    "exclude_spac_etf_reit": True,
    "require_positive_oi": True,
    "require_positive_roe": True,
    "min_avg_volume": 500_000_000,
}
