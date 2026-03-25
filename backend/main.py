"""
Alpha Lab Quant Dashboard — FastAPI Backend

Wraps the existing Streamlit-based data layer (lib/) and exposes it as REST API.
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Optional

# ──────────────────────────────────────────────
# Mock streamlit BEFORE any lib imports
# ──────────────────────────────────────────────
_mock_st = ModuleType("streamlit")


def _passthrough_decorator(*args, **kwargs):
    """Works as both @cache_data and @cache_data(ttl=...) """
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


_mock_st.cache_data = _passthrough_decorator
_mock_st.cache_resource = _passthrough_decorator
_mock_st.error = lambda *a, **kw: None
_mock_st.stop = lambda: (_ for _ in ()).throw(RuntimeError("st.stop called"))
_mock_st.warning = lambda *a, **kw: None
_mock_st.info = lambda *a, **kw: None
_mock_st.session_state = {}
_mock_st.rerun = lambda: None
_mock_st.spinner = lambda *a, **kw: __import__("contextlib").nullcontext()
sys.modules["streamlit"] = _mock_st

# ──────────────────────────────────────────────
# Path setup
# ──────────────────────────────────────────────
ALPHA_LAB_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ALPHA_LAB_DIR))
sys.path.insert(0, str(ALPHA_LAB_DIR / "scripts"))

# ──────────────────────────────────────────────
# Imports from existing lib
# ──────────────────────────────────────────────
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import BACKTEST_CONFIG, ANTHROPIC_API_KEY
from lib.data import (
    load_all_results,
    load_all_robustness_results,
    get_holdings,
    get_monthly_attribution,
    get_portfolio_characteristics,
    get_portfolio_turnover,
    get_latest_price_date,
    get_first_entry_dates,
    list_strategies,
    load_strategy,
    save_strategy,
    delete_strategy,
    run_strategy_backtest,
    compute_regime_analysis,
    run_regime_combo_backtest,
    STRATEGY_LABELS,
    STRATEGY_COLORS,
    ALL_KEYS,
    BASE_STRATEGY_WEIGHTS,
)
from lib.ai import (
    is_ai_available,
    _get_client,
    MODEL_FAST,
    MODEL_SMART,
    STRATEGY_TOOLS,
    chat_strategy_modification,
    _STRATEGY_SYSTEM_TEMPLATE,
)
from lib.factor_engine import DEFAULT_STRATEGY_CODE

# ──────────────────────────────────────────────
# JSON serialization helpers
# ──────────────────────────────────────────────


def _convert_for_json(obj):
    """Recursively convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_convert_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _convert_for_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return _convert_for_json(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


# ──────────────────────────────────────────────
# Simple TTL cache
# ──────────────────────────────────────────────
_cache: dict[str, tuple[float, object]] = {}
_DEFAULT_TTL = 3600  # 1 hour


def _cached(key: str, fn, ttl: int = _DEFAULT_TTL):
    now = time.time()
    if key in _cache:
        ts, val = _cache[key]
        if now - ts < ttl:
            return val
    val = fn()
    _cache[key] = (now, val)
    return val


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(title="Alpha Lab API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Chat system prompt
# ──────────────────────────────────────────────
_SYSTEM_GENERAL = """당신은 이현자산운용의 시니어 퀀트 애널리스트 AI 어시스턴트입니다.
Alpha Lab 대시보드의 모든 데이터에 접근할 수 있으며, 사용자의 질문에 답합니다.

## 당신이 알고 있는 데이터

### 백테스트 결과 (backtest_cache)
- 전략별 누적수익률, CAGR, MDD, Sharpe ratio, 월별수익률, 포트폴리오 가치 추이
- 전략: A0(기존 멀티팩터 전략), KOSPI(벤치마크), 그리고 사용자가 실험실에서 생성한 커스텀 전략
- 유니버스: KOSPI, KOSPI+KOSDAQ | 리밸런싱: 월간, 격주

### 포트폴리오 보유종목 (holdings)
- 종목코드, 종목명, 섹터, 비중(%), 점수, PER, PBR, EV/EBITDA, 시가총액
- HHI(허핀달지수), Top5 비중, 섹터별 비중, 시가총액 분포(초대형/대형/중형/소형)
- 리밸런싱 변화: 신규편입/편출/유지 종목, 회전율

### 일별 주가 데이터 (daily_price)
- 전 종목의 일별 시가/고가/저가/종가/거래량/시가총액
- 기간: 2018년~현재

### 재무 데이터 (finance)
- 연간/분기 실적: 매출액, 영업이익, 당기순이익, 자산총계, 부채총계, 자본총계
- 밸류에이션: PER, PBR, EV/EBITDA, PCF, ROE, ROA, 배당수익률

### 컨센서스 데이터 (consensus)
- 애널리스트 추정치: Forward EPS, Forward PER, Forward EV/EBITDA, 목표주가
- EPS 모멘텀(수정 방향), 매출성장률 추정

### 통계 검증
- IS/OOS 분할 검증, 벤치마크 대비 유의성 검정(t-test, 부트스트랩)
- 롤링 윈도우 초과수익률, 강건성 검증 결과

## SQL 데이터 조회 기능
컨텍스트에 포함되지 않은 상세 데이터(보유종목, 개별 종목 주가, 재무제표 등)가 필요하면
<sql>SELECT ...</sql> 태그로 PostgreSQL SELECT 쿼리를 작성하세요. 시스템이 자동 실행합니다.
중요: SQL 쿼리는 반드시 <sql>...</sql> 태그 안에 넣고, 사용자에게 쿼리를 보여주지 마세요. 결과만 자연스럽게 설명하세요.

### 주요 테이블 스키마 (모든 테이블은 alpha_lab 스키마)
- **alpha_lab.backtest_cache**: name, universe, rebal_type, results_json(jsonb), holdings_json(jsonb)
  - holdings_json->'holdings'->'YYYY-MM-DD' = [{"종목코드","종목명","섹터","비중(%)","점수","PER","PBR","EV/EBITDA","시가총액","value_score"}, ...]
- **alpha_lab.daily_price**: stock_code, trade_date, open, high, low, close, volume, market_cap
- **alpha_lab.fnspace_finance**: stock_code, fiscal_year, fiscal_quarter, revenue, operating_profit, net_income, total_assets, total_liabilities, total_equity, per, pbr, ev_ebitda, pcf, roe, roa, dividend_yield
- **alpha_lab.fnspace_forward**: trade_date, stock_code, f_per, f_pbr, f_ev_ebitda, f_eps, target_price, f_eps_m, f_spsg
- **alpha_lab.fnspace_consensus_daily**: trade_date, stock_code, f_eps_1y, f_eps_2y, f_per_1y, f_revenue_1y, f_op_1y
- **alpha_lab.universe**: rebal_date, rebal_type, stock_code, stock_name, sector, market_cap, size_group
- **alpha_lab.corp_master**: stock_code, stock_name, market, sector, industry

### SQL 쿼리 작성 규칙
- SELECT/WITH만 허용
- 결과가 너무 크지 않도록 LIMIT 사용
- 간단한 보유종목 질문은 컨텍스트의 최신 보유종목 요약을 먼저 참고하고, SQL 없이 답변

### 보유종목 JSONB 쿼리 예시
섹터별 비중 조회:
WITH latest AS (
  SELECT MAX(k) as d FROM alpha_lab.backtest_cache, jsonb_object_keys(holdings_json->'holdings') k
  WHERE name='A0' AND universe='KOSPI' AND rebal_type='monthly'
)
SELECT h->>'섹터' as sector, COUNT(*) as cnt, ROUND(SUM((h->>'비중(%)')::numeric), 1) as weight_pct
FROM alpha_lab.backtest_cache bc, latest l, LATERAL jsonb_array_elements(bc.holdings_json->'holdings'->l.d) h
WHERE bc.name='A0' AND bc.universe='KOSPI' AND bc.rebal_type='monthly'
GROUP BY h->>'섹터' ORDER BY weight_pct DESC

## 규칙
- 한국어로 응답합니다.
- 전문 금융 용어를 사용하되, 코드 변수명이나 프로그래밍 용어는 절대 사용하지 마세요.
- 답변은 간결하고 핵심적으로. 수치를 근거로 들어 답변하세요.
- 사용자가 특정 종목이나 데이터를 물어보면, 컨텍스트 데이터를 먼저 확인하고, 없으면 SQL 쿼리로 조회하세요.
- 사용자가 새로운 전략을 만들어달라고 하면, factor_engine 형식의 파이썬 코드를 ```python 블록으로 생성하세요.
- 생성한 전략 코드는 사용자가 "전략 실험실에서 열기" 버튼으로 바로 백테스트할 수 있습니다.
- 전략 코드를 제안할 때 마지막에 반드시 안내하세요: "전략 실험실에서 유니버스(KOSPI/KOSPI+KOSDAQ), 리밸런싱 주기(월간/격주), Top N 종목 수, Weight Cap 등을 자유롭게 바꿔가며 실험해보세요."

## 전략 코드 형식 (factor_engine)

반드시 아래 6개 변수를 모두 정의해야 합니다. import 문은 절대 사용하지 마세요.

### 사용 가능한 팩터 (이 목록에서만 선택)
밸류(rule1, 낮을수록 좋음): T_PER→t_per_score, F_PER→f_per_score, T_PBR→pbr_score, F_PBR→f_pbr_score, T_EVEBITDA→t_ev_ebitda_score, F_EVEBITDA→f_ev_ebitda_score, T_PCF→t_pcf_score
성장/모멘텀(rule2, 높을수록 좋음): T_SPSG→t_spsg_score, F_SPSG→f_spsg_score, F_EPS_M→f_eps_m_score
가격모멘텀(rule3, 낮을수록 좋음): PRICE_M→price_m_score
회귀매력도(rule2, 높을수록 좋음): ATT_PBR→pbr_roe_attractiveness_score, ATT_EVIC→evic_roic_attractiveness_score, ATT_PER→fper_epsg_attractiveness_score, ATT_EVEBIT→fevebit_ebitg_attractiveness_score
퀄리티(rule3): NDEBT_EBITDA→ndebt_ebitda_score, CURRENT→current_ratio_score

### 회귀모델 (ATT_* 팩터 사용 시 반드시 포함)
ATT_PBR → ("pbr_roe", "roe", "pbr", "ratio")
ATT_EVIC → ("evic_roic", "roic", "ev_ic", "ev_equity")
ATT_PER → ("fper_epsg", "f_epsg", "f_per", "ratio")
ATT_EVEBIT → ("fevebit_ebitg", "f_ebitg", "f_ev_ebit", "ev_equity_ebit")

### 코드 템플릿 예시
```python
\"\"\"
Strategy: 전략 이름
Description: 전략 설명
\"\"\"

SCORING_MODE = {"large": "quartile"}  # "quartile"(0~4) 또는 "decile"(0~10)

WEIGHTS_LARGE = {
    "T_PER": 0.20, "T_PBR": 0.20, "T_EVEBITDA": 0.20,
    "ATT_PBR": 0.10, "ATT_PER": 0.10,
    "F_EPS_M": 0.20,
}  # 합계 = 1.0

WEIGHTS_SMALL = {}  # 대형주만이면 빈 딕셔너리

REGRESSION_MODELS = [
    ("pbr_roe", "roe", "pbr", "ratio"),
    ("fper_epsg", "f_epsg", "f_per", "ratio"),
]  # ATT_* 팩터에 해당하는 모델만 포함. 안 쓰면 []

OUTLIER_FILTERS = {
    "pbr_roe": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 20},
    "fper_epsg": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60},
}  # 회귀모델별 이상치 필터. 안 쓰면 {}

SCORE_MAP = {
    "T_PER": "t_per_score", "T_PBR": "pbr_score", "T_EVEBITDA": "t_ev_ebitda_score",
    "ATT_PBR": "pbr_roe_attractiveness_score", "ATT_PER": "fper_epsg_attractiveness_score",
    "F_EPS_M": "f_eps_m_score",
}  # WEIGHTS_LARGE의 모든 키에 대응하는 스코어 컬럼

SCORING_RULES = {
    # rule1 = 낮을수록 좋음 (밸류: PER, PBR, EV/EBITDA 등)
    # rule2 = 높을수록 좋음 (성장, 모멘텀, 회귀매력도, 유동비율 등)
    # rule3 = 낮을수록 좋음 (가격모멘텀 역방향, 부채비율)
    "t_per": "rule1", "pbr": "rule1", "t_ev_ebitda": "rule1",
    "pbr_roe_attractiveness": "rule2", "fper_epsg_attractiveness": "rule2",
    "f_eps_m": "rule2",
}

PARAMS = {"top_n": 30, "tx_cost_bp": 30, "weight_cap_pct": 10}

QUALITY_FILTER = {
    "exclude_spac_etf_reit": True,
    "require_positive_oi": True,
    "require_positive_roe": True,
    "min_avg_volume": 500_000_000,
}
```

중요: WEIGHTS_LARGE 합계는 0.95~1.05 사이여야 하며, 모든 WEIGHTS 키는 SCORE_MAP에 있어야 합니다.
중요: SCORING_RULES 안에 반드시 rule 방향을 설명하는 주석을 포함하세요:
  # rule1 = 낮을수록 좋음 (밸류 멀티플: PER, PBR, EV/EBITDA, PCF 등)
  # rule2 = 높을수록 좋음 (성장률, 모멘텀, 회귀매력도, 유동비율 등)
  # rule3 = 낮을수록 좋음 (가격모멘텀 역방향, 부채비율 등)
"""


# ══════════════════════════════════════════════
# 1. GET /api/config
# ══════════════════════════════════════════════
@app.get("/api/config")
def get_config():
    return _convert_for_json({
        "backtest_config": BACKTEST_CONFIG,
        "strategy_labels": STRATEGY_LABELS,
        "strategy_colors": STRATEGY_COLORS,
        "all_keys": list(ALL_KEYS),
        "base_strategy_weights": BASE_STRATEGY_WEIGHTS,
        "default_strategy_code": DEFAULT_STRATEGY_CODE,
    })


# ══════════════════════════════════════════════
# 2. GET /api/latest-price-date
# ══════════════════════════════════════════════
@app.get("/api/latest-price-date")
def api_latest_price_date():
    date = _cached("latest_price_date", get_latest_price_date, ttl=600)
    return {"date": date}


@app.get("/api/debug-db")
def api_debug_db():
    from lib.db import get_conn, DATABASE_URL, _is_pg
    conn = get_conn()
    try:
        universe_count = conn.execute("SELECT COUNT(*) FROM universe WHERE rebal_type='monthly'").fetchone()[0]
        a0_return = None
        try:
            import json as _json
            row = conn.execute("SELECT results_json FROM backtest_cache WHERE name='A0' AND universe='KOSPI' AND rebal_type='monthly'").fetchone()
            if row:
                data = _json.loads(row[0]) if isinstance(row[0], str) else row[0]
                a0_return = data.get('total_return')
        except Exception:
            pass
        return {"is_pg": _is_pg, "db_url_prefix": DATABASE_URL[:30] if DATABASE_URL else None, "universe_monthly_rows": universe_count, "a0_total_return": a0_return}
    finally:
        conn.close()


# ══════════════════════════════════════════════
# 3. GET /api/results
# ══════════════════════════════════════════════
@app.get("/api/results")
def api_results(
    start: Optional[str] = None,
    end: Optional[str] = None,
    universe: Optional[str] = None,
    rebal_type: Optional[str] = None,
):
    cache_key = f"results:{start}:{end}:{universe}:{rebal_type}"
    results = _cached(
        cache_key,
        lambda: load_all_results(start, end, universe=universe, rebal_type=rebal_type),
    )
    return _convert_for_json(results)


# ══════════════════════════════════════════════
# 4. GET /api/holdings
# ══════════════════════════════════════════════
@app.get("/api/holdings")
def api_holdings(
    strategy: str,
    date: str,
    universe: Optional[str] = None,
    rebal_type: Optional[str] = None,
):
    df = get_holdings(strategy, date, universe=universe, rebal_type=rebal_type)
    return _convert_for_json(df.to_dict(orient="records") if not df.empty else [])


# ══════════════════════════════════════════════
# 5. GET /api/attribution
# ══════════════════════════════════════════════
@app.get("/api/attribution")
def api_attribution(
    strategy: str,
    start_date: str,
    end_date: str,
    universe: Optional[str] = None,
    rebal_type: Optional[str] = None,
):
    """holdings 캐시 + daily_price에서 실시간으로 종목별 수익률/기여도 계산."""
    # 1) holdings 캐시에서 해당 날짜 종목 가져오기
    holdings_df = get_holdings(strategy, start_date, universe=universe, rebal_type=rebal_type)
    if holdings_df.empty:
        return []

    codes = holdings_df["종목코드"].tolist()
    weights = {row["종목코드"]: row["비중(%)"] for _, row in holdings_df.iterrows()}
    names = {row["종목코드"]: row["종목명"] for _, row in holdings_df.iterrows()}
    sectors = {row["종목코드"]: row.get("섹터", "기타") for _, row in holdings_df.iterrows()}

    # 2) daily_price에서 시작/종료 가격 조회 (PG)
    from lib.db import get_conn
    conn = get_conn()
    placeholders = ",".join(["%s"] * len(codes))
    cur = conn.execute(f"""
        SELECT stock_code,
               MIN(CASE WHEN rn_start = 1 THEN adj_close END) as start_price,
               MIN(CASE WHEN rn_end = 1 THEN adj_close END) as end_price
        FROM (
            SELECT stock_code, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY trade_date ASC) as rn_start,
                   ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY trade_date DESC) as rn_end
            FROM daily_price
            WHERE stock_code IN ({placeholders})
              AND trade_date >= %s AND trade_date <= %s
              AND adj_close > 0
        ) sub
        WHERE rn_start = 1 OR rn_end = 1
        GROUP BY stock_code
    """, (*codes, start_date, end_date))
    price_rows = cur.fetchall()
    price_map = {r[0]: (r[1], r[2]) for r in price_rows}
    conn.close()

    # 3) 수익률/기여도 계산
    result = []
    for code in codes:
        sp, ep = price_map.get(code, (None, None))
        if not sp or sp <= 0:
            continue
        ret = (ep - sp) / sp if ep and ep > 0 else -1.0
        w = weights.get(code, 0) / 100
        sector = sectors.get(code, "기타")
        if sector:
            sector = sector.replace("코스피 ", "").replace("코스닥 ", "")
        result.append({
            "종목명": names.get(code, code),
            "섹터": sector,
            "비중(%)": round(w * 100, 1),
            "종목수익률(%)": round(ret * 100, 1),
            "기여도(%)": round(ret * w * 100, 2),
        })

    result.sort(key=lambda x: x["기여도(%)"], reverse=True)
    return _convert_for_json(result)


# ══════════════════════════════════════════════
# 6. GET /api/characteristics
# ══════════════════════════════════════════════
@app.get("/api/characteristics")
def api_characteristics(
    strategy: str,
    date: str,
    universe: Optional[str] = None,
    rebal_type: Optional[str] = None,
):
    result = get_portfolio_characteristics(strategy, date,
                                           universe=universe, rebal_type=rebal_type)
    return _convert_for_json(result)


# ══════════════════════════════════════════════
# 7. GET /api/turnover
# ══════════════════════════════════════════════
@app.get("/api/turnover")
def api_turnover(
    strategy: str,
    date: str,
    prev_date: str,
    universe: Optional[str] = None,
    rebal_type: Optional[str] = None,
):
    result = get_portfolio_turnover(strategy, date, prev_date,
                                    universe=universe, rebal_type=rebal_type)
    return _convert_for_json({
        "added": result["added"].to_dict(orient="records") if isinstance(result["added"], pd.DataFrame) and not result["added"].empty else [],
        "removed": result["removed"].to_dict(orient="records") if isinstance(result["removed"], pd.DataFrame) and not result["removed"].empty else [],
        "added_count": result["added_count"],
        "removed_count": result["removed_count"],
        "retained_count": result["retained_count"],
        "turnover_rate": result["turnover_rate"],
    })


# ══════════════════════════════════════════════
# 8. GET /api/robustness
# ══════════════════════════════════════════════
@app.get("/api/robustness")
def api_robustness(
    start: Optional[str] = None,
    end: Optional[str] = None,
    is_end: Optional[str] = None,
    oos_start: Optional[str] = None,
    universe: Optional[str] = None,
    rebal_type: Optional[str] = None,
):
    cache_key = f"robustness:{start}:{end}:{is_end}:{oos_start}:{universe}:{rebal_type}"

    def _load():
        is_oos, stat, rolling = load_all_robustness_results(
            start, end, is_end, oos_start, universe=universe,
            rebal_type=rebal_type,
        )
        # Remove boot_means (large array, not needed in API)
        for sig in stat.get("bm_significance", {}).values():
            sig.pop("boot_means", None)
        return {"is_oos": is_oos, "stat": stat, "rolling": rolling}

    result = _cached(cache_key, _load, ttl=600)
    return _convert_for_json(result)


# ══════════════════════════════════════════════
# 9. GET /api/strategies
# ══════════════════════════════════════════════
@app.get("/api/strategies")
def api_list_strategies(
    universe: Optional[str] = None,
    rebal_type: Optional[str] = None,
):
    return list_strategies(universe=universe, rebal_type=rebal_type)


# ══════════════════════════════════════════════
# 10. POST /api/strategies
# ══════════════════════════════════════════════
class SaveStrategyRequest(BaseModel):
    name: str
    code: str
    description: str = ""
    results: Optional[dict] = None
    universe: Optional[str] = None
    rebal_type: Optional[str] = None


def _invalidate_results_cache():
    """전략 저장/삭제 후 results 캐시를 즉시 무효화."""
    keys_to_remove = [k for k in _cache if k.startswith("results:")]
    for k in keys_to_remove:
        del _cache[k]


@app.post("/api/strategies")
def api_save_strategy(req: SaveStrategyRequest):
    try:
        save_strategy(
            name=req.name,
            code=req.code,
            description=req.description,
            results=req.results,
            universe=req.universe,
            rebal_type=req.rebal_type,
        )
        _invalidate_results_cache()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════
# 11a. GET /api/strategies/{name}  — 단일 전략 상세 (code 포함)
# ══════════════════════════════════════════════
@app.get("/api/strategies/{name}")
def api_get_strategy(name: str, universe: Optional[str] = None, rebal_type: Optional[str] = None):
    # Try with provided params first, then fallback to all combos
    combos = []
    if universe and rebal_type:
        combos = [(universe, rebal_type)]
    else:
        combos = [
            (universe or "KOSPI", rebal_type or "monthly"),
            (universe or "KOSPI", "biweekly"),
            ("KOSPI+KOSDAQ", rebal_type or "monthly"),
            ("KOSPI+KOSDAQ", "biweekly"),
        ]
    for u, r in combos:
        data = load_strategy(name, rebal_type=r, universe=u)
        if data and data.get("code"):
            return _convert_for_json(data)
    raise HTTPException(status_code=404, detail="전략을 찾을 수 없습니다.")


# 11b. DELETE /api/strategies/{name}
# ══════════════════════════════════════════════
@app.delete("/api/strategies/{name}")
def api_delete_strategy(name: str):
    try:
        delete_strategy(name)
        _invalidate_results_cache()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════
# 12. POST /api/backtest  (async: start → poll)
# ══════════════════════════════════════════════
import threading, uuid as _uuid

class BacktestRequest(BaseModel):
    strategy_code: str
    universe: Optional[str] = None
    rebal_type: Optional[str] = None
    weight_cap_pct: Optional[int] = None
    tx_cost_bp: Optional[int] = None
    stop_loss_enabled: Optional[bool] = None
    stop_loss_pct: Optional[int] = None
    stop_loss_mode: Optional[str] = None

# In-memory job store
_backtest_jobs: dict[str, dict] = {}


def _run_backtest_job(job_id: str, req: BacktestRequest):
    import traceback as _tb
    try:
        result = run_strategy_backtest(
            strategy_code=req.strategy_code,
            universe=req.universe,
            rebal_type=req.rebal_type,
            weight_cap_pct_override=req.weight_cap_pct,
            tx_cost_bp_override=req.tx_cost_bp,
            stop_loss_enabled=req.stop_loss_enabled,
            stop_loss_pct=req.stop_loss_pct,
            stop_loss_mode=req.stop_loss_mode,
        )
        if result is None:
            _backtest_jobs[job_id] = {"status": "error", "detail": "Backtest returned no results"}
        elif isinstance(result, dict) and "error" in result:
            _backtest_jobs[job_id] = {"status": "error", "detail": result["error"]}
        else:
            _backtest_jobs[job_id] = {"status": "done", "result": _convert_for_json(result)}
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        print(f"[BACKTEST ERROR] {detail}\n{_tb.format_exc()}", flush=True)
        _backtest_jobs[job_id] = {"status": "error", "detail": detail}


@app.post("/api/backtest")
def api_run_backtest(req: BacktestRequest):
    # Validate code first (fast)
    from lib.factor_engine import validate_strategy_code
    is_valid, err = validate_strategy_code(req.strategy_code)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err)

    job_id = str(_uuid.uuid4())[:8]
    _backtest_jobs[job_id] = {"status": "running"}
    t = threading.Thread(target=_run_backtest_job, args=(job_id, req), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/backtest/{job_id}")
def api_backtest_status(job_id: str):
    job = _backtest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "done":
        result = _backtest_jobs.pop(job_id)
        return result
    if job["status"] == "error":
        detail = _backtest_jobs.pop(job_id)["detail"]
        raise HTTPException(status_code=400, detail=detail)
    return {"status": "running"}


# ══════════════════════════════════════════════
# 13. POST /api/chat  — General Q&A with SSE streaming
# ══════════════════════════════════════════════
class ChatRequest(BaseModel):
    messages: list[dict]  # [{role, content}, ...]
    context: Optional[dict] = None  # dashboard context


def _build_chat_context() -> str:
    """Build auto-context from current cached data for AI chat."""
    parts = []

    # 1) 전략 성과 요약 (양쪽 유니버스)
    for univ in ["KOSPI", "KOSPI+KOSDAQ"]:
        try:
            results = load_all_results(universe=univ, rebal_type="monthly")
            sp = [f"\n\n## 전략 성과 요약 ({univ}, 월간)"]
            for key, r in results.items():
                if not isinstance(r, dict) or "cagr" not in r:
                    continue
                label = STRATEGY_LABELS.get(key, key)
                cagr = r.get("cagr", 0)
                mdd = r.get("mdd", 0)
                sharpe = r.get("sharpe", 0)
                total = r.get("total_return", 0)
                monthly = r.get("monthly_returns", [])
                # 최근 6개월 월별 수익률
                recent = monthly[-6:] if monthly else []
                recent_s = ", ".join(f"{m.get('return',0)*100:+.1f}%" for m in recent) if recent else ""
                sp.append(
                    f"- {label}: 총수익률 {total*100:.1f}%, CAGR {cagr*100:.1f}%, "
                    f"MDD {mdd*100:.1f}%, Sharpe {sharpe:.2f}"
                    + (f" | 최근6M: [{recent_s}]" if recent_s else "")
                )
            parts.append("\n".join(sp))
        except Exception:
            pass

    # 2) 최신 보유종목 상세 (KOSPI 월간)
    try:
        from lib.data import _load_holdings_cache
        hcache = _load_holdings_cache(universe="KOSPI", rebal_type="monthly")
        if hcache:
            parts.append("\n\n## 최신 보유종목 (KOSPI, 월간)")
            for sname, dates_data in hcache.items():
                if not isinstance(dates_data, dict):
                    continue
                latest_date = max(dates_data.keys())
                holdings = dates_data[latest_date]
                if not holdings:
                    continue
                label = STRATEGY_LABELS.get(sname, sname)
                parts.append(f"\n### {label} ({latest_date}, {len(holdings)}종목)")
                for h in holdings[:15]:
                    nm = h.get("stock_name", "?")
                    w = h.get("weight", 0)
                    sec = h.get("sector", "")
                    per = h.get("per", "")
                    pbr = h.get("pbr", "")
                    ev = h.get("ev_ebitda", "")
                    parts.append(
                        f"  {nm}: {w:.1f}% | {sec} | PER {per} PBR {pbr} EV/EBITDA {ev}"
                    )
                if len(holdings) > 15:
                    parts.append(f"  ... 외 {len(holdings)-15}종목")
    except Exception:
        pass

    # 3) 포트폴리오 특성 (최신 날짜)
    try:
        from lib.data import _load_holdings_cache
        hcache = _load_holdings_cache(universe="KOSPI", rebal_type="monthly")
        if hcache:
            parts.append("\n\n## 포트폴리오 특성 (가중평균)")
            for sname, dates_data in hcache.items():
                if not isinstance(dates_data, dict):
                    continue
                latest_date = max(dates_data.keys())
                try:
                    chars = get_portfolio_characteristics(sname, latest_date, universe="KOSPI", rebal_type="monthly")
                    if chars:
                        label = STRATEGY_LABELS.get(sname, sname)
                        per_w = chars.get("PER", "-")
                        pbr_w = chars.get("PBR", "-")
                        ev_w = chars.get("EV/EBITDA", "-")
                        parts.append(f"- {label}: PER {per_w}, PBR {pbr_w}, EV/EBITDA {ev_w}")
                except Exception:
                    pass
    except Exception:
        pass

    # 4) 통계검증 요약
    try:
        rob = load_all_robustness_results(universe="KOSPI", rebal_type="monthly")
        if rob:
            stat = rob.get("stat", {})
            bm_sig = stat.get("bm_significance", {})
            if bm_sig:
                parts.append("\n\n## 통계검증 — 벤치마크 대비 유의성")
                for sname, s in bm_sig.items():
                    label = STRATEGY_LABELS.get(sname, sname)
                    pval = s.get("p_value", "-")
                    tstat = s.get("t_stat", "-")
                    sig = "유의" if s.get("significant") else "비유의"
                    excess = s.get("monthly_excess", 0)
                    win = s.get("bootstrap_win_rate", 0)
                    parts.append(
                        f"- {label}: 월평균초과 {excess*100:.2f}%, t={tstat:.2f}, "
                        f"p={pval:.4f}, 부트스트랩승률 {win*100:.1f}%, {sig}"
                    )

            is_oos = rob.get("is_oos", {})
            is_res = is_oos.get("is_results", {})
            oos_res = is_oos.get("oos_results", {})
            if is_res or oos_res:
                parts.append("\n\n## IS/OOS 성과 비교")
                for sname in set(list(is_res.keys()) + list(oos_res.keys())):
                    label = STRATEGY_LABELS.get(sname, sname)
                    ir = is_res.get(sname, {})
                    osr = oos_res.get(sname, {})
                    parts.append(
                        f"- {label}: IS(CAGR {ir.get('cagr',0)*100:.1f}%, Sharpe {ir.get('sharpe',0):.2f}, "
                        f"MDD {ir.get('mdd',0)*100:.1f}%) → OOS(CAGR {osr.get('cagr',0)*100:.1f}%, "
                        f"Sharpe {osr.get('sharpe',0):.2f}, MDD {osr.get('mdd',0)*100:.1f}%)"
                    )
    except Exception:
        pass

    # 5) 레짐 분석 요약
    try:
        regime = compute_regime_analysis(universe="KOSPI", rebal_type="monthly")
        if regime:
            summary = regime.get("summary", {})
            counts = regime.get("regime_counts", {})
            if summary:
                parts.append(f"\n\n## 시장 레짐 분석 (Bull/Sideways/Bear)")
                if counts:
                    parts.append(f"레짐 분포: Bull {counts.get('Bull',0)}개월, Sideways {counts.get('Sideways',0)}개월, Bear {counts.get('Bear',0)}개월")
                for sname, regimes in summary.items():
                    label = STRATEGY_LABELS.get(sname, sname)
                    regime_parts = []
                    for rname in ["Bull", "Sideways", "Bear"]:
                        rd = regimes.get(rname, {})
                        if rd:
                            avg = rd.get("avg_monthly_return", 0)
                            regime_parts.append(f"{rname} {avg*100:.2f}%/월")
                    if regime_parts:
                        parts.append(f"- {label}: {', '.join(regime_parts)}")
    except Exception:
        pass

    return "".join(parts)


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if not is_ai_available():
        raise HTTPException(status_code=503, detail="AI not available (no API key)")

    client = _get_client()
    context_str = _build_chat_context()
    if req.context:
        context_str += f"\n\n추가 컨텍스트:\n{json.dumps(req.context, ensure_ascii=False, default=str)}"

    api_messages = [{"role": m["role"], "content": m["content"]} for m in req.messages]

    def _pick_thinking_label(user_msg: str) -> str:
        """Pick a contextual thinking label based on the user's message."""
        msg = user_msg.lower()
        if any(k in msg for k in ["분석", "성과", "수익", "mdd", "sharpe", "cagr"]):
            return "데이터 분석 중"
        if any(k in msg for k in ["전략", "팩터", "모멘텀", "밸류"]):
            return "전략 설계 중"
        if any(k in msg for k in ["비교", "차이", "vs"]):
            return "비교 분석 중"
        if any(k in msg for k in ["종목", "포트폴리오", "섹터", "편입", "편출"]):
            return "포트폴리오 검토 중"
        if any(k in msg for k in ["sql", "쿼리", "데이터"]):
            return "데이터 조회 중"
        return "생각하는 중"

    def event_generator():
        try:
            # Send thinking status before streaming
            user_msg = api_messages[-1]["content"] if api_messages else ""
            label = _pick_thinking_label(user_msg)
            yield f"data: {json.dumps({'thinking': label}, ensure_ascii=False)}\n\n"

            with client.messages.stream(
                model=MODEL_FAST,
                max_tokens=2000,
                system=_SYSTEM_GENERAL + context_str,
                messages=api_messages,
            ) as stream:
                first = True
                for text in stream.text_stream:
                    if first:
                        yield f"data: {json.dumps({'thinking_done': True})}\n\n"
                        first = False
                    # SSE format
                    yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════
# 14. POST /api/chat/strategy — Strategy modification (tool use)
# ══════════════════════════════════════════════
class StrategyChatRequest(BaseModel):
    messages: list[dict]  # [{role, content}, ...]
    current_code: Optional[str] = None


@app.post("/api/chat/strategy")
def api_chat_strategy(req: StrategyChatRequest):
    if not is_ai_available():
        raise HTTPException(status_code=503, detail="AI not available (no API key)")

    current_code = req.current_code or DEFAULT_STRATEGY_CODE

    try:
        response_text, updated_code, changes_summary = chat_strategy_modification(
            req.messages,
            current_code,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "response": response_text or "",
        "updated_code": updated_code,
        "changes_summary": changes_summary,
    }


# ══════════════════════════════════════════════
# 15. POST /api/chat/sql — Execute read-only SQL against PG
# ══════════════════════════════════════════════
class SqlQueryRequest(BaseModel):
    query: str


_SQL_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY|EXECUTE|DO)\b",
    re.IGNORECASE,
)


@app.post("/api/chat/sql")
def api_chat_sql(req: SqlQueryRequest):
    query = req.query.strip().rstrip(";")

    # Only allow SELECT / WITH (CTE) queries
    if not re.match(r"^\s*(SELECT|WITH)\b", query, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")

    if _SQL_FORBIDDEN.search(query):
        raise HTTPException(status_code=400, detail="Query contains forbidden statements")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")

    import psycopg2
    import psycopg2.extras

    try:
        conn = psycopg2.connect(database_url)
        conn.set_session(readonly=True, autocommit=True)
        cur = conn.cursor()
        cur.execute(query)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Convert rows to list of lists for JSON
        data = [
            [_convert_for_json(cell) for cell in row]
            for row in rows
        ]

        return {
            "columns": columns,
            "data": data,
            "row_count": len(data),
        }
    except psycopg2.Error as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {e.pgerror or str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════
# 16. GET /api/regime
# ══════════════════════════════════════════════
@app.get("/api/regime")
def api_regime(
    start: Optional[str] = None,
    end: Optional[str] = None,
    universe: Optional[str] = None,
    rebal_type: Optional[str] = None,
    ma_window: Optional[int] = 50,
):
    cache_key = f"regime:{start}:{end}:{universe}:{rebal_type}:{ma_window}"
    result = _cached(
        cache_key,
        lambda: compute_regime_analysis(start, end, universe=universe, rebal_type=rebal_type, ma_window=ma_window),
        ttl=600,
    )
    return _convert_for_json(result)


# ══════════════════════════════════════════════
# 17. GET /api/first-entry-dates
# ══════════════════════════════════════════════
@app.get("/api/first-entry-dates")
def api_first_entry_dates(
    strategy: str,
    universe: Optional[str] = None,
    rebal_type: Optional[str] = None,
):
    cache_key = f"first_entry:{strategy}:{universe}:{rebal_type}"
    result = _cached(
        cache_key,
        lambda: get_first_entry_dates(strategy, universe=universe, rebal_type=rebal_type),
        ttl=3600,
    )
    return result


# ══════════════════════════════════════════════
# 18. POST /api/regime-combo  (async job)
# ══════════════════════════════════════════════
def _run_regime_combo_job(job_id: str, bull_key: str, bear_key: str, universe: str, rebal_type: str, ma_window: int = 50, regime_mode: str = "ma"):
    import traceback as _tb
    try:
        result = run_regime_combo_backtest(bull_key, bear_key, universe=universe, rebal_type=rebal_type, ma_window=ma_window, regime_mode=regime_mode)
        if result is None:
            _backtest_jobs[job_id] = {"status": "error", "detail": "결과 없음"}
        elif "error" in result:
            _backtest_jobs[job_id] = {"status": "error", "detail": result["error"]}
        else:
            _backtest_jobs[job_id] = {"status": "done", "result": result}
    except Exception as e:
        _backtest_jobs[job_id] = {"status": "error", "detail": f"{type(e).__name__}: {e}"}
        print(f"[REGIME COMBO ERROR] {_tb.format_exc()}", flush=True)


@app.post("/api/regime-combo")
def api_regime_combo_start(
    bull_key: str,
    bear_key: str,
    universe: Optional[str] = "KOSPI",
    rebal_type: Optional[str] = "monthly",
    ma_window: Optional[int] = 50,
    regime_mode: Optional[str] = "ma",
):
    job_id = str(_uuid.uuid4())[:8]
    _backtest_jobs[job_id] = {"status": "running"}
    t = threading.Thread(
        target=_run_regime_combo_job,
        args=(job_id, bull_key, bear_key, universe, rebal_type, ma_window, regime_mode),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "running"}



@app.get("/api/regime-combo/{job_id}")
def api_regime_combo_poll(job_id: str):
    job = _backtest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


# ══════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════
@app.get("/api/health")
def health():
    return {"status": "ok", "ai_available": is_ai_available()}


@app.on_event("startup")
def _warmup_cache():
    """Pre-load results for all universe/rebal combos on startup."""
    import threading

    def _warm():
        combos = [
            ("KOSPI", "monthly"),
            ("KOSPI", "biweekly"),
            ("KOSPI+KOSDAQ", "monthly"),
            ("KOSPI+KOSDAQ", "biweekly"),
        ]
        for uni, rt in combos:
            key = f"results:None:None:{uni}:{rt}"
            try:
                _cached(key, lambda u=uni, r=rt: load_all_results(None, None, universe=u, rebal_type=r))
                print(f"  [warmup] {uni}/{rt} cached")
            except Exception as e:
                print(f"  [warmup] {uni}/{rt} failed: {e}")

    threading.Thread(target=_warm, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
