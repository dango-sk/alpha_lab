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
    list_strategies,
    save_strategy,
    delete_strategy,
    run_strategy_backtest,
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

## 규칙
- 한국어로 응답합니다.
- 전문 금융 용어를 사용하되, 코드 변수명이나 프로그래밍 용어는 절대 사용하지 마세요.
- 답변은 간결하고 핵심적으로. 수치를 근거로 들어 답변하세요.
- 사용자가 특정 종목이나 데이터를 물어보면, 당신이 가지고 있는 컨텍스트 데이터를 기반으로 답변하세요.
- 사용자가 새로운 전략을 만들어달라고 하면, factor_engine 형식의 파이썬 코드를 ```python 블록으로 생성하세요.
- 생성한 전략 코드는 사용자가 "전략 실험실에서 열기" 버튼으로 바로 백테스트할 수 있습니다.

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
    "t_per": "rule1", "pbr": "rule1", "t_ev_ebitda": "rule1",
    "pbr_roe_attractiveness": "rule2", "fper_epsg_attractiveness": "rule2",
    "f_eps_m": "rule2",
}  # 각 팩터의 점수 방향

PARAMS = {"top_n": 30, "tx_cost_bp": 30, "weight_cap_pct": 10}

QUALITY_FILTER = {
    "exclude_spac_etf_reit": True,
    "require_positive_oi": True,
    "require_positive_roe": True,
    "min_avg_volume": 500_000_000,
}
```

중요: WEIGHTS_LARGE 합계는 0.95~1.05 사이여야 하며, 모든 WEIGHTS 키는 SCORE_MAP에 있어야 합니다.
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
    df = get_monthly_attribution(strategy, start_date, end_date,
                                 universe=universe, rebal_type=rebal_type)
    return _convert_for_json(df.to_dict(orient="records") if not df.empty else [])


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
):
    cache_key = f"robustness:{start}:{end}:{is_end}:{oos_start}:{universe}"

    def _load():
        is_oos, stat, rolling = load_all_robustness_results(
            start, end, is_end, oos_start, universe=universe,
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
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════
# 11. DELETE /api/strategies/{name}
# ══════════════════════════════════════════════
@app.delete("/api/strategies/{name}")
def api_delete_strategy(name: str):
    try:
        delete_strategy(name)
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
    try:
        results = load_all_results(universe="KOSPI", rebal_type="monthly")
        summary_parts = ["\n\n## 현재 전략 성과 요약 (KOSPI, 월간)"]
        for key, r in results.items():
            if isinstance(r, dict) and "cagr" in r:
                cagr = r.get("cagr", 0)
                mdd = r.get("mdd", 0)
                sharpe = r.get("sharpe", 0)
                total = r.get("total_return", 0)
                label = STRATEGY_LABELS.get(key, key)
                cagr_s = f"{cagr*100:.1f}%" if cagr else "-"
                mdd_s = f"{mdd*100:.1f}%" if mdd else "-"
                summary_parts.append(
                    f"- {label}: 총수익률 {total*100:.1f}%, CAGR {cagr_s}, MDD {mdd_s}, Sharpe {sharpe:.2f}"
                )
        return "\n".join(summary_parts)
    except Exception:
        return ""


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
