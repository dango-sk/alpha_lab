"""
AI 모듈: Claude API 기반 코멘터리 생성 + 전략 코드 생성 tool use
"""
import json
from typing import Optional

import streamlit as st

from config.settings import ANTHROPIC_API_KEY
from lib.factor_engine import validate_strategy_code, DEFAULT_STRATEGY_CODE

MODEL_FAST = "claude-sonnet-4-5-20250929"   # 코멘터리용 (빠름)
MODEL_SMART = "claude-sonnet-4-5-20250929"  # 전략 수정 tool use용

SYSTEM_PROMPT = """당신은 자산운용사의 시니어 퀀트 애널리스트입니다.
한국 주식시장의 팩터 투자 전략 백테스트 결과를 분석하고 인사이트를 제공합니다.

규칙:
- 자산운용사 리서치 애널리스트가 읽는다고 가정합니다.
- 전문 금융 용어를 사용하되, 코드 변수명이나 프로그래밍 용어는 절대 사용하지 마세요.
- 핵심 인사이트를 3-5문장으로 간결하게 전달하세요.
- 수치는 반드시 포함하되, 해석을 덧붙이세요.
- 한국어로 응답하세요.
"""

# ═══════════════════════════════════════════════════════
# 전략 수정 도구 (단일 update_strategy)
# ═══════════════════════════════════════════════════════

STRATEGY_TOOLS = [
    {
        "name": "update_strategy",
        "description": "전략 설정 코드(strategy.py)를 생성/수정합니다. 반드시 전체 strategy.py 코드를 출력하세요.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "strategy.py 전체 내용 (Python). 필수 변수: WEIGHTS_LARGE, WEIGHTS_SMALL, REGRESSION_MODELS, OUTLIER_FILTERS, SCORE_MAP, SCORING_RULES, PARAMS, QUALITY_FILTER",
                },
                "explanation": {
                    "type": "string",
                    "description": "사용자에게 보여줄 변경 설명 (한국어)",
                },
                "changes_summary": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "개별 변경 목록 (한국어)",
                },
            },
            "required": ["code", "explanation", "changes_summary"],
        },
    },
]

# 전략 수정 AI 시스템 프롬프트
_STRATEGY_SYSTEM_TEMPLATE = """당신은 한국 주식시장 팩터 투자 전략을 설계하는 퀀트 개발자입니다.
사용자의 요청에 따라 전략 설정 코드(strategy.py)를 수정합니다.

## 현재 전략 코드
```python
{current_code}
```

## 사용 가능한 원시 데이터 컬럼
DB에서 로딩되는 재무/주가/Forward 데이터:
- 밸류에이션: pbr, per, psr, ev_ebitda, pcf, div_yield
- 수익성: roe, roic, oi_margin
- 기업가치: ev, ic, ebit, ebitda, net_debt, interest_debt, total_equity, market_cap
- 포워드: fwd_eps, fwd_per, fwd_ebit, fwd_ebitda, fwd_ev_ebitda, fwd_revenue, fwd_oi, fwd_ni, fwd_roe, fwd_bps
- 주가: close, close_3m, trade_amount
- 재무: revenue, operating_income, net_income, eps, bps
- 전기: prev_revenue

## 자동 계산되는 파생 지표
엔진이 자동으로 계산하는 컬럼 (직접 사용 가능):
- 밸류: t_per (trailing PER), f_per (forward PER), t_ev_ebitda, f_ev_ebitda, f_ev_ebit, ev_ic, f_pbr, t_pcf
- 성장: f_epsg (forward EPS 성장률), f_ebitg (forward EBIT 성장률), t_spsg (trailing 매출성장률), f_spsg (forward 매출성장률)
- 모멘텀: f_eps_m (forward EPS 3개월 변화율), price_m (3개월 주가 모멘텀)
- 재무건전성: ndebt_ebitda (순차입/EBITDA), current_ratio (유동비율)

## 회귀 모델 formula_type
- "ratio": fitted / actual - 1 (예: PBR-ROE, Forward PER-이익성장)
- "ev_equity": (fitted * IC - net_debt) / market_cap - 1 (예: EV/IC-ROIC)
- "ev_equity_ebit": (fitted * |fwd_ebit| - net_debt) / market_cap - 1 (예: Forward EV/EBIT-EBIT성장)
- "simple": (fitted - actual) / |actual| (범용 잔차 기반)

회귀 모델 추가 시: REGRESSION_MODELS에 (name, x_col, y_col, formula_type) 튜플을 추가.
결과 컬럼은 자동으로 `{{name}}_attractiveness`가 됩니다.

## 스코어링 규칙
- "rule1": 낮을수록 좋음 (밸류 멀티플: PER, PBR, EV/EBITDA 등)
- "rule2": 높을수록 좋음 (회귀 매력도, 성장률, EPS 모멘텀 등)
- "rule3": 낮을수록 좋음 (주가 모멘텀 역방향, 부채비율)

## 채점 방식 (SCORING_MODE)
SCORING_MODE는 대형주와 중소형주의 채점 방식을 지정합니다:
- "quartile": 사분위 채점 (0~4점). 종목 수가 적을 때 적합. 기본 A0 전략의 대형주에 사용.
- "decile": 십분위 채점 (0~10점). 종목 수가 충분할 때 세밀한 분류 가능. 기본 A0 전략의 중소형주에 사용.

점수 정규화: 사분위는 max=4, 십분위는 max=10으로 나눠서 0~100 점수로 변환.
SCORING_MODE가 없으면 기본값은 {{"large": "quartile", "small": "decile"}}.

## 중요 규칙
1. 반드시 전체 strategy.py 코드를 출력하세요 (수정된 부분만이 아닌 전체 파일)
2. WEIGHTS_LARGE와 WEIGHTS_SMALL의 가중치 합은 각각 1.0이어야 합니다
3. SCORE_MAP의 키는 WEIGHTS의 키와 매칭되어야 합니다
4. import, def, class, exec, eval 등 코드 실행 구문은 사용 금지
5. 순수 데이터 선언(dict, list, tuple, str, int, float)만 사용
6. 회귀 모델을 추가하면 SCORING_RULES에 `{{name}}_attractiveness: "rule2"` 추가, SCORE_MAP에 매핑 추가 필요
7. 사용자 요청에 맞게 가중치를 재분배하되, 합계 1.0 유지
8. SCORING_MODE를 포함하세요 (기본: large=quartile, small=decile)
9. 한국어로 설명하세요

## 변경 예시

### 예시 1: 밸류 비중 50%로 높이기
밸류에이션 카테고리(T_PER, F_PER, T_EVEBITDA, F_EVEBITDA, T_PBR, F_PBR, T_PCF)의 합이 50%가 되도록 조정.
나머지 카테고리 비중을 비례적으로 줄여서 합계 100% 유지.

### 예시 2: 커스텀 회귀 추가 (PBR vs 매출성장)
1. REGRESSION_MODELS에 ("custom_pbr_spsg", "f_spsg", "pbr", "ratio") 추가
2. OUTLIER_FILTERS에 "custom_pbr_spsg": {{"x_min": -50, "x_max": 200, "y_min": 0, "y_max": 20}} 추가
3. SCORING_RULES에 "custom_pbr_spsg_attractiveness": "rule2" 추가
4. SCORE_MAP에 "ATT_CUSTOM1": "custom_pbr_spsg_attractiveness_score" 추가
5. WEIGHTS_LARGE, WEIGHTS_SMALL에 "ATT_CUSTOM1" 가중치 배분

### 예시 3: 팩터 제거
해당 팩터의 가중치를 0으로 설정하고, 남은 팩터들의 가중치를 비례적으로 올려서 합계 1.0 유지.
"""

# 팩터 코드 → 금융 용어 매핑 (UI 표시용 유지)
FACTOR_LABELS = {
    "T_PER": "Trailing PER",
    "F_PER": "Forward PER",
    "T_EVEBITDA": "Trailing EV/EBITDA",
    "F_EVEBITDA": "Forward EV/EBITDA",
    "T_PBR": "PBR",
    "F_PBR": "Forward PBR",
    "T_PCF": "PCF (잉여현금흐름 배수)",
    "ATT_PBR": "PBR-ROE 괴리도",
    "ATT_EVIC": "EV/IC-ROIC 괴리도",
    "ATT_PER": "Forward PER-이익성장 괴리도",
    "ATT_EVEBIT": "Forward EV/EBIT-EBIT성장 괴리도",
    "T_SPSG": "Trailing 매출성장률",
    "F_SPSG": "Forward 매출성장률",
    "F_EPS_M": "Forward EPS 모멘텀",
    "PRICE_M": "3개월 주가모멘텀",
    "NDEBT_EBITDA": "순차입/EBITDA",
    "CURRENT": "유동비율",
}

FACTOR_CATEGORIES = {
    "밸류에이션": ["T_PER", "F_PER", "T_EVEBITDA", "F_EVEBITDA", "T_PBR", "F_PBR", "T_PCF"],
    "회귀 매력도": ["ATT_PBR", "ATT_EVIC", "ATT_PER", "ATT_EVEBIT"],
    "성장성": ["T_SPSG", "F_SPSG"],
    "차별화": ["F_EPS_M", "PRICE_M", "NDEBT_EBITDA", "CURRENT"],
}

# 채팅 예시 (UI에서 사용)
CHAT_EXAMPLES = [
    "밸류 비중을 50%로 올려줘",
    "PBR vs 매출성장 커스텀 회귀를 추가해줘",
    "Forward EPS 모멘텀 팩터를 제거해줘",
    "종목수 20개, 거래비용 50bp로 변경해줘",
    "밸류에이션과 성장성만 남기고 나머지 카테고리는 제거해줘",
    "ROE 양수 필터를 제거해줘",
]


def _get_client():
    """Anthropic 클라이언트 반환. API 키 없으면 None."""
    if not ANTHROPIC_API_KEY:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def is_ai_available() -> bool:
    """AI 기능 사용 가능 여부"""
    return bool(ANTHROPIC_API_KEY)


_DISPLAY_KEYS = {"A0", "ATT2", "KOSPI"}
_KEY_LABELS = {
    "A0": "기존전략",
    "ATT2": "회귀only",
    "KOSPI": "KOSPI 200",
}


def generate_commentary(results: dict, context: str = "성과 비교") -> Optional[str]:
    """백테스트 결과에 대한 AI 코멘터리 생성"""
    client = _get_client()
    if not client:
        return None

    # 대시보드에 표시되는 전략만 필터
    summary_lines = []
    for key in ["A0", "ATT2", "KOSPI"]:
        r = results.get(key)
        if not r or not isinstance(r, dict) or "cagr" not in r:
            continue
        label = _KEY_LABELS.get(key, key)
        line = (
            f"- {label}: 총수익률 {r['total_return']:+.1%}, "
            f"CAGR {r['cagr']:+.1%}, MDD {r['mdd']:.1%}, Sharpe {r['sharpe']:.2f}"
        )
        if "avg_turnover" in r:
            line += f", 평균회전율 {r['avg_turnover']:.1%}"
        if "avg_portfolio_size" in r:
            line += f", 평균종목수 {r['avg_portfolio_size']:.0f}"
        summary_lines.append(line)

    user_msg = f"""다음은 한국 주식시장 팩터 투자 전략 백테스트 결과입니다.
대시보드에 표시되는 전략은 기존전략(사분위 밸류 기반), 회귀only(회귀 매력도 기반), KOSPI 200(벤치마크) 세 가지뿐입니다.
이 세 전략만 언급해주세요.

분석 맥락: {context}

{chr(10).join(summary_lines)}

다음 구조로 종합 분석해주세요:
1. 전략별 절대 성과 요약 (수익률, 위험)
2. 벤치마크 대비 초과수익 평가
3. 두 팩터 전략(기존전략 vs 회귀only) 간 비교 — 어떤 전략이 더 우수한지, 왜 그런지
4. 리스크 대비 수익(Sharpe) 관점에서의 평가
5. 종합 결론 및 시사점"""

    try:
        response = client.messages.create(
            model=MODEL_FAST,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text
    except Exception as e:
        return f"AI 분석 생성 실패: {e}"


def generate_stat_commentary(stat_data: dict) -> Optional[str]:
    """통계 검증 결과에 대한 AI 코멘터리"""
    client = _get_client()
    if not client:
        return None

    lines = []
    for key in ["A0", "ATT2"]:
        sig = stat_data.get("bm_significance", {}).get(key)
        if not sig:
            continue
        label = _KEY_LABELS.get(key, key)
        lines.append(
            f"- {label}: 월평균 초과수익 {sig['mean_diff']*100:+.3f}%, "
            f"t-stat {sig['t_stat']:.2f}, p-value {sig['p_value']:.4f}, "
            f"95% CI [{sig['ci_lower']*100:+.3f}%, {sig['ci_upper']*100:+.3f}%], "
            f"Bootstrap 승률 {sig['win_rate']:.1%}, "
            f"{'통계적으로 유의' if sig['significant'] else '유의하지 않음'}"
        )

    user_msg = f"""다음은 기존전략(사분위 밸류)과 회귀only(회귀 매력도) 전략의 KOSPI 200 대비 초과수익률 통계 검증 결과입니다.
이 두 전략만 분석해주세요.

{chr(10).join(lines)}

다음을 포함하여 분석해주세요:
1. 통계적 유의성 판단 (p-value, 신뢰구간 해석)
2. 실무적 유의성 (월 초과수익 크기가 운용에 의미 있는 수준인지)
3. 두 전략 간 비교
4. 투자 의사결정에 대한 시사점"""

    try:
        response = client.messages.create(
            model=MODEL_FAST,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text
    except Exception as e:
        return f"AI 분석 생성 실패: {e}"


# ═══════════════════════════════════════════════════════
# 전략 수정 채팅 (코드 생성 방식)
# ═══════════════════════════════════════════════════════

def _execute_update_strategy(tool_input: dict) -> tuple[str, str | None]:
    """
    update_strategy 도구 실행.
    코드 검증 → 성공 시 코드 반환, 실패 시 에러 메시지 반환.

    Returns: (tool_result_message, validated_code_or_None)
    """
    code = tool_input["code"]

    is_valid, err = validate_strategy_code(code)
    if not is_valid:
        return f"코드 검증 실패: {err}\n수정 후 다시 시도하세요.", None

    return "코드 검증 성공. 전략이 업데이트되었습니다.", code


def chat_strategy_modification(
    display_messages: list,
    current_strategy_code: str,
) -> tuple[str, str | None, list]:
    """
    전략 수정 채팅. tool use 멀티턴 처리 (코드 생성 방식).

    Args:
        display_messages: [{role, content}] 형태의 대화 이력
        current_strategy_code: 현재 전략 코드 문자열

    Returns:
        (ai_response_text, updated_code_or_None, changes_summary_list)
    """
    client = _get_client()
    if not client:
        return "ANTHROPIC_API_KEY가 설정되지 않았습니다.", None, []

    system = _STRATEGY_SYSTEM_TEMPLATE.format(current_code=current_strategy_code)

    # display_messages → API messages
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in display_messages
    ]

    updated_code = None
    changes_summary = []
    final_text_parts = []

    for _ in range(3):  # max turns (검증 실패 시 재시도)
        try:
            response = client.messages.create(
                model=MODEL_SMART,
                max_tokens=4096,
                system=system,
                messages=api_messages,
                tools=STRATEGY_TOOLS,
            )
        except Exception as e:
            return f"AI 응답 생성 실패: {e}", None, []

        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        final_text_parts = text_parts

        # No tool use → done
        if response.stop_reason != "tool_use":
            break

        # Handle tool use — convert pydantic objects to plain dicts
        # to avoid serialization issues with by_alias
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        api_messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for tc in tool_calls:
            if tc["name"] == "update_strategy":
                result_msg, code = _execute_update_strategy(tc["input"])
                if code is not None:
                    updated_code = code
                    changes_summary = tc["input"].get("changes_summary", [])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result_msg,
                })
            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": "알 수 없는 도구입니다.",
                    "is_error": True,
                })

        api_messages.append({"role": "user", "content": tool_results})

    return "\n".join(final_text_parts), updated_code, changes_summary


def format_weights_for_display(strategy_code: str) -> list:
    """전략 코드에서 가중치를 추출하여 대시보드 표시용 데이터로 변환.
    대형주와 중소형주 가중치를 모두 표시한다.
    """
    try:
        namespace = {}
        exec(compile(strategy_code, "<strategy>", "exec"), {"__builtins__": {}}, namespace)
        weights_large = namespace.get("WEIGHTS_LARGE", {})
        weights_small = namespace.get("WEIGHTS_SMALL", {})
    except Exception:
        return []

    # 모든 팩터 키 수집 (카테고리 순서 유지)
    all_known = set()
    for factors in FACTOR_CATEGORIES.values():
        all_known.update(factors)

    rows = []
    for cat, factors in FACTOR_CATEGORIES.items():
        for f in factors:
            wl = weights_large.get(f, 0)
            ws = weights_small.get(f, 0)
            if wl > 0 or ws > 0:
                rows.append({
                    "카테고리": cat,
                    "팩터": FACTOR_LABELS.get(f, f),
                    "대형주": f"{wl*100:.0f}%" if wl > 0 else "-",
                    "중소형주": f"{ws*100:.0f}%" if ws > 0 else "-",
                    "_wl": wl,
                    "_ws": ws,
                    "_code": f,
                })

    # 커스텀 팩터
    for f in set(list(weights_large.keys()) + list(weights_small.keys())):
        if f not in all_known:
            wl = weights_large.get(f, 0)
            ws = weights_small.get(f, 0)
            if wl > 0 or ws > 0:
                rows.append({
                    "카테고리": "커스텀",
                    "팩터": FACTOR_LABELS.get(f, f),
                    "대형주": f"{wl*100:.0f}%" if wl > 0 else "-",
                    "중소형주": f"{ws*100:.0f}%" if ws > 0 else "-",
                    "_wl": wl,
                    "_ws": ws,
                    "_code": f,
                })

    return rows


def extract_strategy_summary(strategy_code: str) -> dict:
    """전략 코드에서 요약 정보를 추출한다."""
    try:
        namespace = {}
        exec(compile(strategy_code, "<strategy>", "exec"), {"__builtins__": {}}, namespace)
    except Exception:
        return {}

    scoring_mode = namespace.get("SCORING_MODE", {"large": "quartile", "small": "decile"})
    weights_large = namespace.get("WEIGHTS_LARGE", {})
    weights_small = namespace.get("WEIGHTS_SMALL", {})
    regression_models = namespace.get("REGRESSION_MODELS", [])

    # 카테고리별 합계 (LARGE 기준)
    cat_totals = {}
    for cat, factors in FACTOR_CATEGORIES.items():
        total = sum(weights_large.get(f, 0) for f in factors)
        if total > 0:
            cat_totals[cat] = total

    # 커스텀 팩터 카테고리
    all_known = set()
    for factors in FACTOR_CATEGORIES.values():
        all_known.update(factors)
    custom_total = sum(v for k, v in weights_large.items() if k not in all_known and v > 0)
    if custom_total > 0:
        cat_totals["커스텀"] = custom_total

    return {
        "scoring_mode": scoring_mode,
        "n_factors_large": sum(1 for v in weights_large.values() if v > 0),
        "n_factors_small": sum(1 for v in weights_small.values() if v > 0),
        "n_regressions": len(regression_models),
        "cat_totals": cat_totals,
        "large_only": [f for f in weights_large if weights_large[f] > 0 and weights_small.get(f, 0) == 0],
        "small_only": [f for f in weights_small if weights_small[f] > 0 and weights_large.get(f, 0) == 0],
    }
