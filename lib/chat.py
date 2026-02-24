"""
AI 채팅 패널 — 왼쪽 컬럼에 상시 표시되는 대화 인터페이스
일반 Q&A와 전략 실험실 수정을 모두 지원
"""
import json
import streamlit as st
from lib.ai import (
    is_ai_available, _get_client, MODEL_FAST,
    chat_strategy_modification, CHAT_EXAMPLES,
)
from lib.factor_engine import DEFAULT_STRATEGY_CODE


_SYSTEM_GENERAL = """당신은 자산운용사의 시니어 퀀트 애널리스트 AI 어시스턴트입니다.
대시보드에 표시된 백테스트 결과를 바탕으로 사용자의 질문에 답합니다.

규칙:
- 한국어로 응답합니다.
- 전문 금융 용어를 사용하되, 코드 변수명이나 프로그래밍 용어는 절대 사용하지 마세요.
- 답변은 간결하고 핵심적으로. 너무 길지 않게.
- 수치를 근거로 들어 답변하세요.
- 대시보드에 표시된 전략: 기존전략(사분위 밸류 기반), 회귀only(회귀 매력도 기반), KOSPI 200(벤치마크)
"""


def render_chat_column(active_page: str):
    """왼쪽 컬럼에 AI 채팅 패널 렌더링"""
    is_lab = (active_page == "전략 실험실")

    if not is_ai_available():
        st.markdown("#### AI 어시스턴트")
        st.info("ANTHROPIC_API_KEY를 설정하면 AI 채팅을 사용할 수 있습니다.")
        return

    # Header
    if is_lab:
        st.markdown("#### 전략 수정 AI")
        st.caption("가중치, 팩터, 회귀 모델, 파라미터 등 자유롭게 수정")
    else:
        st.markdown("#### AI 어시스턴트")

    # 세션 히스토리 선택
    history_key = "lab_messages" if is_lab else "chat_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    messages = st.session_state[history_key]

    # 채팅 히스토리 (스크롤 가능 컨테이너)
    container = st.container(height=520)
    with container:
        if not messages:
            if is_lab:
                st.caption("예시:")
                st.caption("&bull; 밸류 비중을 50%로 올려줘")
                st.caption("&bull; PBR vs 매출성장 커스텀 회귀 추가")
                st.caption("&bull; Forward EPS 모멘텀 팩터 제거")
            else:
                st.caption("무엇이든 물어보세요:")
                st.caption("&bull; 두 전략 중 어느 게 더 나아?")
                st.caption("&bull; MDD가 큰 이유가 뭘까?")
                st.caption("&bull; OOS 성과를 분석해줘")

        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 입력 폼
    with st.form(f"chat_form_{history_key}", clear_on_submit=True):
        user_input = st.text_input(
            "",
            placeholder="전략 수정 요청..." if is_lab else "질문하기...",
            label_visibility="collapsed",
            key=f"input_{history_key}",
        )
        submitted = st.form_submit_button("전송", use_container_width=True)

    if submitted and user_input:
        messages.append({"role": "user", "content": user_input})

        if is_lab:
            _handle_lab_message(messages)
        else:
            _handle_general_message(messages)

        st.rerun()

    # 대화 초기화 버튼
    if messages:
        if st.button("대화 초기화", key=f"clear_{history_key}", use_container_width=True):
            st.session_state[history_key] = []
            st.rerun()


def _handle_general_message(messages: list):
    """일반 Q&A 메시지 처리"""
    # 기본 성과 요약 (항상 포함) + 현재 페이지 상세 컨텍스트
    base = st.session_state.get("base_context", {})
    page = st.session_state.get("page_context", {})
    combined = {}
    if base:
        combined["전략_성과_요약"] = base
    if page:
        combined["현재_페이지"] = page
    context_str = ""
    if combined:
        context_str = f"\n\n현재 대시보드 데이터:\n{json.dumps(combined, ensure_ascii=False, default=str)}"

    client = _get_client()
    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    try:
        response = client.messages.create(
            model=MODEL_FAST,
            max_tokens=1000,
            system=_SYSTEM_GENERAL + context_str,
            messages=api_messages,
        )
        reply = response.content[0].text
    except Exception as e:
        reply = f"오류가 발생했습니다: {e}"

    messages.append({"role": "assistant", "content": reply})


def _handle_lab_message(messages: list):
    """전략 수정 메시지 처리 (코드 생성 방식)"""
    # 현재 전략 코드 가져오기
    current_code = st.session_state.get("lab_strategy_code", DEFAULT_STRATEGY_CODE)

    try:
        response_text, updated_code, changes_summary = chat_strategy_modification(
            messages,
            current_code,
        )
    except Exception as e:
        messages.append({"role": "assistant", "content": f"오류: {e}"})
        return

    display_text = response_text or ""

    if updated_code:
        # 전략 코드 업데이트
        st.session_state.lab_strategy_code = updated_code
        # 이전 백테스트 결과 초기화
        st.session_state.lab_modified_results = None

        if changes_summary:
            if display_text:
                display_text += "\n\n"
            display_text += "**적용된 변경:**\n" + "\n".join(f"- {c}" for c in changes_summary)

    if not display_text:
        display_text = "요청을 처리했습니다."

    messages.append({"role": "assistant", "content": display_text})


# ═══════════════════════════════════════════════════════
# 컨텍스트 빌더
# ═══════════════════════════════════════════════════════

def build_performance_context(results: dict) -> dict:
    """성과 비교 페이지용 컨텍스트"""
    ctx = {}
    for key in ["A0", "ATT2", "KOSPI"]:
        r = results.get(key)
        if not r or not isinstance(r, dict):
            continue
        ctx[key] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in r.items()
            if k in ("total_return", "cagr", "mdd", "sharpe", "avg_turnover", "avg_portfolio_size")
        }
    return {"page": "전략 성과 비교", "results": ctx}


def build_stat_context(stat_data: dict, rolling_all: dict) -> dict:
    """통계 검증 페이지용 컨텍스트"""
    ctx = {"page": "통계 검증"}
    sig = {}
    for key in ["A0", "ATT2"]:
        s = stat_data.get("bm_significance", {}).get(key)
        if s:
            sig[key] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in s.items()
                if k != "boot_means"
            }
    ctx["significance"] = sig
    roll = {}
    for key in ["A0", "ATT2"]:
        rd = rolling_all.get(key)
        if rd:
            roll[key] = {k: v for k, v in rd.items() if k != "windows"}
    ctx["rolling"] = roll
    return ctx


def _df_to_summary(df, cols=None, limit=30):
    """DataFrame → 컨텍스트용 list[dict] 변환 (상위 limit개)."""
    if df is None or not hasattr(df, "empty") or df.empty:
        return []
    if cols:
        cols = [c for c in cols if c in df.columns]
    else:
        cols = list(df.columns)
    return df[cols].head(limit).to_dict(orient="records")


def build_portfolio_context(common, a0_only, att2_only, selected_date: str,
                            a0_chars: dict = None, att2_chars: dict = None) -> dict:
    """포트폴리오 구성 페이지용 컨텍스트"""
    stock_cols = ["종목명", "섹터", "비중(%)", "PER", "PBR", "EV/EBITDA"]

    ctx = {
        "page": "포트폴리오 구성",
        "rebalance_date": selected_date,
        "공통종목": _df_to_summary(common, stock_cols),
        "기존전략_단독": _df_to_summary(a0_only, stock_cols),
        "회귀only_단독": _df_to_summary(att2_only, stock_cols),
    }
    if a0_chars:
        ctx["기존전략_가중평균"] = a0_chars
    if att2_chars:
        ctx["회귀only_가중평균"] = att2_chars
    return ctx
