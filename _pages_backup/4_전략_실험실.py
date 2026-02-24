"""
Page 5: 전략 실험실 — AI 기반 전략 수정 + 비교
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

from lib.data import (
    load_backtest_results, run_custom_backtest,
    save_strategy, load_strategy, list_strategies, delete_strategy,
    STRATEGY_LABELS, BACKTEST_CONFIG,
)
from lib.ai import (
    is_ai_available, chat_strategy_modification,
    format_weights_for_display, FACTOR_LABELS, FACTOR_CATEGORIES,
)
from lib.charts import strategy_weight_chart, comparison_cumulative_chart
from lib.style import inject_css, section_header

st.set_page_config(page_title="전략 실험실", page_icon="🧪", layout="wide")
inject_css()
st.title("전략 실험실")
st.caption("AI와 대화하며 전략을 자유롭게 수정하고, 수정된 전략의 성과를 비교해보세요.")

# ─── Default weights (from step3 WEIGHTS_LARGE) ───
DEFAULT_WEIGHTS = {
    "T_PER": .05, "F_PER": .05, "T_EVEBITDA": .05, "F_EVEBITDA": .05,
    "T_PBR": .05, "F_PBR": .05, "T_PCF": .05,
    "ATT_PBR": .05, "ATT_EVIC": .05, "ATT_PER": .10, "ATT_EVEBIT": .10,
    "T_SPSG": .10, "F_SPSG": .10,
    "F_EPS_M": .15,
}
DEFAULT_PARAMS = {
    "top_n": BACKTEST_CONFIG["top_n_stocks"],
    "tx_cost_bp": BACKTEST_CONFIG["transaction_cost_bp"],
    "weight_cap": BACKTEST_CONFIG.get("weight_cap_pct", 15),
}

# ─── Session state ───
if "lab_weights" not in st.session_state:
    st.session_state.lab_weights = dict(DEFAULT_WEIGHTS)
if "lab_params" not in st.session_state:
    st.session_state.lab_params = dict(DEFAULT_PARAMS)
if "lab_messages" not in st.session_state:
    st.session_state.lab_messages = []
if "lab_modified_results" not in st.session_state:
    st.session_state.lab_modified_results = None


def _params_changed() -> bool:
    """파라미터가 기본값과 다른지 확인"""
    p = st.session_state.lab_params
    return (
        p["top_n"] != DEFAULT_PARAMS["top_n"]
        or p["tx_cost_bp"] != DEFAULT_PARAMS["tx_cost_bp"]
        or p["weight_cap"] != DEFAULT_PARAMS["weight_cap"]
    )


def _weights_changed() -> bool:
    """가중치가 기본값과 다른지 확인"""
    for k, v in DEFAULT_WEIGHTS.items():
        if st.session_state.lab_weights.get(k, 0) != v:
            return True
    return False


# ═══════════════════════════════════════════════════════
# Sidebar: 전략 관리 (저장/불러오기)
# ═══════════════════════════════════════════════════════
st.sidebar.header("전략 관리")

# 저장
with st.sidebar.expander("전략 저장", expanded=False):
    save_name = st.text_input("전략 이름", key="strat_save_name")
    save_memo = st.text_area("메모 (선택)", key="strat_save_memo", height=80)
    if st.button("저장", key="btn_save") and save_name:
        save_strategy(
            name=save_name,
            weights=st.session_state.lab_weights,
            params=st.session_state.lab_params,
            results=st.session_state.lab_modified_results,
            memo=save_memo,
        )
        st.success(f"'{save_name}' 저장 완료")

# 불러오기
strategies = list_strategies()
if strategies:
    with st.sidebar.expander("저장된 전략", expanded=False):
        strat_names = [s["name"] for s in strategies]
        selected = st.selectbox("전략 선택", strat_names, key="strat_load_select")

        sel_data = next((s for s in strategies if s["name"] == selected), None)
        if sel_data:
            st.caption(f"생성: {sel_data.get('created_at', '')[:10]}")
            if sel_data.get("memo"):
                st.caption(sel_data["memo"])
            if sel_data.get("summary"):
                for k, v in sel_data["summary"].items():
                    st.caption(f"{k}: {v}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("불러오기", key="btn_load"):
                data = load_strategy(selected)
                if data:
                    st.session_state.lab_weights = data.get("weights", dict(DEFAULT_WEIGHTS))
                    st.session_state.lab_params = data.get("params", dict(DEFAULT_PARAMS))
                    st.session_state.lab_modified_results = data.get("results")
                    st.session_state.lab_messages = []
                    st.rerun()
        with col2:
            if st.button("삭제", key="btn_delete"):
                delete_strategy(selected)
                st.rerun()

# 초기화
if st.sidebar.button("전략 초기화", key="btn_reset"):
    st.session_state.lab_weights = dict(DEFAULT_WEIGHTS)
    st.session_state.lab_params = dict(DEFAULT_PARAMS)
    st.session_state.lab_messages = []
    st.session_state.lab_modified_results = None
    st.rerun()


# ═══════════════════════════════════════════════════════
# Section 1: 현재 전략 구조
# ═══════════════════════════════════════════════════════
st.subheader("현재 전략 구조")

# 파라미터 카드
p = st.session_state.lab_params
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("편입 종목수", f"{p['top_n']}개")
with col2:
    st.metric("편도 거래비용", f"{p['tx_cost_bp']}bp")
with col3:
    st.metric("개별종목 비중상한", f"{p['weight_cap']}%")

# 가중치 테이블 + 차트
tab1, tab2 = st.tabs(["가중치 테이블", "가중치 차트"])

with tab1:
    weights_data = format_weights_for_display(st.session_state.lab_weights)
    if weights_data:
        df = pd.DataFrame(weights_data)
        # 카테고리별 소계 추가
        cat_totals = {}
        for row in weights_data:
            cat = row["카테고리"]
            cat_totals[cat] = cat_totals.get(cat, 0) + row["_weight"]

        display_df = df[["카테고리", "팩터", "비중"]].copy()
        st.dataframe(display_df, width="stretch", hide_index=True)

        # 카테고리 소계
        total = sum(cat_totals.values())
        summary = " | ".join(f"{c} {v*100:.0f}%" for c, v in cat_totals.items())
        st.caption(f"카테고리 합계: {summary} | **전체 {total*100:.0f}%**")
    else:
        st.info("모든 팩터 가중치가 0입니다.")

with tab2:
    st.plotly_chart(
        strategy_weight_chart(st.session_state.lab_weights),
        width="stretch",
    )

# 변경 감지 표시
if _weights_changed():
    st.info("팩터 가중치가 기본값에서 변경되었습니다. (시각적 변경 — 백테스트에는 파라미터 변경만 반영됩니다)")
if _params_changed():
    st.success("운용 파라미터가 변경되었습니다. 아래에서 백테스트를 실행하여 성과를 비교해보세요.")


# ═══════════════════════════════════════════════════════
# Section 2: AI 채팅 인터페이스
# ═══════════════════════════════════════════════════════
st.divider()
st.subheader("전략 수정")

if not is_ai_available():
    st.warning(
        "ANTHROPIC_API_KEY가 설정되지 않아 AI 전략 수정을 사용할 수 없습니다. "
        "환경변수에 `ANTHROPIC_API_KEY`를 설정해주세요."
    )
else:
    st.caption(
        "자연어로 전략 수정을 요청하세요.  "
        "예: \"종목수를 20개로 줄여봐\", \"거래비용을 50bp로 올려봐\", "
        "\"밸류에이션 비중을 50%로 올려줘\""
    )

    # Display chat history
    for msg in st.session_state.lab_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("전략 수정 요청..."):
        # Add user message
        st.session_state.lab_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call Claude (modifies weights/params in-place)
        with st.spinner("AI가 분석 중..."):
            response_text, tool_calls = chat_strategy_modification(
                st.session_state.lab_messages,
                st.session_state.lab_weights,
                st.session_state.lab_params,
            )

        # Build change summary from tool_calls
        changes = []
        for tc in tool_calls:
            name = tc["name"]
            inp = tc["input"]
            if name == "set_factor_weight":
                label = FACTOR_LABELS.get(inp["factor"], inp["factor"])
                changes.append(f"- {label}: {inp['weight']*100:.0f}%")
            elif name == "remove_factor":
                label = FACTOR_LABELS.get(inp["factor"], inp["factor"])
                changes.append(f"- {label} 제거")
            elif name == "set_portfolio_size":
                changes.append(f"- 편입 종목수: {inp['n']}개")
            elif name == "set_transaction_cost":
                changes.append(f"- 편도 거래비용: {inp['bp']}bp")
            elif name == "set_weight_cap":
                changes.append(f"- 개별종목 비중상한: {inp['cap_pct']}%")

        # Build display text
        display_text = response_text or ""
        if changes:
            if display_text:
                display_text += "\n\n"
            display_text += "**적용된 변경:**\n" + "\n".join(changes)

        if not display_text:
            display_text = "요청을 처리했습니다."

        st.session_state.lab_messages.append({"role": "assistant", "content": display_text})
        with st.chat_message("assistant"):
            st.markdown(display_text)

        # Clear previous results when changes are made
        if changes:
            st.session_state.lab_modified_results = None

        st.rerun()


# ═══════════════════════════════════════════════════════
# Section 3: 백테스트 실행 + 비교
# ═══════════════════════════════════════════════════════
if _params_changed() or st.session_state.lab_modified_results:
    st.divider()
    st.subheader("성과 비교")

    if _params_changed() and not st.session_state.lab_modified_results:
        if st.button("백테스트 실행", type="primary", key="btn_backtest"):
            with st.spinner("수정된 전략 백테스트 실행 중 (약 1-2분)..."):
                modified = run_custom_backtest(
                    top_n=st.session_state.lab_params["top_n"],
                    tx_cost_bp=st.session_state.lab_params["tx_cost_bp"],
                    weight_cap=st.session_state.lab_params["weight_cap"],
                )
                st.session_state.lab_modified_results = modified
                st.rerun()

    if st.session_state.lab_modified_results:
        # Load original for comparison
        with st.spinner("기존 결과 로딩 중..."):
            original = load_backtest_results()
        modified = st.session_state.lab_modified_results

        # Comparison table
        comp_data = []
        for key in ["A0", "KOSPI"]:
            r = original.get(key, {})
            if r:
                label = STRATEGY_LABELS.get(key, key)
                row = {
                    "전략": f"기존 {label}",
                    "총수익률": f"{r.get('total_return', 0):+.1%}",
                    "CAGR": f"{r.get('cagr', 0):+.1%}",
                    "MDD": f"{r.get('mdd', 0):.1%}",
                    "Sharpe": f"{r.get('sharpe', 0):.2f}",
                }
                if "avg_turnover" in r:
                    row["회전율"] = f"{r['avg_turnover']:.1%}"
                comp_data.append(row)

        m = modified.get("A0", {})
        if m:
            row = {
                "전략": "수정 전략",
                "총수익률": f"{m.get('total_return', 0):+.1%}",
                "CAGR": f"{m.get('cagr', 0):+.1%}",
                "MDD": f"{m.get('mdd', 0):.1%}",
                "Sharpe": f"{m.get('sharpe', 0):.2f}",
            }
            if "avg_turnover" in m:
                row["회전율"] = f"{m['avg_turnover']:.1%}"
            comp_data.append(row)

        st.dataframe(comp_data, width="stretch", hide_index=True)

        # Comparison chart
        st.plotly_chart(
            comparison_cumulative_chart(original, modified),
            width="stretch",
        )

        # 변경 내용 요약
        param_changes = []
        if st.session_state.lab_params["top_n"] != DEFAULT_PARAMS["top_n"]:
            param_changes.append(
                f"종목수: {DEFAULT_PARAMS['top_n']} → {st.session_state.lab_params['top_n']}"
            )
        if st.session_state.lab_params["tx_cost_bp"] != DEFAULT_PARAMS["tx_cost_bp"]:
            param_changes.append(
                f"거래비용: {DEFAULT_PARAMS['tx_cost_bp']}bp → {st.session_state.lab_params['tx_cost_bp']}bp"
            )
        if st.session_state.lab_params["weight_cap"] != DEFAULT_PARAMS["weight_cap"]:
            param_changes.append(
                f"비중상한: {DEFAULT_PARAMS['weight_cap']}% → {st.session_state.lab_params['weight_cap']}%"
            )
        if param_changes:
            st.caption("변경된 파라미터: " + " | ".join(param_changes))

        # Re-run button
        if st.button("파라미터 변경 후 재실행", key="btn_rerun"):
            st.session_state.lab_modified_results = None
            st.rerun()
