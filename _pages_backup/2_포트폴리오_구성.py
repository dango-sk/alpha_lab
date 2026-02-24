"""
Page 3: 포트폴리오 구성 — A0 vs ATT2 종목 비교
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from lib.data import (
    load_backtest_results, get_stock_comparison,
    STRATEGY_KEYS, STRATEGY_LABELS, BACKTEST_CONFIG,
)
from lib.charts import sector_pie_chart
from lib.style import inject_css, section_header
from lib.chat import render_chat, build_portfolio_context

st.set_page_config(page_title="포트폴리오 구성", page_icon="📋", layout="wide")
inject_css()
st.title("포트폴리오 구성")

with st.spinner("백테스트 데이터 준비 중..."):
    results = load_backtest_results()

# ─── Sidebar: 리밸런싱 날짜 선택 ───
ref_key = next((k for k in STRATEGY_KEYS if k in results), None)
if not ref_key:
    st.error("백테스트 결과가 없습니다.")
    st.stop()

rb_dates = results[ref_key]["rebalance_dates"]
selected_date = st.sidebar.selectbox(
    "리밸런싱 날짜",
    rb_dates[:-1],
    index=len(rb_dates) - 2,
)

# ─── 종목 비교 ───
with st.spinner("종목 비교 데이터 조회 중..."):
    common, a0_only, att2_only = get_stock_comparison(selected_date)

n_common = len(common)
n_a0 = len(a0_only)
n_att2 = len(att2_only)

render_chat(build_portfolio_context(common, a0_only, att2_only, selected_date))

st.subheader(f"A0 vs ATT2 종목 비교 — {selected_date}")

# 요약 지표
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("공통 보유", f"{n_common}종목")
with col2:
    st.metric("A0 단독", f"{n_a0}종목")
with col3:
    st.metric("ATT2 단독", f"{n_att2}종목")

# 공통 종목
if not common.empty:
    st.subheader(f"공통 보유 종목 ({n_common}개)")
    st.caption("두 전략 모두 편입한 종목")
    st.dataframe(
        common,
        width="stretch",
        hide_index=True,
        height=min(len(common) * 35 + 40, 400),
    )

# A0 단독
if not a0_only.empty:
    st.subheader(f"A0 단독 종목 ({n_a0}개)")
    st.caption("원본 사분위 밸류 전략에만 편입된 종목")
    st.dataframe(
        a0_only,
        width="stretch",
        hide_index=True,
        height=min(len(a0_only) * 35 + 40, 400),
    )

# ATT2 단독
if not att2_only.empty:
    st.subheader(f"ATT2 단독 종목 ({n_att2}개)")
    st.caption("회귀 매력도 전략에만 편입된 종목")
    st.dataframe(
        att2_only,
        width="stretch",
        hide_index=True,
        height=min(len(att2_only) * 35 + 40, 400),
    )

# ─── 섹터 비교 ───
if not common.empty or not a0_only.empty or not att2_only.empty:
    st.subheader("섹터 비중 비교")
    col1, col2 = st.columns(2)

    # A0 전체 (공통 + A0단독)
    from lib.data import get_holdings
    a0_holdings = get_holdings("A0", selected_date)
    att2_holdings = get_holdings("ATT2", selected_date)

    with col1:
        st.caption("A0: 원본 사분위 밸류")
        if not a0_holdings.empty:
            st.plotly_chart(sector_pie_chart(a0_holdings), width="stretch")

    with col2:
        st.caption("ATT2: 회귀 매력도")
        if not att2_holdings.empty:
            st.plotly_chart(sector_pie_chart(att2_holdings), width="stretch")
