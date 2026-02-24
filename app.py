"""
Alpha Lab — 퀀트 전략 백테스트 대시보드
왼쪽: AI 채팅 | 오른쪽: 분석 콘텐츠
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from lib.style import inject_css
from lib.views import (
    render_performance, render_monthly,
    render_portfolio, render_statistics, render_lab_content,
)
from lib.chat import render_chat_column, build_performance_context
from lib.data import get_latest_price_date, load_all_results, BACKTEST_CONFIG

st.set_page_config(page_title="Alpha Lab", page_icon="📊", layout="wide")
inject_css()

# ─── 상단 네비게이션 ───
NAV_ITEMS = ["성과 비교", "월별 분석", "포트폴리오", "통계 검증", "전략 실험실"]
page = st.radio(
    "", NAV_ITEMS,
    horizontal=True, label_visibility="collapsed", key="active_page",
)
latest = get_latest_price_date()
if latest:
    st.markdown(
        f'<p style="text-align:right;color:#888;font-size:0.78rem;margin:-0.3rem 0 0.2rem 0">'
        f'주가 데이터: {latest}</p>',
        unsafe_allow_html=True,
    )
st.divider()

# ─── 기본 성과 컨텍스트 (AI 채팅용, 모든 탭에서 공유) ───
if "base_context" not in st.session_state:
    _base_results = load_all_results(BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"])
    st.session_state["base_context"] = build_performance_context(_base_results)

# ─── 2열 레이아웃: 채팅 | 콘텐츠 ───
chat_col, main_col = st.columns([3, 7], gap="medium")

# 콘텐츠 (우측) — 먼저 실행하여 page_context 세팅
with main_col:
    if page == "성과 비교":
        render_performance()
    elif page == "월별 분석":
        render_monthly()
    elif page == "포트폴리오":
        render_portfolio()
    elif page == "통계 검증":
        render_statistics()
    elif page == "전략 실험실":
        render_lab_content()

# 채팅 (좌측) — page_context 읽어서 사용
with chat_col:
    render_chat_column(page)
