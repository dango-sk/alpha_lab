"""
Page 2: 월별 분석
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from lib.data import load_backtest_results
from lib.charts import monthly_heatmap, rolling_excess_chart, monthly_distribution_chart
from lib.style import inject_css, section_header
from lib.chat import render_chat, build_performance_context

st.set_page_config(page_title="월별 분석", page_icon="📅", layout="wide")
inject_css()
st.title("월별 분석")

with st.spinner("백테스트 데이터 준비 중..."):
    results = load_backtest_results()

render_chat(build_performance_context(results))

# ─── Heatmap ───
section_header("월별 수익률 히트맵")
st.plotly_chart(monthly_heatmap(results), width="stretch")

# ─── Rolling excess ───
section_header("롤링 12개월 누적 초과수익률 vs KOSPI 200")
st.plotly_chart(rolling_excess_chart(results, window=12), width="stretch")

# ─── Distribution ───
section_header("월별 수익률 분포")
st.plotly_chart(monthly_distribution_chart(results), width="stretch")
