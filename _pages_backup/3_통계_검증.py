"""
Page 4: 통계 검증
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from lib.data import load_robustness_results, STRATEGY_KEYS, STRATEGY_LABELS, BACKTEST_CONFIG
from lib.charts import bootstrap_histogram, is_oos_comparison_chart, rolling_window_chart
from lib.style import inject_css, section_header, color_value
from lib.chat import render_chat, build_stat_context

st.set_page_config(page_title="통계 검증", page_icon="📐", layout="wide")
inject_css()
st.title("통계 검증")
st.caption("모든 검증은 KOSPI 200 대비 초과수익률 기준")

# ─── Load robustness ───
with st.spinner("강건성 검증 데이터 준비 중..."):
    is_oos_data, stat_data, rolling_all = load_robustness_results()

# ─── Sidebar: AI 대화창 ───
render_chat(build_stat_context(stat_data, rolling_all))

# ═══════════════════════════════════════════════════════
# Section 1: IS vs OOS
# ═══════════════════════════════════════════════════════
section_header("1. In-Sample vs Out-of-Sample")
st.caption(
    f"IS: {BACKTEST_CONFIG['start']} ~ {BACKTEST_CONFIG.get('insample_end', '2024-06-30')}  |  "
    f"OOS: {BACKTEST_CONFIG.get('oos_start', '2024-07-01')} ~ {BACKTEST_CONFIG['end']}"
)

st.plotly_chart(is_oos_comparison_chart(is_oos_data), width="stretch")

# IS/OOS detail table
is_oos_table = []
for key in STRATEGY_KEYS:
    is_r = is_oos_data["is_results"].get(key, {})
    oos_r = is_oos_data["oos_results"].get(key, {})
    is_oos_table.append({
        "전략": STRATEGY_LABELS[key],
        "IS 수익률": f"{is_r.get('total_return', 0):+.1%}",
        "IS CAGR": f"{is_r.get('cagr', 0):+.1%}",
        "IS Sharpe": f"{is_r.get('sharpe', 0):.3f}",
        "IS MDD": f"{is_r.get('mdd', 0):.1%}",
        "OOS 수익률": f"{oos_r.get('total_return', 0):+.1%}",
        "OOS CAGR": f"{oos_r.get('cagr', 0):+.1%}",
        "OOS Sharpe": f"{oos_r.get('sharpe', 0):.3f}",
        "OOS MDD": f"{oos_r.get('mdd', 0):.1%}",
    })

# BM row
bm = is_oos_data.get("benchmarks", {})
is_bm = bm.get("is", {}).get("KOSPI", {})
oos_bm = bm.get("oos", {}).get("KOSPI", {})
if is_bm or oos_bm:
    is_oos_table.append({
        "전략": "BM: KOSPI 200",
        "IS 수익률": f"{is_bm.get('total_return', 0):+.1%}",
        "IS CAGR": f"{is_bm.get('cagr', 0):+.1%}",
        "IS Sharpe": "-",
        "IS MDD": "-",
        "OOS 수익률": f"{oos_bm.get('total_return', 0):+.1%}",
        "OOS CAGR": f"{oos_bm.get('cagr', 0):+.1%}",
        "OOS Sharpe": "-",
        "OOS MDD": "-",
    })

df_isoos = pd.DataFrame(is_oos_table)
styled = df_isoos.style.map(
    lambda v: color_value(v), subset=["IS 수익률", "IS CAGR", "OOS 수익률", "OOS CAGR"]
).map(
    lambda v: color_value(v, reverse=True), subset=["IS MDD", "OOS MDD"]
)
st.dataframe(styled, width="stretch", hide_index=True)

# ═══════════════════════════════════════════════════════
# Section 2: Bootstrap + t-test
# ═══════════════════════════════════════════════════════
section_header("2. 통계적 유의성 (Bootstrap 10,000회 + t-test)")

st.plotly_chart(bootstrap_histogram(stat_data), width="stretch")

sig_table = []
for key in STRATEGY_KEYS:
    sig = stat_data["bm_significance"].get(key)
    if not sig:
        continue
    sig_table.append({
        "전략": STRATEGY_LABELS[key],
        "월평균 초과수익": f"{sig['mean_diff']*100:+.3f}%",
        "t-stat": f"{sig['t_stat']:.2f}",
        "p-value": f"{sig['p_value']:.4f}",
        "95% CI 하한": f"{sig['ci_lower']*100:+.3f}%",
        "95% CI 상한": f"{sig['ci_upper']*100:+.3f}%",
        "Bootstrap 승률": f"{sig['win_rate']:.1%}",
        "유의 여부": "유의 (p<0.05)" if sig['significant'] else "유의하지 않음",
    })

df_sig = pd.DataFrame(sig_table)
styled_sig = df_sig.style.map(
    lambda v: color_value(v), subset=["월평균 초과수익"]
).map(
    lambda v: "color: #4CAF50; font-weight: 600" if v == "유의 (p<0.05)" else "color: #EF5350" if "유의하지" in str(v) else "",
    subset=["유의 여부"]
)
st.dataframe(styled_sig, width="stretch", hide_index=True)

# ═══════════════════════════════════════════════════════
# Section 3: Rolling window
# ═══════════════════════════════════════════════════════
section_header("3. 롤링 24개월 윈도우")

st.plotly_chart(rolling_window_chart(rolling_all), width="stretch")

rolling_table = []
for key in STRATEGY_KEYS:
    rd = rolling_all.get(key)
    if not rd:
        continue
    rolling_table.append({
        "전략": STRATEGY_LABELS[key],
        "총 윈도우": rd["total_windows"],
        "양의 알파": rd["positive_windows"],
        "승률": f"{rd['win_rate']:.0%}",
    })

df_roll = pd.DataFrame(rolling_table)
styled_roll = df_roll.style.map(
    lambda v: color_value(v), subset=["승률"]
)
st.dataframe(styled_roll, width="stretch", hide_index=True)
