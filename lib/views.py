"""
대시보드 뷰 함수 — 각 탭의 콘텐츠를 렌더링
"""
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

from config.settings import IS_DEV
from lib.data import (
    load_backtest_results, load_all_results, load_robustness_results,
    load_all_robustness_results,
    get_holdings, get_monthly_attribution,
    get_portfolio_characteristics, get_portfolio_turnover,
    run_strategy_backtest, save_strategy, load_strategy, list_strategies, delete_strategy,
    STRATEGY_KEYS, ALL_KEYS, STRATEGY_LABELS, STRATEGY_COLORS, BACKTEST_CONFIG,
    BASE_STRATEGY_WEIGHTS,
)
from lib.charts import (
    cumulative_return_chart, drawdown_chart,
    monthly_heatmap, rolling_excess_chart, attribution_chart,
    bootstrap_histogram, is_oos_comparison_chart, rolling_window_chart,
    sector_pie_chart, sector_comparison_chart,
    strategy_weight_chart, comparison_cumulative_chart,
    market_cap_distribution_chart, concentration_chart,
)
from lib.ai import (
    is_ai_available, format_weights_for_display, extract_strategy_summary,
    FACTOR_LABELS, FACTOR_CATEGORIES,
)
from lib.factor_engine import DEFAULT_STRATEGY_CODE
from lib.style import kpi_card, section_header, color_value, show_loading
from lib.chat import build_performance_context, build_stat_context

_dt = __import__("datetime")


# ═══════════════════════════════════════════════════════
# 공통: 기간 설정 (session_state 기반, 전 탭 공유)
# ═══════════════════════════════════════════════════════
def _init_period():
    """session_state에 기간 기본값 설정"""
    if "bt_start" not in st.session_state:
        st.session_state.bt_start = date.fromisoformat(BACKTEST_CONFIG["start"])
    if "bt_end" not in st.session_state:
        st.session_state.bt_end = date.fromisoformat(BACKTEST_CONFIG["end"])
    if "bt_split" not in st.session_state:
        total_days = (st.session_state.bt_end - st.session_state.bt_start).days
        st.session_state.bt_split = st.session_state.bt_start + _dt.timedelta(days=int(total_days * 0.7))


def _get_period() -> tuple[str, str, str, str]:
    """현재 기간 설정을 (start, end, is_end, oos_start) 문자열로 반환"""
    _init_period()
    start_str = st.session_state.bt_start.isoformat()
    end_str = st.session_state.bt_end.isoformat()
    is_end = st.session_state.bt_split.isoformat()
    oos_start = (st.session_state.bt_split + _dt.timedelta(days=1)).isoformat()
    return start_str, end_str, is_end, oos_start


def _clamp_split():
    """bt_start/bt_end 변경 시 bt_split을 유효 범위로 클램핑 (콜백)"""
    if "bt_split" in st.session_state:
        st.session_state.bt_split = max(
            st.session_state.bt_start,
            min(st.session_state.bt_split, st.session_state.bt_end),
        )


def render_period_selector():
    """기간 설정 UI (어느 탭에서든 호출 가능)"""
    _init_period()
    with st.expander("기간 설정", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.date_input(
                "시작일", value=st.session_state.bt_start,
                min_value=date(2018, 1, 1), max_value=st.session_state.bt_end,
                key="bt_start", on_change=_clamp_split,
            )
        with c2:
            st.date_input(
                "종료일", value=st.session_state.bt_end,
                min_value=st.session_state.bt_start, max_value=date(2026, 12, 31),
                key="bt_end", on_change=_clamp_split,
            )
        with c3:
            total_days = (st.session_state.bt_end - st.session_state.bt_start).days
            default_split = st.session_state.bt_start + _dt.timedelta(days=int(total_days * 0.7))
            split_val = st.session_state.get("bt_split", default_split)
            split_val = max(st.session_state.bt_start, min(split_val, st.session_state.bt_end))
            st.date_input(
                "IS/OOS 분할점", value=split_val,
                min_value=st.session_state.bt_start, max_value=st.session_state.bt_end,
                key="bt_split",
            )


# ═══════════════════════════════════════════════════════
# Dev 모드: 파라미터 조절 패널
# ═══════════════════════════════════════════════════════

def _init_params():
    """session_state에 조절 가능한 파라미터 기본값 설정"""
    if "param_weight_cap" not in st.session_state:
        st.session_state.param_weight_cap = BACKTEST_CONFIG.get("weight_cap_pct", 10)
    if "param_universe" not in st.session_state:
        st.session_state.param_universe = "KOSPI"
    if "param_min_mcap" not in st.session_state:
        st.session_state.param_min_mcap = int(BACKTEST_CONFIG.get("min_market_cap", 500_000_000_000) / 1e8)
    if "param_rebal_type" not in st.session_state:
        st.session_state.param_rebal_type = BACKTEST_CONFIG.get("rebal_type", "monthly")


def get_active_params() -> dict:
    """현재 활성화된 파라미터를 반환."""
    _init_params()
    return {
        "weight_cap_pct": st.session_state.param_weight_cap,
        "universe": st.session_state.param_universe,
        "top_n": st.session_state.get("param_top_n", BACKTEST_CONFIG.get("top_n_stocks", 30)),
        "min_market_cap": st.session_state.param_min_mcap * 1e8,
        "rebal_type": st.session_state.param_rebal_type,
    }


def render_param_panel():
    """파라미터 조절 패널."""
    _init_params()
    with st.expander("파라미터 조절", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.number_input(
                "비중캡 (%)", min_value=0, max_value=30, step=1,
                key="param_weight_cap",
                help="0% = 캡 없음 (시총비례 그대로), 10% = 개별종목 최대 10%",
            )
        with c2:
            st.number_input(
                "시총하한 (억원)", min_value=0, max_value=20000, step=500,
                key="param_min_mcap",
                help="유니버스 시총 하한 필터 (0 = 필터 없음)",
            )
        with c3:
            st.selectbox(
                "유니버스",
                options=["KOSPI", "KOSPI+KOSDAQ"],
                key="param_universe",
                help="KOSPI: 벤치마크 KODEX200 / KOSPI+KOSDAQ: 벤치마크 KRX300",
            )
        with c4:
            st.selectbox(
                "리밸런싱",
                options=["monthly", "biweekly"],
                format_func=lambda x: "월간" if x == "monthly" else "격주",
                key="param_rebal_type",
                help="월간: 매월 1회 / 격주: 월 2회",
            )


def _sync_labels(universe: str, weight_cap_pct: int = None):
    """유니버스·캡에 따라 전략 라벨을 동적으로 갱신."""
    STRATEGY_LABELS["KOSPI"] = "KRX 300" if universe == "KOSPI+KOSDAQ" else "KODEX 200"

    # 기존전략(A0)은 항상 디폴트 캡 표시
    default_cap = BACKTEST_CONFIG.get("weight_cap_pct", 10)
    cap_suffix = f" ({default_cap}%캡)" if default_cap > 0 else " (캡없음)"
    STRATEGY_LABELS["A0"] = "기존전략" + cap_suffix


def _render_universe_selector(key: str) -> str:
    """페이지별 유니버스 선택 라디오."""
    _init_params()
    default = st.session_state.get("param_universe", "KOSPI")
    options = ["KOSPI", "KOSPI+KOSDAQ"]
    selected = st.radio(
        "유니버스",
        options,
        index=options.index(default) if default in options else 0,
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )
    st.session_state["_active_universe"] = selected
    _sync_labels(selected)
    return selected


def render_strategy_filter(results: dict) -> dict:
    """전략 선택 필터 UI (멀티셀렉트). 선택된 전략만 포함된 results를 반환.

    위젯 key 기반으로 탭 간 선택 상태를 자동 공유한다.
    """
    available = [k for k in ALL_KEYS if k in results]
    # results에 있지만 ALL_KEYS에 없는 키도 포함 (유니버스 전환 시 벤치마크 등)
    for k in results:
        if k not in available:
            available.append(k)
    labels = {k: STRATEGY_LABELS.get(k, k) for k in available}

    _uni_suffix = st.session_state.get("_active_universe", "KOSPI")
    widget_key = f"strategy_filter_{_uni_suffix}"
    tracking_key = "_available_strategies"

    # available 전략이 바뀌면 widget 상태를 리셋하여 default가 적용되도록 함
    prev_available = sorted(st.session_state.get(tracking_key, []))
    curr_available = sorted(available)
    if prev_available != curr_available:
        st.session_state.pop(widget_key, None)
    st.session_state[tracking_key] = list(available)

    selected = st.multiselect(
        "전략 선택",
        options=available,
        default=available,
        format_func=lambda k: labels[k],
        key=widget_key,
        placeholder="표시할 전략을 선택하세요",
    )

    if not selected:
        st.warning("최소 1개 전략을 선택하세요.")
        return results

    return {k: v for k, v in results.items() if k in selected}


# ═══════════════════════════════════════════════════════
# 연도별 성과 분석 (월별/연도별 집계)
# ═══════════════════════════════════════════════════════

def _render_yearly_performance(results: dict, kpi_keys: list):
    """월별/연도별 성과 테이블 + 연도별 Sharpe/MDD."""
    for key in kpi_keys:
        r = results.get(key)
        if not r:
            continue

        label = STRATEGY_LABELS.get(key, key)
        rets = r["monthly_returns"]
        dates = r["rebalance_dates"][:-1]  # 시작일 기준

        if not rets or not dates:
            continue

        # ── 월별 수익률 테이블 (연도 × 월) ──
        st.subheader(f"{label}")
        monthly_data = {}
        for d, ret in zip(dates, rets):
            year = d[:4]
            month = int(d[5:7])
            monthly_data.setdefault(year, {})[month] = ret

        years = sorted(monthly_data.keys())
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        table_rows = []
        for year in years:
            row = {"연도": year}
            year_rets = []
            for m in range(1, 13):
                ret = monthly_data[year].get(m)
                if ret is not None:
                    row[month_names[m - 1]] = f"{ret * 100:+.1f}%"
                    year_rets.append(ret)
                else:
                    row[month_names[m - 1]] = ""
            # YTD (연간 누적)
            if year_rets:
                ytd = np.prod([1 + r for r in year_rets]) - 1
                row["YTD"] = f"{ytd * 100:+.1f}%"
            else:
                row["YTD"] = ""
            table_rows.append(row)

        df_monthly = pd.DataFrame(table_rows)
        # 색상 적용
        value_cols = month_names + ["YTD"]
        styled_monthly = df_monthly.style.map(
            lambda v: color_value(v), subset=[c for c in value_cols if c in df_monthly.columns]
        )
        st.dataframe(styled_monthly, width="stretch", hide_index=True)

        # ── 연도별 지표 (Sharpe, MDD, Drawdown) ──
        yearly_stats = []
        for year in years:
            year_rets_list = [monthly_data[year].get(m) for m in range(1, 13)
                              if monthly_data[year].get(m) is not None]
            if not year_rets_list:
                continue
            arr = np.array(year_rets_list)
            ytd = np.prod(1 + arr) - 1
            sharpe = (arr.mean() / arr.std() * np.sqrt(12)) if len(arr) > 1 and arr.std() > 0 else 0
            # 연도 내 MDD
            cum = np.cumprod(1 + arr)
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            mdd = float(dd.min())
            yearly_stats.append({
                "연도": year,
                "수익률": f"{ytd * 100:+.1f}%",
                "Sharpe": f"{sharpe:.2f}",
                "MDD": f"{mdd:.1%}",
                "월수": f"{len(year_rets_list)}",
            })

        if yearly_stats:
            df_yearly = pd.DataFrame(yearly_stats)
            styled_yearly = df_yearly.style.map(
                lambda v: color_value(v), subset=["수익률", "Sharpe"]
            ).map(
                lambda v: color_value(v, reverse=True), subset=["MDD"]
            )
            st.dataframe(styled_yearly, width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════
# 1. 전략 성과 비교
# ═══════════════════════════════════════════════════════
def render_performance():
    render_period_selector()
    render_param_panel()

    params = get_active_params()
    universe = params["universe"]
    rebal_type = params["rebal_type"]
    weight_cap_pct = params["weight_cap_pct"]
    min_market_cap = params["min_market_cap"]

    start_str, end_str, is_end, oos_start = _get_period()
    _sync_labels(universe, weight_cap_pct)
    cap_label = f"{weight_cap_pct}% 캡" if weight_cap_pct > 0 else "캡 없음"
    bm_label = "KRX 300" if universe == "KOSPI+KOSDAQ" else "KODEX 200"
    rebal_label = "격주" if rebal_type == "biweekly" else "월간"

    st.caption(
        f"기간: {start_str} ~ {end_str}  |  "
        f"리밸런싱: {rebal_label}, 상위 {BACKTEST_CONFIG['top_n_stocks']}종목  |  "
        f"비중: 시총비례 + {cap_label}  |  "
        f"시총하한: {min_market_cap/1e8:,.0f}억  |  "
        f"거래비용: 편도 {BACKTEST_CONFIG['transaction_cost_bp']}bp  |  "
        f"유니버스: {universe} (BM: {bm_label})"
    )

    loading = st.empty()
    show_loading(loading, "백테스트 결과를 불러오는 중")
    all_results = load_all_results(
        start_str, end_str,
        weight_cap_pct=weight_cap_pct,
        universe=universe,
    )
    loading.empty()

    results = render_strategy_filter(all_results)

    # 채팅 컨텍스트 저장
    st.session_state["page_context"] = build_performance_context(results)

    # KPI Cards
    kpi_keys = [k for k in ALL_KEYS if k in results]
    cols = st.columns(len(kpi_keys))
    for i, key in enumerate(kpi_keys):
        r = results.get(key)
        if not r:
            continue
        with cols[i]:
            kpi_card(
                label=STRATEGY_LABELS[key],
                value=f"{r['total_return']:+.1%}",
                sub_items=[("MDD", f"{r['mdd']:.1%}"), ("Sharpe", f"{r['sharpe']:.3f}")],
                color=STRATEGY_COLORS.get(key, "#757575"),
            )
    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison Table
    section_header("성과 비교")
    with st.expander("지표 설명", expanded=False):
        st.markdown("""
| 지표 | 설명 |
|------|------|
| **총수익률** | 백테스트 전 기간의 누적 수익률 (복리) |
| **CAGR** | 연환산 복합 성장률 — 매년 균등하게 벌었다면의 연수익률 |
| **MDD** | 고점 대비 최대 하락폭. 매 시점에서 직전 최고점 대비 하락률을 계산 → 그 중 최대값 |
| **Sharpe** | (월평균 수익률 / 월 수익률 표준편차) × √12. 위험 대비 수익 효율. 1 이상이면 양호 |
| **평균 회전율** | 매월 (신규 편입 + 편출 종목 수) / (2 × 전체 종목 수). 높으면 거래비용 증가 |
| **비중 산출** | 각 종목 시총 / 30종목 시총 합계 → 개별 상한 10% 적용 → 초과분 재배분 반복 |
""")
    table_data = []
    for key in kpi_keys:
        r = results.get(key)
        if not r:
            continue
        row = {"전략": STRATEGY_LABELS[key], "총수익률": f"{r['total_return']:+.1%}", "CAGR": f"{r['cagr']:+.1%}", "MDD": f"{r['mdd']:.1%}", "Sharpe": f"{r['sharpe']:.3f}"}
        if "avg_turnover" in r:
            row["평균 회전율"] = f"{r['avg_turnover']:.1%}"
            row["평균 종목수"] = f"{r['avg_portfolio_size']:.0f}"
        table_data.append(row)

    df_perf = pd.DataFrame(table_data)
    styled = df_perf.style.map(lambda v: color_value(v), subset=["총수익률", "CAGR"]).map(lambda v: color_value(v, reverse=True), subset=["MDD"]).map(lambda v: color_value(v), subset=["Sharpe"])
    st.dataframe(styled, width="stretch", hide_index=True)

    # ── 연도별 성과 ──
    section_header("연도별 성과")
    _render_yearly_performance(results, kpi_keys)

    # IS/OOS
    ref_key = next((k for k in STRATEGY_KEYS if k in results), None)
    if ref_key:
        rb_dates = results[ref_key].get("rebalance_dates", [])
        split_idx = next((i for i, d in enumerate(rb_dates) if d >= oos_start), 0)

        section_header("IS / OOS 분할")
        st.caption(f"IS: {start_str} ~ {is_end}  |  OOS: {oos_start} ~ {end_str}")
        with st.expander("IS/OOS란?", expanded=False):
            st.markdown("""
**과적합(overfitting) 검증 방법**

- **IS (In-Sample)**: 모델을 만들 때 참고한 기간. 여기서 성과가 좋은 건 당연함
- **OOS (Out-of-Sample)**: 모델이 한 번도 본 적 없는 기간. 여기서도 좋아야 '진짜' 유효한 전략
- IS에서만 좋고 OOS에서 나쁘면 → 과적합 의심 (과거에만 맞는 전략)
- 현재 전체 기간의 앞 70%를 IS, 뒤 30%를 OOS로 분할
""")
        is_oos_data = []
        for key in kpi_keys:
            r = results.get(key)
            if not r:
                continue
            rets = np.array(r["monthly_returns"])
            is_rets, oos_rets = rets[:split_idx], rets[split_idx:]
            is_cum = np.prod(1 + is_rets) - 1 if len(is_rets) > 0 else 0
            oos_cum = np.prod(1 + oos_rets) - 1 if len(oos_rets) > 0 else 0
            is_sh = (is_rets.mean() / is_rets.std() * np.sqrt(12)) if len(is_rets) > 1 and is_rets.std() > 0 else 0
            oos_sh = (oos_rets.mean() / oos_rets.std() * np.sqrt(12)) if len(oos_rets) > 1 and oos_rets.std() > 0 else 0
            is_oos_data.append({"전략": STRATEGY_LABELS[key], "IS 수익률": f"{is_cum:+.1%}", "IS Sharpe": f"{is_sh:.3f}", "OOS 수익률": f"{oos_cum:+.1%}", "OOS Sharpe": f"{oos_sh:.3f}"})

        df_isoos = pd.DataFrame(is_oos_data)
        styled_isoos = df_isoos.style.map(lambda v: color_value(v), subset=["IS 수익률", "OOS 수익률", "IS Sharpe", "OOS Sharpe"])
        st.dataframe(styled_isoos, width="stretch", hide_index=True)

    # Charts
    section_header("누적 수익률")
    st.plotly_chart(cumulative_return_chart(results), width="stretch")
    section_header("Drawdown")
    st.plotly_chart(drawdown_chart(results), width="stretch")


# ═══════════════════════════════════════════════════════
# 2. 월별 분석
# ═══════════════════════════════════════════════════════
def render_monthly():
    render_period_selector()
    universe = _render_universe_selector("monthly_universe")
    start_str, end_str, _, _ = _get_period()
    _sync_labels(universe)

    loading = st.empty()
    show_loading(loading, "월별 분석 데이터를 불러오는 중")
    all_results = load_all_results(
        start_str, end_str,
        universe=universe,
    )
    loading.empty()

    results = render_strategy_filter(all_results)

    # 채팅 컨텍스트 저장
    st.session_state["page_context"] = build_performance_context(results)

    # ── 히트맵 (클릭 시 종목별 기여도 드릴다운) ──
    heatmap_event = st.plotly_chart(
        monthly_heatmap(results), width="stretch",
        on_select="rerun", key="heatmap_select",
    )

    # 클릭 이벤트 처리
    ref_key = next((k for k in STRATEGY_KEYS if k in results), None)
    sel_points = (heatmap_event or {}).get("select", {}).get("points", [])
    if sel_points and ref_key:
        pt = sel_points[0]
        col_idx = pt.get("point_index", [None, None])
        # Heatmap point_index = [row, col]
        if isinstance(col_idx, list) and len(col_idx) == 2:
            row_idx, month_idx = col_idx
        else:
            row_idx, month_idx = None, None

        keys = [k for k in ALL_KEYS if k in results]
        rb_dates = results[ref_key]["rebalance_dates"]

        if row_idx is not None and month_idx is not None and row_idx < len(keys) and month_idx < len(rb_dates) - 1:
            sel_key = keys[row_idx]
            sel_start = rb_dates[month_idx]
            sel_end = rb_dates[month_idx + 1]
            sel_label = STRATEGY_LABELS.get(sel_key, sel_key)
            month_label = sel_start[:7]

            section_header(f"종목별 기여도: {sel_label} · {month_label}")
            attr_df = get_monthly_attribution(sel_key, sel_start, sel_end)
            if not attr_df.empty:
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.plotly_chart(
                        attribution_chart(attr_df, sel_label, month_label),
                        width="stretch",
                    )
                with c2:
                    total_ret = attr_df["기여도(%)"].sum()
                    top3 = attr_df.tail(3).iloc[::-1]
                    bot3 = attr_df.head(3)
                    st.markdown(f"**포트폴리오 수익률: {total_ret:+.1f}%**")
                    st.markdown("**상위 기여 종목**")
                    st.dataframe(
                        top3[["종목명", "섹터", "비중(%)", "종목수익률(%)", "기여도(%)"]],
                        hide_index=True, width=500,
                    )
                    st.markdown("**하위 기여 종목**")
                    st.dataframe(
                        bot3[["종목명", "섹터", "비중(%)", "종목수익률(%)", "기여도(%)"]],
                        hide_index=True, width=500,
                    )
            else:
                st.info("해당 월의 종목 데이터를 찾을 수 없습니다.")

    _bm = "KRX 300" if universe == "KOSPI+KOSDAQ" else "KOSPI 200"
    section_header(f"롤링 12개월 누적 초과수익률 vs {_bm}")
    st.plotly_chart(rolling_excess_chart(results, window=12, bm_name=_bm), width="stretch")


# ═══════════════════════════════════════════════════════
# 3. 포트폴리오 구성
# ═══════════════════════════════════════════════════════
def render_portfolio():
    universe = _render_universe_selector("portfolio_universe")
    start_str, end_str, _, _ = _get_period()
    _sync_labels(universe)

    loading = st.empty()
    show_loading(loading, "포트폴리오 데이터를 불러오는 중")
    all_results = load_all_results(
        start_str, end_str,
        universe=universe,
    )
    loading.empty()

    results = render_strategy_filter(all_results)

    strat_keys = [k for k in STRATEGY_KEYS if k in results]
    if not strat_keys:
        st.info("표시할 전략이 없습니다.")
        return

    ref_key = strat_keys[0]
    rb_dates = results[ref_key]["rebalance_dates"]
    selected_date = st.selectbox(
        "리밸런싱 날짜", rb_dates[:-1], index=len(rb_dates) - 2, key="port_date",
    )

    # 각 전략별 보유 종목 로딩
    holdings = {}
    for key in strat_keys:
        h = get_holdings(key, selected_date)
        if not h.empty:
            holdings[key] = h

    if not holdings:
        st.warning("선택된 전략의 포트폴리오 데이터를 찾을 수 없습니다.")
        return

    chars = {}
    for key in holdings:
        c = get_portfolio_characteristics(key, selected_date)
        if c:
            chars[key] = c

    # 채팅 컨텍스트 저장
    st.session_state["page_context"] = (
        f"포트폴리오 분석 ({selected_date}), "
        f"전략: {', '.join(STRATEGY_LABELS.get(k, k) for k in holdings)}"
    )

    active_keys = list(holdings.keys())

    # ── 1. 포트폴리오 특성 요약 ──
    section_header("포트폴리오 특성 요약")
    if chars:
        char_data = []
        for metric in ["PER", "PBR", "EV/EBITDA"]:
            row_w = {"지표": f"가중평균 {metric}"}
            row_s = {"지표": f"단순평균 {metric}"}
            for key in active_keys:
                c = chars.get(key, {})
                short = STRATEGY_LABELS.get(key, key).split(":")[0].strip()
                v_w = c.get(metric)
                v_s = c.get(f"{metric}_simple")
                row_w[short] = f"{v_w:.2f}" if v_w is not None else "-"
                row_s[short] = f"{v_s:.2f}" if v_s is not None else "-"
            char_data.append(row_w)
            char_data.append(row_s)
        st.dataframe(pd.DataFrame(char_data), width="stretch", hide_index=True)
        st.caption("가중평균: 포트폴리오 비중 가중 | 단순평균: 30개 종목 동일 가중")

    # ── 2. 비중 집중도 분석 ──
    section_header("비중 집중도 분석")
    with st.expander("HHI란?", expanded=False):
        st.markdown("""
- **HHI (Herfindahl-Hirschman Index)** = 각 종목 비중(%)²의 합
- 30종목 균등 배분이면 HHI ≈ 333 / 높을수록 소수 종목에 집중
- 예: 한 종목이 50%이면 HHI = 2,500+ → 극도로 집중된 포트폴리오
- **Top5 비중합**: 상위 5개 종목의 비중 합계 — 50% 이상이면 집중도 높음
""")
    kpi_cols = st.columns(min(len(active_keys), 4))
    for i, key in enumerate(active_keys):
        with kpi_cols[i % len(kpi_cols)]:
            h_df = holdings[key]
            w = h_df["비중(%)"].values
            hhi = float((w ** 2).sum())
            top5_sum = float(h_df.nlargest(5, "비중(%)")["비중(%)"].sum())
            kpi_card(
                label=STRATEGY_LABELS.get(key, key),
                value=f"HHI {hhi:.0f}",
                sub_items=[("Top5 비중합", f"{top5_sum:.1f}%")],
                color=STRATEGY_COLORS.get(key, "#757575"),
            )

    if len(holdings) >= 2:
        st.plotly_chart(concentration_chart(holdings), width="stretch")

    # ── 3. 섹터 비중 비교 ──
    section_header("섹터 비중 비교")
    st.plotly_chart(sector_comparison_chart(holdings), width="stretch")

    # ── 4. 시가총액 분포 ──
    MCAP_CATS = ["초대형", "대형", "중형", "소형"]

    def _classify_mcap(mcap):
        if mcap >= 10_000_000_000_000:
            return "초대형"
        elif mcap >= 1_000_000_000_000:
            return "대형"
        elif mcap >= 300_000_000_000:
            return "중형"
        return "소형"

    def _mcap_dist(h_df):
        if h_df.empty or "시가총액" not in h_df.columns:
            return {}
        df = h_df.copy()
        df["구분"] = df["시가총액"].apply(_classify_mcap)
        dist = df.groupby("구분")["비중(%)"].sum()
        return {c: round(float(dist.get(c, 0)), 1) for c in MCAP_CATS}

    def _mcap_count(h_df):
        if h_df.empty or "시가총액" not in h_df.columns:
            return {}
        df = h_df.copy()
        df["구분"] = df["시가총액"].apply(_classify_mcap)
        return dict(df.groupby("구분").size())

    dist_dict = {}
    for key in active_keys:
        d = _mcap_dist(holdings[key])
        if d:
            dist_dict[key] = d

    if dist_dict:
        section_header("시가총액 분포")

        mcap_cols = st.columns(min(len(active_keys), 4))
        for i, key in enumerate(active_keys):
            with mcap_cols[i % len(mcap_cols)]:
                h_df = holdings[key]
                if not h_df.empty and "시가총액" in h_df.columns:
                    w = h_df["비중(%)"].values / 100
                    avg_mcap = float((w * h_df["시가총액"].values).sum() / max(w.sum(), 1e-9))
                    avg_tril = avg_mcap / 1_000_000_000_000
                    kpi_card(
                        label=STRATEGY_LABELS.get(key, key),
                        value=f"가중평균 시총 {avg_tril:.1f}조",
                        sub_items=[("종목수", f"{len(h_df)}")],
                        color=STRATEGY_COLORS.get(key, "#757575"),
                    )

        st.plotly_chart(market_cap_distribution_chart(dist_dict), width="stretch")

        mcap_table_data = []
        for c in MCAP_CATS:
            if not any(dist_dict.get(k, {}).get(c, 0) > 0 for k in active_keys):
                continue
            row = {"구분": c}
            for key in active_keys:
                short = STRATEGY_LABELS.get(key, key).split(":")[0].strip()
                row[f"{short} 비중"] = f"{dist_dict.get(key, {}).get(c, 0):.1f}%"
                cnt = _mcap_count(holdings[key])
                row[f"{short} 종목수"] = f"{cnt.get(c, 0)}"
            mcap_table_data.append(row)
        st.dataframe(pd.DataFrame(mcap_table_data), width="stretch", hide_index=True)
        st.caption("초대형: 시총 10조+ | 대형: 1~10조 | 중형: 3000억~1조 | 소형: 3000억 미만")

    # ── 5. 리밸런싱 변화 ──
    rb_idx = rb_dates.index(selected_date) if selected_date in rb_dates else -1
    if rb_idx > 0:
        prev_date = rb_dates[rb_idx - 1]
        section_header("리밸런싱 변화")
        st.caption(f"비교: {prev_date} → {selected_date}")

        turn_cols = st.columns(min(len(active_keys), 4))
        for i, key in enumerate(active_keys):
            with turn_cols[i % len(turn_cols)]:
                st.markdown(f"**{STRATEGY_LABELS.get(key, key)}**")
                to = get_portfolio_turnover(key, selected_date, prev_date)
                tc1, tc2, tc3, tc4 = st.columns(4)
                with tc1:
                    st.metric("신규 편입", f"{to['added_count']}")
                with tc2:
                    st.metric("편출", f"{to['removed_count']}")
                with tc3:
                    st.metric("유지", f"{to['retained_count']}")
                with tc4:
                    st.metric("회전율", f"{to['turnover_rate']:.0%}")
                if not to["added"].empty:
                    st.markdown("_신규 편입_")
                    st.dataframe(to["added"].sort_values("비중(%)", ascending=False),
                                 width="stretch", hide_index=True,
                                 height=min(len(to["added"]) * 35 + 40, 250))
                if not to["removed"].empty:
                    st.markdown("_편출_")
                    st.dataframe(to["removed"].sort_values("비중(%)", ascending=False),
                                 width="stretch", hide_index=True,
                                 height=min(len(to["removed"]) * 35 + 40, 250))

    # ── 6. 종목 상세 ──
    section_header("종목 상세")
    for key in active_keys:
        h_df = holdings[key]
        display_cols = [c for c in ["종목코드", "종목명", "섹터", "비중(%)", "점수",
                                     "PER", "PBR", "EV/EBITDA"] if c in h_df.columns]
        st.subheader(f"{STRATEGY_LABELS.get(key, key)} ({len(h_df)}종목)")
        st.dataframe(
            h_df[display_cols].sort_values("비중(%)", ascending=False),
            width="stretch", hide_index=True,
            height=min(len(h_df) * 35 + 40, 400),
        )


# ═══════════════════════════════════════════════════════
# 4. 통계 검증
# ═══════════════════════════════════════════════════════
def render_statistics():
    render_period_selector()
    universe = _render_universe_selector("statistics_universe")
    start_str, end_str, is_end, oos_start = _get_period()
    _sync_labels(universe)

    all_results = load_all_results(
        start_str, end_str,
        universe=universe,
    )
    results = render_strategy_filter(all_results)
    selected_keys = [k for k in STRATEGY_KEYS if k in results]

    loading = st.empty()
    show_loading(loading, "통계 검증 결과를 불러오는 중")
    is_oos_data, stat_data, rolling_all = load_all_robustness_results(
        start_str, end_str, is_end, oos_start,
        universe=universe,
    )
    loading.empty()

    # 채팅 컨텍스트 저장
    st.session_state["page_context"] = build_stat_context(stat_data, rolling_all)

    section_header("1. In-Sample vs Out-of-Sample")
    st.caption(f"IS: {start_str} ~ {is_end}  |  OOS: {oos_start} ~ {end_str}")
    with st.expander("IS/OOS란?", expanded=False):
        st.markdown("""
- **IS (In-Sample)**: 전체 기간의 앞 70%. 모델이 학습에 참고한 기간 → 성과가 좋은 게 당연
- **OOS (Out-of-Sample)**: 뒤 30%. 모델이 본 적 없는 기간 → 여기서도 좋아야 진짜
- IS와 OOS 성과가 비슷하면 과적합 위험이 낮음. IS만 좋으면 과거에만 맞는 전략일 수 있음
""")
    st.plotly_chart(is_oos_comparison_chart(is_oos_data, keys=selected_keys), width="stretch")

    # IS/OOS table
    is_oos_table = []
    for key in selected_keys:
        is_r = is_oos_data["is_results"].get(key, {})
        oos_r = is_oos_data["oos_results"].get(key, {})
        if not is_r and not oos_r:
            continue
        is_oos_table.append({
            "전략": STRATEGY_LABELS[key],
            "IS 수익률": f"{is_r.get('total_return', 0):+.1%}", "IS CAGR": f"{is_r.get('cagr', 0):+.1%}",
            "IS Sharpe": f"{is_r.get('sharpe', 0):.3f}", "IS MDD": f"{is_r.get('mdd', 0):.1%}",
            "OOS 수익률": f"{oos_r.get('total_return', 0):+.1%}", "OOS CAGR": f"{oos_r.get('cagr', 0):+.1%}",
            "OOS Sharpe": f"{oos_r.get('sharpe', 0):.3f}", "OOS MDD": f"{oos_r.get('mdd', 0):.1%}",
        })
    bm = is_oos_data.get("benchmarks", {})
    is_bm = bm.get("is", {}).get("KOSPI", {})
    oos_bm = bm.get("oos", {}).get("KOSPI", {})
    if is_bm or oos_bm:
        is_oos_table.append({"전략": STRATEGY_LABELS.get("KOSPI", "KOSPI 200"),
            "IS 수익률": f"{is_bm.get('total_return', 0):+.1%}", "IS CAGR": f"{is_bm.get('cagr', 0):+.1%}", "IS Sharpe": "-", "IS MDD": "-",
            "OOS 수익률": f"{oos_bm.get('total_return', 0):+.1%}", "OOS CAGR": f"{oos_bm.get('cagr', 0):+.1%}", "OOS Sharpe": "-", "OOS MDD": "-"})
    if is_oos_table:
        df = pd.DataFrame(is_oos_table)
        styled = df.style.map(lambda v: color_value(v), subset=["IS 수익률", "IS CAGR", "OOS 수익률", "OOS CAGR"]).map(lambda v: color_value(v, reverse=True), subset=["IS MDD", "OOS MDD"])
        st.dataframe(styled, width="stretch", hide_index=True)

    # Bootstrap
    section_header("2. 통계적 유의성 (Bootstrap 10,000회 + t-test)")
    st.plotly_chart(bootstrap_histogram(stat_data, keys=selected_keys), width="stretch")
    sig_table = []
    for key in selected_keys:
        sig = stat_data["bm_significance"].get(key)
        if not sig:
            continue
        sig_table.append({
            "전략": STRATEGY_LABELS[key], "월평균 초과수익": f"{sig['mean_diff']*100:+.3f}%",
            "t-stat": f"{sig['t_stat']:.2f}", "p-value": f"{sig['p_value']:.4f}",
            "95% CI": f"[{sig['ci_lower']*100:+.3f}%, {sig['ci_upper']*100:+.3f}%]",
            "Bootstrap 승률": f"{sig['win_rate']:.1%}",
            "유의 여부": "유의 (p<0.05)" if sig['significant'] else "유의하지 않음",
        })
    if sig_table:
        df_sig = pd.DataFrame(sig_table)
        styled_sig = df_sig.style.map(lambda v: color_value(v), subset=["월평균 초과수익"]).map(
            lambda v: "color: #4CAF50; font-weight: 600" if v == "유의 (p<0.05)" else "color: #EF5350" if "유의하지" in str(v) else "", subset=["유의 여부"])
        st.dataframe(styled_sig, width="stretch", hide_index=True)

    # Rolling
    section_header("3. 롤링 24개월 윈도우")
    with st.expander("롤링 윈도우란?", expanded=False):
        st.markdown("""
- 24개월 윈도우를 한 달씩 이동하며 KOSPI 200 대비 초과수익을 계산
- **승률** = 초과수익이 양수인 구간의 비율 (전략 vs KOSPI 200 비교이며, 전략 간 비교가 아님)
- 승률 50% 초과 → 과반 이상의 기간에서 벤치마크를 이기고 있다는 의미
- 승률이 높을수록 전략이 다양한 시장 환경에서 안정적으로 작동
""")
    st.plotly_chart(rolling_window_chart(rolling_all, keys=selected_keys), width="stretch")
    rolling_table = []
    for key in selected_keys:
        rd = rolling_all.get(key)
        if rd:
            rolling_table.append({"전략": STRATEGY_LABELS[key], "총 윈도우": rd["total_windows"], "양의 알파": rd["positive_windows"], "승률": f"{rd['win_rate']:.0%}"})
    if rolling_table:
        st.dataframe(pd.DataFrame(rolling_table), width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════
# 5. 전략 실험실 (콘텐츠만 — 채팅은 왼쪽 패널에서 처리)
# ═══════════════════════════════════════════════════════

_CAT_COLORS = {
    "밸류에이션": "#2196F3", "회귀 매력도": "#E91E63",
    "성장성": "#4CAF50",     "차별화": "#FF9800",
    "커스텀": "#9C27B0",
}


def _build_weights_html(
    weights_data: list,
    large_only_set: set | None = None,
    small_only_set: set | None = None,
) -> str:
    """가중치 데이터(format_weights_for_display 형식)를 HTML로 변환 (대형주 전용)."""
    if not weights_data:
        return ""

    cat_totals = {}
    cat_groups = {}
    for row in weights_data:
        cat = row["카테고리"]
        cat_totals[cat] = cat_totals.get(cat, 0) + row["_wl"]
        cat_groups.setdefault(cat, []).append(row)

    total = sum(cat_totals.values())
    max_w = max((r["_wl"] for r in weights_data), default=0.01)

    # ── 스택드 오버뷰 바 ──
    segs = ""
    for cat, val in cat_totals.items():
        if val <= 0:
            continue
        c = _CAT_COLORS.get(cat, "#999")
        segs += (
            f'<div class="wv-seg" style="flex:{val*100}; background:{c};">'
            f'<span class="wv-seg-label">{cat}</span>'
            f'<span class="wv-seg-pct">{val*100:.0f}%</span>'
            f'</div>'
        )
    html = f'<div class="wv-overview">{segs}</div>'

    # ── 카테고리 카드 ──
    for cat, rows in cat_groups.items():
        c = _CAT_COLORS.get(cat, "#999")
        ct = cat_totals.get(cat, 0)

        factors_html = ""
        for row in rows:
            wl = row["_wl"]
            wl_w = (wl / max_w) * 100 if wl > 0 else 0

            factors_html += (
                f'<div class="wv-factor">'
                f'  <span class="wv-fname">{row["팩터"]}</span>'
                f'  <div class="wv-fbars">'
                f'    <div class="wv-fbar">'
                f'      <div class="wv-fbar-track"><div class="wv-fbar-fill" style="width:{wl_w}%;background:{c};"></div></div>'
                f'      <span class="wv-fbar-val">{row["비중"] if "비중" in row else row.get("대형주", "-")}</span>'
                f'    </div>'
                f'  </div>'
                f'</div>'
            )

        html += (
            f'<div class="wv-card" style="border-left:3px solid {c};">'
            f'  <div class="wv-card-head">'
            f'    <span class="wv-card-title">{cat}</span>'
            f'    <span class="wv-card-pct" style="color:{c};">{ct*100:.0f}%</span>'
            f'  </div>'
            f'  {factors_html}'
            f'</div>'
        )

    html += f'<div class="wv-total">전체 합계 {total*100:.0f}%</div>'
    return html


def _extract_strategy_info(strategy_code: str) -> dict:
    """전략 코드에서 PARAMS, WEIGHTS_LARGE를 추출."""
    try:
        ns = {}
        exec(compile(strategy_code, "<strategy>", "exec"), {"__builtins__": {}}, ns)
        return {
            "params": ns.get("PARAMS", {}),
            "weights_large": ns.get("WEIGHTS_LARGE", {}),
            "weights_small": ns.get("WEIGHTS_SMALL", {}),
            "regression_models": ns.get("REGRESSION_MODELS", []),
        }
    except Exception:
        return {"params": {}, "weights_large": {}, "weights_small": {}, "regression_models": []}


def _render_weight_editor_inline(strategy_code: str, weights_data: list):
    """Dev 모드: 가중치 테이블에서 직접 % 수정. 바 + number_input 인라인."""
    if not weights_data:
        return

    cat_groups = {}
    for row in weights_data:
        cat_groups.setdefault(row["카테고리"], []).append(row)

    max_w = max((r["_wl"] for r in weights_data), default=0.01)
    edited_weights = {}

    for cat, rows in cat_groups.items():
        color = _CAT_COLORS.get(cat, "#999")
        cat_total = sum(r["_wl"] for r in rows)

        # 카테고리 헤더
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'border-left:3px solid {color};padding:4px 8px;margin-top:12px;">'
            f'<b>{cat}</b>'
            f'<span style="color:{color};font-weight:700;font-size:1.1rem;">{cat_total*100:.0f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        for row in rows:
            wl_w = (row["_wl"] / max_w) * 100 if row["_wl"] > 0 else 0
            c1, c2, c3 = st.columns([3, 5, 1.5])
            with c1:
                st.markdown(
                    f'<div style="padding:8px 0;font-size:0.85rem;">{row["팩터"]}</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div style="padding:10px 0;">'
                    f'<div style="background:#333;border-radius:4px;height:8px;width:100%;">'
                    f'<div style="background:{color};border-radius:4px;height:8px;width:{wl_w}%;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                val = st.number_input(
                    row["팩터"], min_value=0, max_value=50, step=1,
                    value=int(round(row["_wl"] * 100)),
                    key=f"wei_{row['_code']}",
                    label_visibility="collapsed",
                )
                edited_weights[row["_code"]] = val / 100

    # 합계
    total_pct = sum(edited_weights.values()) * 100
    if abs(total_pct - 100) < 0.5:
        st.success(f"합계: {total_pct:.0f}%")
    else:
        st.warning(f"합계: {total_pct:.0f}% (100%가 되어야 합니다)")

    # 적용 버튼
    if st.button("가중치 적용", key="btn_apply_weights_inline", type="primary", use_container_width=True):
        if abs(total_pct - 100) >= 1:
            st.error(f"합계가 {total_pct:.0f}%입니다. 100%로 맞춰주세요.")
        else:
            new_code = _update_weights_in_code(strategy_code, edited_weights)
            st.session_state.lab_strategy_code = new_code
            st.session_state.lab_modified_results = None
            st.rerun()


def _render_weight_editor(strategy_code: str):
    """Dev 모드: 팩터 가중치를 직접 숫자로 편집하는 UI (별도 탭용, 미사용)."""
    weights_data = format_weights_for_display(strategy_code)
    if not weights_data:
        st.info("가중치 데이터를 추출할 수 없습니다.")
        return

    st.caption("팩터별 비중(%)을 직접 수정하세요. 합계가 100%가 되어야 합니다.")

    # 카테고리별 그룹핑
    cat_groups = {}
    for row in weights_data:
        cat_groups.setdefault(row["카테고리"], []).append(row)

    edited_weights = {}
    for cat, rows in cat_groups.items():
        color = _CAT_COLORS.get(cat, "#999")
        st.markdown(f"**<span style='color:{color}'>{cat}</span>**", unsafe_allow_html=True)
        cols = st.columns(min(len(rows), 4))
        for i, row in enumerate(rows):
            with cols[i % min(len(rows), 4)]:
                val = st.number_input(
                    row["팩터"],
                    min_value=0, max_value=50, step=1,
                    value=int(round(row["_wl"] * 100)),
                    key=f"we_{row['_code']}",
                )
                edited_weights[row["_code"]] = val / 100

    # 합계 표시
    total = sum(edited_weights.values())
    total_pct = total * 100
    if abs(total_pct - 100) < 0.5:
        st.success(f"합계: {total_pct:.0f}%")
    else:
        st.warning(f"합계: {total_pct:.0f}% (100%가 되어야 합니다)")

    # 적용 버튼
    if st.button("가중치 적용", key="btn_apply_weights", type="primary", use_container_width=True):
        if abs(total_pct - 100) >= 1:
            st.error(f"합계가 {total_pct:.0f}%입니다. 100%로 맞춰주세요.")
        else:
            # 현재 전략 코드에서 WEIGHTS_LARGE만 교체
            new_code = _update_weights_in_code(strategy_code, edited_weights)
            st.session_state.lab_strategy_code = new_code
            st.session_state.lab_modified_results = None
            st.rerun()


def _update_weights_in_code(strategy_code: str, new_weights: dict) -> str:
    """전략 코드의 WEIGHTS_LARGE를 새 가중치로 교체."""
    import re
    # WEIGHTS_LARGE = { ... } 패턴을 찾아서 교체
    new_dict_str = "WEIGHTS_LARGE = {\n"
    for key, val in new_weights.items():
        if val > 0:
            new_dict_str += f'    "{key}": {val:.2f},\n'
    new_dict_str += "}"

    # 기존 WEIGHTS_LARGE 블록을 교체
    pattern = r'WEIGHTS_LARGE\s*=\s*\{[^}]*\}'
    if re.search(pattern, strategy_code):
        return re.sub(pattern, new_dict_str, strategy_code)
    else:
        return strategy_code


def render_lab_content():
    """전략 실험실 콘텐츠 (우측 패널). 채팅은 render_chat_column에서 처리."""

    # Session state init
    if "lab_strategy_code" not in st.session_state:
        st.session_state.lab_strategy_code = DEFAULT_STRATEGY_CODE
    if "lab_messages" not in st.session_state:
        st.session_state.lab_messages = []
    if "lab_modified_results" not in st.session_state:
        st.session_state.lab_modified_results = None

    current_code = st.session_state.lab_strategy_code
    is_modified = current_code != DEFAULT_STRATEGY_CODE
    info = _extract_strategy_info(current_code)
    params = info["params"]

    # 초기 진입 시에만 기본값 설정 (저장된 전략 로드는 _on_strat_select에서 처리)
    if "param_weight_cap" not in st.session_state:
        st.session_state.param_weight_cap = params.get("weight_cap_pct", BACKTEST_CONFIG.get("weight_cap_pct", 10))
    if "param_top_n" not in st.session_state:
        st.session_state.param_top_n = params.get("top_n", BACKTEST_CONFIG.get("top_n_stocks", 30))

    # ─── 상단 바: 저장 전략 선택 / 삭제 / 초기화 ───
    strategies = list_strategies()

    if strategies:
        top_cols = st.columns([5, 1, 2])

        with top_cols[0]:
            strat_names = [s["name"] for s in strategies]
            strat_display = []
            for s in strategies:
                label = s["name"]
                if s.get("summary"):
                    label += f"  ({s['summary'].get('CAGR', '')}, Sharpe {s['summary'].get('Sharpe', '')})"
                strat_display.append(label)

            def _on_strat_select():
                idx = st.session_state.strat_load_select
                data = load_strategy(strat_names[idx])
                if data and "code" in data:
                    st.session_state.lab_strategy_code = data["code"]
                    st.session_state.lab_modified_results = data.get("results")
                    st.session_state.lab_messages = []
                    # 전략의 파라미터를 UI에 반영
                    _si = _extract_strategy_info(data["code"])
                    if _si and _si.get("params"):
                        _sp = _si["params"]
                        if "weight_cap_pct" in _sp:
                            st.session_state.param_weight_cap = _sp["weight_cap_pct"]
                        if "top_n" in _sp:
                            st.session_state.param_top_n = _sp["top_n"]

            selected_idx = st.selectbox(
                "저장된 전략", range(len(strat_names)),
                format_func=lambda i: strat_display[i],
                key="strat_load_select", label_visibility="collapsed",
                on_change=_on_strat_select,
            )
        with top_cols[1]:
            if st.button("삭제", key="btn_delete", use_container_width=True):
                delete_strategy(strat_names[selected_idx])
                st.rerun()
        with top_cols[2]:
            if st.button("기본 전략으로 초기화", key="btn_reset", use_container_width=True):
                st.session_state.lab_strategy_code = DEFAULT_STRATEGY_CODE
                st.session_state.lab_messages = []
                st.session_state.lab_modified_results = None
                st.rerun()
    else:
        if is_modified:
            if st.button("기본 전략으로 초기화", key="btn_reset"):
                st.session_state.lab_strategy_code = DEFAULT_STRATEGY_CODE
                st.session_state.lab_messages = []
                st.session_state.lab_modified_results = None
                st.rerun()

    st.markdown("---")

    # ═══════════════════════════════════════════════════════
    # 통합 전략 설정 카드 (Dev 모드에서만 편집 가능)
    # ═══════════════════════════════════════════════════════
    section_header("전략 설정")
    _ap_lab = get_active_params()
    _sync_labels(_ap_lab["universe"], _ap_lab["weight_cap_pct"])

    _init_params()
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.selectbox(
            "유니버스",
            options=["KOSPI", "KOSPI+KOSDAQ"],
            key="param_universe",
            help="KOSPI → 벤치마크 KODEX200 / KOSPI+KOSDAQ → 벤치마크 KRX300",
        )
    with pc2:
        st.number_input(
            "비중캡 (%)", min_value=0, max_value=30, step=1,
            key="param_weight_cap",
            help="0% = 캡 없음 (시총비례), 10% = 개별종목 최대 10%",
        )
    with pc3:
        st.number_input(
            "편입 종목수", min_value=10, max_value=100, step=5,
            value=_ap_lab.get("top_n", 30),
            key="param_top_n",
            help="리밸런싱 시 편입할 상위 종목 수",
        )

    if IS_DEV:
        # ── 팩터 가중치 편집 ──
        st.markdown("")
        weights_data = format_weights_for_display(current_code)
        if weights_data:
            _render_weight_editor_inline(current_code, weights_data)
    else:
        # Production 모드: 읽기 전용 요약
        _view_code = current_code
        _view_info = _extract_strategy_info(_view_code) if _view_code else None
        _view_params = _view_info["params"] if _view_info else {}
        strat_summary = extract_strategy_summary(_view_code) if _view_code else {}
        scoring_mode = strat_summary.get("scoring_mode", {})
        large_mode_label = "사분위(0-4점)" if scoring_mode.get("large") == "quartile" else "십분위(0-10점)"

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("유니버스", "KOSPI")
        with col2:
            st.metric("비중상한", f"{_view_params.get('weight_cap_pct', 10)}%")
        with col3:
            st.metric("편입 종목수", f"{_view_params.get('top_n', 30)}개")
        with col4:
            n_l = strat_summary.get("n_factors_large", 0)
            st.metric("팩터 수", f"{n_l}개")
        with col5:
            st.metric("채점방식", large_mode_label)

        # 읽기 전용 가중치 테이블
        if _view_code:
            weights_data = format_weights_for_display(_view_code)
            if weights_data:
                large_only_set = set(strat_summary.get("large_only", []))
                small_only_set = set(strat_summary.get("small_only", []))
                with st.expander("팩터 가중치", expanded=False):
                    html = _build_weights_html(weights_data, large_only_set, small_only_set)
                    st.markdown(html, unsafe_allow_html=True)

    # ── 전략 선택 필터 (비교용) ──
    st.markdown("---")
    _VIEW_EXPERIMENT = "__experiment__"
    view_options = [
        ("A0", STRATEGY_LABELS["A0"]),
    ]
    if is_modified:
        view_options.append((_VIEW_EXPERIMENT, "수정 전략"))
    for s in strategies:
        view_options.append((s["name"], s["name"]))

    view_keys = [k for k, _ in view_options]
    view_labels = [v for _, v in view_options]

    if is_modified and st.session_state.get("lab_view_strategy") != _VIEW_EXPERIMENT:
        st.session_state.lab_view_strategy = _VIEW_EXPERIMENT
    if not is_modified and st.session_state.get("lab_view_strategy") == _VIEW_EXPERIMENT:
        st.session_state.lab_view_strategy = "A0"

    # ── 가중치 차트 & 전략 코드 (접기) ──
    _view_code_sel = current_code if (is_modified or not strategies) else None
    if _view_code_sel:
        _vi = _extract_strategy_info(_view_code_sel)
        with st.expander("가중치 차트 / 전략 코드", expanded=False):
            tc1, tc2 = st.tabs(["가중치 차트", "전략 코드"])
            with tc1:
                if _vi:
                    st.plotly_chart(strategy_weight_chart(_vi["weights_large"]), width="stretch")
            with tc2:
                st.code(_view_code_sel, language="python")

    # ─── 2. 백테스트 ───
    st.markdown("---")
    section_header("백테스트")
    _bm_lab = "KRX 300" if get_active_params().get("universe") == "KOSPI+KOSDAQ" else "KOSPI 200"
    st.caption(f"저장하지 않아도 바로 실행할 수 있습니다. 수정 전략 vs 기존전략 vs {_bm_lab} 비교.")

    if not st.session_state.lab_modified_results:
        if st.button("백테스트 실행", type="primary", key="btn_backtest", use_container_width=True):
            progress_bar = st.progress(0, text="백테스트 실행 중...")

            def progress_callback(current, total):
                pct = current / max(total, 1)
                progress_bar.progress(pct, text=f"백테스트 실행 중... ({current}/{total})")

            try:
                _ap = get_active_params()
                modified = run_strategy_backtest(
                    current_code,
                    progress_callback=progress_callback,
                    universe=_ap.get("universe"),
                    weight_cap_pct_override=_ap.get("weight_cap_pct"),
                )
                progress_bar.empty()
                if modified and "error" in modified:
                    st.error(f"전략 코드 오류: {modified['error']}")
                else:
                    st.session_state.lab_modified_results = modified
                    st.rerun()
            except Exception as e:
                progress_bar.empty()
                st.error(f"백테스트 실행 오류: {e}")
    else:
        # 기존전략(A0)은 디폴트 파라미터로 로딩
        original = load_backtest_results()
        modified = st.session_state.lab_modified_results

        # 벤치마크는 modified 결과(유니버스·캡 반영됨)를 우선 사용
        _bm_source = modified if "KOSPI" in modified else original

        comp_data = []
        _mod_cap = get_active_params().get("weight_cap_pct", 10)
        _mod_cap_label = f" ({_mod_cap}%캡)" if _mod_cap > 0 else " (캡없음)"
        m = modified.get("CUSTOM", {})
        if m:
            comp_data.append({
                "전략": f"수정전략{_mod_cap_label}",
                "총수익률": f"{m.get('total_return', 0):+.1%}",
                "CAGR": f"{m.get('cagr', 0):+.1%}",
                "MDD": f"{m.get('mdd', 0):.1%}",
                "Sharpe": f"{m.get('sharpe', 0):.3f}",
            })
        # 기존전략은 original에서, 벤치마크는 _bm_source에서
        for key, source in [("A0", original), ("KOSPI", _bm_source)]:
            r = source.get(key, {})
            if r:
                # 벤치마크는 결과 데이터의 strategy 필드 우선 (유니버스 반영)
                if key == "KOSPI":
                    label = r.get("strategy", STRATEGY_LABELS.get(key, key))
                else:
                    label = STRATEGY_LABELS.get(key) or r.get("strategy", key)
                comp_data.append({
                    "전략": label,
                    "총수익률": f"{r.get('total_return', 0):+.1%}",
                    "CAGR": f"{r.get('cagr', 0):+.1%}",
                    "MDD": f"{r.get('mdd', 0):.1%}",
                    "Sharpe": f"{r.get('sharpe', 0):.3f}",
                })

        df_comp = pd.DataFrame(comp_data)
        styled_comp = df_comp.style.map(
            lambda v: color_value(v), subset=["총수익률", "CAGR", "Sharpe"]
        ).map(
            lambda v: color_value(v, reverse=True), subset=["MDD"]
        )
        st.dataframe(styled_comp, width="stretch", hide_index=True)

        # 누적 수익률 차트
        st.plotly_chart(comparison_cumulative_chart(original, modified, mod_label=f"수정전략{_mod_cap_label}"), width="stretch")

        if st.button("전략 수정 후 재실행", key="btn_rerun", use_container_width=True):
            st.session_state.lab_modified_results = None
            st.rerun()

    # ─── 3. 전략 저장 ───
    st.markdown("---")
    section_header("전략 저장")
    st.caption("백테스트 결과가 있으면 저장 후 성과 비교, 월별 분석 등 다른 탭에서도 표시됩니다.")

    save_cols = st.columns([3, 3, 1])
    with save_cols[0]:
        save_name = st.text_input("전략 이름", key="strat_save_name", label_visibility="collapsed",
                                   placeholder="전략 이름")
    with save_cols[1]:
        save_desc = st.text_input("설명", key="strat_save_desc", label_visibility="collapsed",
                                   placeholder="설명 (선택)")
    with save_cols[2]:
        if st.button("저장", key="btn_save", use_container_width=True):
            if save_name:
                custom_results = (st.session_state.lab_modified_results or {}).get("CUSTOM")
                save_strategy(
                    name=save_name,
                    code=current_code,
                    description=save_desc,
                    results=custom_results,
                )
                if custom_results:
                    st.success(f"'{save_name}' 저장 완료 — 다른 탭에서도 표시됩니다.")
                else:
                    st.success(f"'{save_name}' 저장 완료 — 백테스트를 실행하면 다른 탭에도 표시됩니다.")
            else:
                st.warning("전략 이름을 입력하세요.")
