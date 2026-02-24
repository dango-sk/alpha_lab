"""
대시보드 뷰 함수 — 각 탭의 콘텐츠를 렌더링
"""
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

from lib.data import (
    load_backtest_results, load_all_results, load_robustness_results,
    load_all_robustness_results,
    get_stock_comparison, get_holdings, get_monthly_attribution,
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
from lib.chat import build_performance_context, build_stat_context, build_portfolio_context

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
                min_value=date(2019, 1, 1), max_value=st.session_state.bt_end,
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


def render_strategy_filter(results: dict) -> dict:
    """전략 선택 필터 UI (멀티셀렉트). 선택된 전략만 포함된 results를 반환.

    위젯 key 기반으로 탭 간 선택 상태를 자동 공유한다.
    """
    available = [k for k in ALL_KEYS if k in results]
    labels = {k: STRATEGY_LABELS.get(k, k) for k in available}

    widget_key = "strategy_filter"
    tracking_key = "_available_strategies"

    # 새로 추가된 전략 감지 → 자동 선택에 포함
    prev_available = set(st.session_state.get(tracking_key, []))
    new_strategies = set(available) - prev_available
    st.session_state[tracking_key] = list(available)

    if widget_key in st.session_state:
        current = list(st.session_state[widget_key])
        # 더 이상 없는 전략 제거
        current = [k for k in current if k in available]
        # 새로 추가된 전략 자동 포함
        for k in available:
            if k in new_strategies and k not in current:
                current.append(k)
        st.session_state[widget_key] = current

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
# 1. 전략 성과 비교
# ═══════════════════════════════════════════════════════
def render_performance():
    render_period_selector()

    start_str, end_str, is_end, oos_start = _get_period()

    st.caption(
        f"기간: {start_str} ~ {end_str}  |  "
        f"리밸런싱: 월 1회, 상위 {BACKTEST_CONFIG['top_n_stocks']}종목  |  "
        f"비중: 시총비례 + 15% 캡  |  거래비용: 편도 {BACKTEST_CONFIG['transaction_cost_bp']}bp"
    )

    loading = st.empty()
    show_loading(loading, "백테스트 결과를 불러오는 중")
    all_results = load_all_results(start_str, end_str)
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
                value=f"{r['cagr']:+.1%}",
                sub_items=[("MDD", f"{r['mdd']:.1%}"), ("Sharpe", f"{r['sharpe']:.3f}")],
                color=STRATEGY_COLORS.get(key, "#757575"),
            )
    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison Table
    section_header("성과 비교")
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

    # IS/OOS
    ref_key = next((k for k in STRATEGY_KEYS if k in results), None)
    if ref_key:
        rb_dates = results[ref_key].get("rebalance_dates", [])
        split_idx = next((i for i, d in enumerate(rb_dates) if d >= oos_start), 0)

        section_header("IS / OOS 분할")
        st.caption(f"IS: {start_str} ~ {is_end}  |  OOS: {oos_start} ~ {end_str}")
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
    start_str, end_str, _, _ = _get_period()

    loading = st.empty()
    show_loading(loading, "월별 분석 데이터를 불러오는 중")
    all_results = load_all_results(start_str, end_str)
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

    section_header("롤링 12개월 누적 초과수익률 vs KOSPI 200")
    st.plotly_chart(rolling_excess_chart(results, window=12), width="stretch")


# ═══════════════════════════════════════════════════════
# 3. 포트폴리오 구성
# ═══════════════════════════════════════════════════════
def render_portfolio():
    start_str, end_str, _, _ = _get_period()

    loading = st.empty()
    show_loading(loading, "포트폴리오 데이터를 불러오는 중")
    all_results = load_all_results(start_str, end_str)
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
    if "A0" in holdings and "ATT2" in holdings:
        common, a0_only, att2_only = get_stock_comparison(selected_date)
        st.session_state["page_context"] = build_portfolio_context(
            common, a0_only, att2_only, selected_date,
            a0_chars=chars.get("A0", {}), att2_chars=chars.get("ATT2", {}),
        )
    else:
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
            row = {"지표": f"가중평균 {metric}"}
            for key in active_keys:
                v = chars.get(key, {}).get(metric)
                short = STRATEGY_LABELS.get(key, key).split(":")[0].strip()
                row[short] = f"{v:.2f}" if v is not None else "-"
            char_data.append(row)
        st.dataframe(pd.DataFrame(char_data), width="stretch", hide_index=True)
        st.caption("각 종목의 포트폴리오 비중을 가중치로 사용한 가중평균")

    # ── 2. 비중 집중도 분석 ──
    section_header("비중 집중도 분석")
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
    start_str, end_str, is_end, oos_start = _get_period()

    all_results = load_all_results(start_str, end_str)
    results = render_strategy_filter(all_results)
    selected_keys = [k for k in STRATEGY_KEYS if k in results]

    loading = st.empty()
    show_loading(loading, "통계 검증 결과를 불러오는 중")
    is_oos_data, stat_data, rolling_all = load_all_robustness_results(
        start_str, end_str, is_end, oos_start,
    )
    loading.empty()

    # 채팅 컨텍스트 저장
    st.session_state["page_context"] = build_stat_context(stat_data, rolling_all)

    section_header("1. In-Sample vs Out-of-Sample")
    st.caption(f"IS: {start_str} ~ {is_end}  |  OOS: {oos_start} ~ {end_str}")
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
    """가중치 데이터(format_weights_for_display 형식)를 HTML로 변환."""
    if not weights_data:
        return ""

    large_only_set = large_only_set or set()
    small_only_set = small_only_set or set()

    cat_totals = {}
    cat_groups = {}
    for row in weights_data:
        cat = row["카테고리"]
        cat_totals[cat] = cat_totals.get(cat, 0) + row["_wl"]
        cat_groups.setdefault(cat, []).append(row)

    total = sum(cat_totals.values())
    max_w = max(
        max((r["_wl"] for r in weights_data), default=0.01),
        max((r["_ws"] for r in weights_data), default=0.01),
    )

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

    # ── 범례 ──
    html += (
        '<div class="wv-legend">'
        '<span><span class="wv-legend-dot" style="background:#fff; opacity:0.85;"></span>대형주</span>'
        '<span><span class="wv-legend-dot" style="background:#fff; opacity:0.35;"></span>중소형주</span>'
        '</div>'
    )

    # ── 카테고리 카드 ──
    for cat, rows in cat_groups.items():
        c = _CAT_COLORS.get(cat, "#999")
        ct = cat_totals.get(cat, 0)

        factors_html = ""
        for row in rows:
            wl, ws = row["_wl"], row["_ws"]
            wl_w = (wl / max_w) * 100 if wl > 0 else 0
            ws_w = (ws / max_w) * 100 if ws > 0 else 0

            badge = ""
            code = row["_code"]
            if code in large_only_set:
                badge = '<span class="wv-badge" style="background:rgba(33,150,243,0.15); color:#64B5F6;">대형주</span>'
            elif code in small_only_set:
                badge = '<span class="wv-badge" style="background:rgba(255,152,0,0.15); color:#FFB74D;">중소형주</span>'

            factors_html += (
                f'<div class="wv-factor">'
                f'  <span class="wv-fname">{row["팩터"]}{badge}</span>'
                f'  <div class="wv-fbars">'
                f'    <div class="wv-fbar">'
                f'      <div class="wv-fbar-track"><div class="wv-fbar-fill" style="width:{wl_w}%;background:{c};"></div></div>'
                f'      <span class="wv-fbar-val">{row["대형주"]}</span>'
                f'    </div>'
                f'    <div class="wv-fbar">'
                f'      <div class="wv-fbar-track"><div class="wv-fbar-fill" style="width:{ws_w}%;background:{c};opacity:0.4;"></div></div>'
                f'      <span class="wv-fbar-val" style="opacity:0.55;">{row["중소형주"]}</span>'
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

    # ─── 1. 전략 구성 ───
    section_header("전략 구성")

    # 전략 선택 필터 구성
    _VIEW_EXPERIMENT = "__experiment__"
    view_options = [
        ("A0", STRATEGY_LABELS["A0"]),
        ("ATT2", STRATEGY_LABELS["ATT2"]),
    ]
    if is_modified:
        view_options.append((_VIEW_EXPERIMENT, "수정 전략"))
    for s in strategies:
        view_options.append((s["name"], s["name"]))

    view_keys = [k for k, _ in view_options]
    view_labels = [v for _, v in view_options]

    # 수정 발생 시 자동으로 수정 전략 선택
    if is_modified and st.session_state.get("lab_view_strategy") != _VIEW_EXPERIMENT:
        st.session_state.lab_view_strategy = _VIEW_EXPERIMENT
    # 수정 전략이 사라졌는데 선택값이 남아있으면 A0로 복귀
    if not is_modified and st.session_state.get("lab_view_strategy") == _VIEW_EXPERIMENT:
        st.session_state.lab_view_strategy = "A0"

    selected_view = st.selectbox(
        "전략 선택", view_keys,
        format_func=lambda k: view_labels[view_keys.index(k)],
        key="lab_view_strategy",
    )

    # 선택된 전략의 코드 · 정보 준비
    if selected_view == _VIEW_EXPERIMENT:
        _view_code = current_code
    elif selected_view in BASE_STRATEGY_WEIGHTS:
        # 기존 전략 → 가상 코드 문자열 생성 (format_weights_for_display 호환)
        bw = BASE_STRATEGY_WEIGHTS[selected_view]
        _view_code = (
            f"WEIGHTS_LARGE = {bw['weights_large']!r}\n"
            f"WEIGHTS_SMALL = {bw['weights_small']!r}\n"
            f"REGRESSION_MODELS = {bw['regression_models']!r}\n"
            f"SCORING_MODE = {bw['scoring']!r}\n"
            f"PARAMS = {{'top_n': 30, 'tx_cost_bp': 30, 'weight_cap_pct': 15}}\n"
            f"OUTLIER_FILTERS = {{}}\nSCORE_MAP = {{}}\nSCORING_RULES = {{}}\nQUALITY_FILTER = {{}}\n"
        )
    else:
        _loaded = load_strategy(selected_view)
        _view_code = _loaded["code"] if _loaded and "code" in _loaded else None

    _view_info = _extract_strategy_info(_view_code) if _view_code else None
    _view_params = _view_info["params"] if _view_info else {}

    # ── KPI ──
    strat_summary = extract_strategy_summary(_view_code) if _view_code else {}
    scoring_mode = strat_summary.get("scoring_mode", {})
    large_mode_label = "사분위(0-4점)" if scoring_mode.get("large") == "quartile" else "십분위(0-10점)"
    small_mode_label = "사분위(0-4점)" if scoring_mode.get("small") == "quartile" else "십분위(0-10점)"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("편입 종목수", f"{_view_params.get('top_n', 30)}개")
    with col2:
        st.metric("편도 거래비용", f"{_view_params.get('tx_cost_bp', 30)}bp")
    with col3:
        st.metric("개별종목 비중상한", f"{_view_params.get('weight_cap_pct', 15)}%")
    with col4:
        n_reg = len(_view_info["regression_models"]) if _view_info else 0
        st.metric("회귀 모델 수", f"{n_reg}개")
    with col5:
        n_l = strat_summary.get("n_factors_large", 0)
        n_s = strat_summary.get("n_factors_small", 0)
        st.metric("팩터 수 (대형/중소)", f"{n_l} / {n_s}")

    # ── 탭: 가중치 테이블 | 가중치 차트 | 전략 코드 ──
    tab1, tab2, tab3 = st.tabs(["가중치 테이블", "가중치 차트", "전략 코드"])
    with tab1:
        st.markdown(
            f"**채점**: 대형주 {large_mode_label} · 중소형주 {small_mode_label}"
        )
        if _view_code:
            weights_data = format_weights_for_display(_view_code)
            if weights_data:
                large_only_set = set(strat_summary.get("large_only", []))
                small_only_set = set(strat_summary.get("small_only", []))
                html = _build_weights_html(weights_data, large_only_set, small_only_set)
                st.markdown(html, unsafe_allow_html=True)
    with tab2:
        if _view_info:
            st.plotly_chart(strategy_weight_chart(_view_info["weights_large"]), width="stretch")
    with tab3:
        if _view_code:
            st.code(_view_code, language="python")

    # ─── 2. 백테스트 ───
    st.markdown("---")
    section_header("백테스트")
    st.caption("저장하지 않아도 바로 실행할 수 있습니다. 수정 전략 vs 기존전략 vs KOSPI 200 비교.")

    if not st.session_state.lab_modified_results:
        if st.button("백테스트 실행", type="primary", key="btn_backtest", use_container_width=True):
            progress_bar = st.progress(0, text="백테스트 실행 중...")

            def progress_callback(current, total):
                pct = current / max(total, 1)
                progress_bar.progress(pct, text=f"백테스트 실행 중... ({current}/{total})")

            try:
                modified = run_strategy_backtest(
                    current_code,
                    progress_callback=progress_callback,
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
        original = load_backtest_results()
        modified = st.session_state.lab_modified_results

        comp_data = []
        m = modified.get("CUSTOM", {})
        if m:
            comp_data.append({
                "전략": "수정 전략",
                "총수익률": f"{m.get('total_return', 0):+.1%}",
                "CAGR": f"{m.get('cagr', 0):+.1%}",
                "MDD": f"{m.get('mdd', 0):.1%}",
                "Sharpe": f"{m.get('sharpe', 0):.3f}",
            })
        for key in ["A0", "KOSPI"]:
            r = original.get(key, {})
            if r:
                comp_data.append({
                    "전략": STRATEGY_LABELS.get(key, key),
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
        st.plotly_chart(comparison_cumulative_chart(original, modified), width="stretch")

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
