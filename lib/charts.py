"""
Plotly chart builders for the dashboard.
"""
import numpy as np
import plotly.graph_objects as go

from lib.data import STRATEGY_KEYS, ALL_KEYS, STRATEGY_LABELS, STRATEGY_COLORS, BACKTEST_CONFIG


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout(**kwargs):
    defaults = dict(
        font=dict(family="AppleGothic, Nanum Gothic, sans-serif", size=13),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=12),
        ),
        margin=dict(t=60, b=50, l=60, r=20),
    )
    defaults.update(kwargs)
    return defaults


# ═══════════════════════════════════════════════════════
# Page 1
# ═══════════════════════════════════════════════════════

def cumulative_return_chart(results: dict) -> go.Figure:
    fig = go.Figure()
    oos_start = BACKTEST_CONFIG.get("oos_start", "2024-07-01")

    for key in ALL_KEYS:
        r = results.get(key)
        if not r:
            continue
        dates = r["rebalance_dates"]
        cum_pct = [(v - 1) * 100 for v in r["portfolio_values"]]
        fig.add_trace(go.Scatter(
            x=dates, y=cum_pct,
            name=STRATEGY_LABELS[key],
            line=dict(
                color=STRATEGY_COLORS[key],
                width=2.5 if key != "KOSPI" else 1.5,
                dash="dot" if key == "KOSPI" else "solid",
            ),
            hovertemplate="%{y:+.1f}%<extra>" + STRATEGY_LABELS[key] + "</extra>",
        ))

    fig.add_shape(type="line", x0=oos_start, x1=oos_start, y0=0, y1=1,
                  yref="paper", line=dict(color="white", dash="dash", width=1), opacity=0.4)
    fig.add_annotation(x=oos_start, y=1, yref="paper", text="IS | OOS",
                       showarrow=False, font=dict(color="white", size=11), yshift=10)

    fig.update_layout(**_base_layout(
        title="누적 수익률 (%)",
        yaxis_title="수익률 (%)",
        height=450,
    ))
    return fig


def drawdown_chart(results: dict) -> go.Figure:
    fig = go.Figure()

    for key in ALL_KEYS:
        r = results.get(key)
        if not r:
            continue
        values = r["portfolio_values"]
        dates = r["rebalance_dates"]

        peak = values[0]
        dd = []
        for v in values:
            if v > peak:
                peak = v
            dd.append((v / peak - 1) * 100)

        fig.add_trace(go.Scatter(
            x=dates, y=dd,
            name=STRATEGY_LABELS[key],
            fill="tozeroy",
            line=dict(color=STRATEGY_COLORS[key], width=1),
            fillcolor=_hex_to_rgba(STRATEGY_COLORS[key], 0.15),
            hovertemplate="%{y:.1f}%<extra>" + STRATEGY_LABELS[key] + "</extra>",
        ))

    fig.update_layout(**_base_layout(
        title="Drawdown (%)",
        yaxis_title="하락폭 (%)",
        height=350,
    ))
    return fig


# ═══════════════════════════════════════════════════════
# Page 2
# ═══════════════════════════════════════════════════════

def monthly_heatmap(results: dict) -> go.Figure:
    keys = [k for k in ALL_KEYS if k in results]
    labels = [STRATEGY_LABELS[k] for k in keys]

    ref_key = keys[0]
    dates = results[ref_key]["rebalance_dates"][:-1]
    date_labels = [d[:7] for d in dates]

    z = []
    text = []
    for k in keys:
        rets = results[k]["monthly_returns"]
        z.append([r * 100 for r in rets])
        text.append([f"{r*100:+.1f}%" for r in rets])

    n_months = len(date_labels)
    show_text = n_months <= 24

    fig = go.Figure(go.Heatmap(
        z=z, x=date_labels, y=labels,
        text=text,
        texttemplate="%{text}" if show_text else None,
        textfont=dict(size=11) if show_text else None,
        colorscale="RdYlGn", zmid=0,
        colorbar=dict(title="수익률(%)", len=0.9),
        hovertemplate="전략: %{y}<br>기간: %{x}<br>수익률: %{text}<extra></extra>",
        ygap=4,
        xgap=1,
    ))

    dtick = 1 if n_months <= 24 else (3 if n_months <= 48 else 6)
    fig.update_layout(**_base_layout(
        title="월별 수익률 히트맵",
        height=max(len(keys) * 100 + 120, 350),
        xaxis=dict(type="category", dtick=dtick, tickangle=-45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=13)),
    ))
    return fig


def attribution_chart(attr_df, strategy_label: str, month_label: str) -> go.Figure:
    """종목별 기여도 가로 바 차트 (상위/하위 각 10개)."""
    if attr_df.empty:
        return go.Figure()

    # 상위 5 + 하위 5 선택 (이미 기여도 오름차순 정렬됨)
    n = len(attr_df)
    if n > 15:
        bottom = attr_df.head(5)
        top = attr_df.tail(5)
        df = __import__("pandas").concat([bottom, top])
    else:
        df = attr_df

    colors = ["#EF5350" if v < 0 else "#4CAF50" for v in df["기여도(%)"]]

    fig = go.Figure(go.Bar(
        y=df["종목명"],
        x=df["기여도(%)"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in df["기여도(%)"]],
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "비중: %{customdata[0]:.1f}%<br>"
            "종목수익률: %{customdata[1]:+.1f}%<br>"
            "기여도: %{x:+.2f}%"
            "<extra></extra>"
        ),
        customdata=list(zip(df["비중(%)"], df["종목수익률(%)"])),
    ))

    fig.add_vline(x=0, line_color="white", opacity=0.3)
    fig.update_layout(**_base_layout(
        title=f"{strategy_label} — {month_label} 종목별 기여도",
        xaxis_title="기여도 (%p)",
        height=max(len(df) * 35 + 120, 300),
        showlegend=False,
        hovermode="closest",
    ))
    return fig


def rolling_excess_chart(results: dict, window: int = 12) -> go.Figure:
    fig = go.Figure()

    bm_rets = np.array(results.get("KOSPI", {}).get("monthly_returns", []))
    if len(bm_rets) == 0:
        return fig

    ref_key = next((k for k in STRATEGY_KEYS if k in results), None)
    if not ref_key:
        return fig
    dates = results[ref_key]["rebalance_dates"][:-1]

    for key in STRATEGY_KEYS:
        r = results.get(key)
        if not r:
            continue
        strat_rets = np.array(r["monthly_returns"])
        n = min(len(strat_rets), len(bm_rets))
        excess = strat_rets[:n] - bm_rets[:n]

        rolling = []
        for i in range(n):
            start = max(0, i - window + 1)
            rolling.append(np.sum(excess[start:i + 1]) * 100)

        fig.add_trace(go.Scatter(
            x=dates[:n], y=rolling,
            name=STRATEGY_LABELS[key],
            line=dict(color=STRATEGY_COLORS[key], width=2),
        ))

    fig.add_hline(y=0, line_color="white", opacity=0.3)
    fig.update_layout(**_base_layout(
        title=f"롤링 {window}개월 누적 초과수익률 vs KOSPI 200 (%)",
        yaxis_title="초과수익률 (%)",
        height=400,
    ))
    return fig


def monthly_distribution_chart(results: dict) -> go.Figure:
    fig = go.Figure()

    for key in ALL_KEYS:
        r = results.get(key)
        if not r:
            continue
        rets_pct = [x * 100 for x in r["monthly_returns"]]
        fig.add_trace(go.Box(
            y=rets_pct,
            name=STRATEGY_LABELS[key],
            marker_color=STRATEGY_COLORS[key],
            boxmean=True,
        ))

    fig.update_layout(**_base_layout(
        title="월별 수익률 분포 (%)",
        yaxis_title="수익률 (%)",
        height=400,
        showlegend=False,
    ))
    return fig


# ═══════════════════════════════════════════════════════
# Page 3
# ═══════════════════════════════════════════════════════

def sector_pie_chart(holdings_df) -> go.Figure:
    if holdings_df.empty:
        return go.Figure()

    sector_weights = holdings_df.groupby("섹터")["비중(%)"].sum().sort_values(ascending=False)

    fig = go.Figure(go.Pie(
        labels=sector_weights.index,
        values=sector_weights.values,
        hole=0.4,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title="섹터 비중",
        height=400,
        showlegend=False,
    ))
    return fig


def sector_comparison_chart(holdings_dict: dict, min_pct: float = 3.0) -> go.Figure:
    """N개 전략 섹터 비중 비교. holdings_dict = {strategy_key: holdings_df}"""
    sector_data = {}
    for key, df in holdings_dict.items():
        if not df.empty:
            sector_data[key] = dict(df.groupby("섹터")["비중(%)"].sum())

    if not sector_data:
        return go.Figure()

    all_sectors_raw = set()
    for sd in sector_data.values():
        all_sectors_raw |= set(sd.keys())

    main_sectors = [s for s in all_sectors_raw
                    if any(sd.get(s, 0) >= min_pct for sd in sector_data.values())]
    main_sectors.sort(key=lambda s: sum(sd.get(s, 0) for sd in sector_data.values()), reverse=True)

    etc_vals = {}
    for key, sd in sector_data.items():
        etc_vals[key] = sum(sd.get(s, 0) for s in all_sectors_raw if s not in main_sectors)
    if any(v > 0 for v in etc_vals.values()):
        main_sectors.append("기타")

    def _short(s):
        return s.replace("코스피 ", "") if s != "기타" else s

    labels = [_short(s) for s in main_sectors]

    fig = go.Figure()
    for key in holdings_dict:
        sd = sector_data.get(key, {})
        etc = etc_vals.get(key, 0)
        vals = [sd.get(s, 0) if s != "기타" else etc for s in main_sectors]
        fig.add_trace(go.Bar(
            name=STRATEGY_LABELS.get(key, key), y=labels, x=vals,
            orientation="h", marker_color=STRATEGY_COLORS.get(key, "#999"),
            text=[f"{v:.1f}%" for v in vals], textposition="outside",
        ))

    fig.update_layout(**_base_layout(
        title="섹터 비중 비교",
        xaxis_title="비중 (%)",
        barmode="group",
        height=max(len(labels) * 50 + 100, 350),
        yaxis=dict(autorange="reversed"),
    ))
    return fig


def market_cap_distribution_chart(dist_dict: dict) -> go.Figure:
    """N개 전략 시가총액 분포 비교. dist_dict = {strategy_key: {cat: pct}}"""
    all_cats = ["초대형", "대형", "중형", "소형"]
    cats = [c for c in all_cats
            if any(d.get(c, 0) > 0 for d in dist_dict.values())]
    if not cats:
        return go.Figure()

    fig = go.Figure()
    for key, dist in dist_dict.items():
        vals = [dist.get(c, 0) for c in cats]
        fig.add_trace(go.Bar(
            name=STRATEGY_LABELS.get(key, key), y=cats, x=vals, orientation="h",
            marker_color=STRATEGY_COLORS.get(key, "#999"),
            text=[f"{v:.1f}%" for v in vals], textposition="outside",
        ))

    fig.update_layout(**_base_layout(
        title="시가총액 분포 비교 (비중 기준)",
        xaxis_title="비중 (%)",
        barmode="group",
        height=max(len(cats) * 70 + 100, 300),
        yaxis=dict(autorange="reversed"),
    ))
    return fig


def concentration_chart(holdings_dict: dict) -> go.Figure:
    """N개 전략 Top 5 합집합 종목 비중 비교 (마커 차트)."""
    _symbols = ["circle", "diamond", "square", "cross", "triangle-up", "star"]

    weight_maps = {}
    all_names = []
    seen = set()

    for key, df in holdings_dict.items():
        if df.empty:
            continue
        weight_maps[key] = dict(zip(df["종목명"], df["비중(%)"]))
        top5 = df.nlargest(5, "비중(%)")
        for _, row in top5.iterrows():
            if row["종목명"] not in seen:
                all_names.append(row["종목명"])
                seen.add(row["종목명"])

    if not all_names:
        return go.Figure()

    all_names.sort(key=lambda n: max(wm.get(n, 0) for wm in weight_maps.values()))

    fig = go.Figure()
    keys = list(holdings_dict.keys())

    # 연결선 (2개 전략일 때만)
    if len(keys) == 2:
        k1, k2 = keys
        for name in all_names:
            v1 = weight_maps.get(k1, {}).get(name, 0)
            v2 = weight_maps.get(k2, {}).get(name, 0)
            fig.add_trace(go.Scatter(
                x=[v1, v2], y=[name, name],
                mode="lines", line=dict(color="rgba(255,255,255,0.25)", width=2),
                showlegend=False, hoverinfo="skip",
            ))

    for i, key in enumerate(keys):
        wm = weight_maps.get(key, {})
        fig.add_trace(go.Scatter(
            x=[wm.get(n, 0) for n in all_names],
            y=all_names, mode="markers+text",
            name=STRATEGY_LABELS.get(key, key),
            marker=dict(color=STRATEGY_COLORS.get(key, "#999"), size=12,
                        symbol=_symbols[i % len(_symbols)]),
            text=[f"{wm.get(n, 0):.1f}" if wm.get(n, 0) > 0 else "" for n in all_names],
            textposition="top center", textfont=dict(size=11),
            hovertemplate="%{y}: %{x:.1f}%<extra>" + STRATEGY_LABELS.get(key, key) + "</extra>",
        ))

    fig.update_layout(**_base_layout(
        title="주요 보유 종목 비중 비교",
        xaxis_title="비중 (%)",
        height=max(len(all_names) * 50 + 100, 350),
        xaxis=dict(range=[0, None]),
    ))
    return fig


def overlap_heatmap(overlap_df) -> go.Figure:
    labels = list(overlap_df.index)
    z = overlap_df.values.tolist()
    text = [[str(int(v)) for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale="Blues",
        hovertemplate="%{y} vs %{x}: %{z}종목 중복<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title="전략 간 종목 중복도 (30종목 기준)",
        height=350,
        yaxis=dict(autorange="reversed"),
    ))
    return fig


# ═══════════════════════════════════════════════════════
# Page 4
# ═══════════════════════════════════════════════════════

def bootstrap_histogram(stat_data: dict, keys: list = None) -> go.Figure:
    fig = go.Figure()

    for key in (keys or STRATEGY_KEYS):
        sig = stat_data["bm_significance"].get(key)
        if not sig:
            continue
        boot_pct = sig["boot_means"] * 100
        fig.add_trace(go.Histogram(
            x=boot_pct, nbinsx=80,
            name=f"{STRATEGY_LABELS.get(key, key)} (p={sig['p_value']:.3f})",
            marker_color=STRATEGY_COLORS[key],
            opacity=0.6,
        ))

    fig.add_vline(x=0, line_color="white", line_width=2,
                  annotation_text="차이=0", annotation_font_color="white")

    fig.update_layout(**_base_layout(
        title="Bootstrap 분포: 월간 초과수익률 vs KOSPI 200",
        xaxis_title="월간 초과수익률 (%)",
        yaxis_title="빈도",
        height=400,
        barmode="overlay",
    ))
    return fig


def is_oos_comparison_chart(is_oos_data: dict, keys: list = None) -> go.Figure:
    is_results = is_oos_data["is_results"]
    oos_results = is_oos_data["oos_results"]
    bm = is_oos_data.get("benchmarks", {})

    labels = []
    is_sharpes = []
    oos_sharpes = []
    is_returns = []
    oos_returns = []

    for key in (keys or STRATEGY_KEYS):
        labels.append(STRATEGY_LABELS.get(key, key))
        is_r = is_results.get(key, {})
        oos_r = oos_results.get(key, {})
        is_sharpes.append(is_r.get("sharpe", 0))
        oos_sharpes.append(oos_r.get("sharpe", 0))
        is_returns.append((is_r.get("total_return", 0)) * 100)
        oos_returns.append((oos_r.get("total_return", 0)) * 100)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="In-Sample", x=labels, y=is_sharpes,
        marker_color="#2196F3", opacity=0.8,
        text=[f"{s:.2f}" for s in is_sharpes], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Out-of-Sample", x=labels, y=oos_sharpes,
        marker_color="#E91E63", opacity=0.8,
        text=[f"{s:.2f}" for s in oos_sharpes], textposition="outside",
    ))

    # BM reference lines
    is_bm_sh = bm.get("is", {}).get("KOSPI", {})
    oos_bm_sh = bm.get("oos", {}).get("KOSPI", {})

    fig.update_layout(**_base_layout(
        title="IS vs OOS Sharpe Ratio",
        yaxis_title="Sharpe",
        barmode="group",
        height=400,
    ))
    return fig


def rolling_window_chart(rolling_all: dict, keys: list = None) -> go.Figure:
    fig = go.Figure()

    for key in (keys or STRATEGY_KEYS):
        rd = rolling_all.get(key)
        if not rd:
            continue
        dates = [r["start_date"] for r in rd["rolling_data"] if r["start_date"]]
        excess = [r["excess_return"] * 100 for r in rd["rolling_data"] if r["start_date"]]

        fig.add_trace(go.Scatter(
            x=dates, y=excess,
            name=f"{STRATEGY_LABELS.get(key, key)} (승률 {rd['win_rate']:.0%})",
            line=dict(color=STRATEGY_COLORS[key], width=2),
        ))

    fig.add_hline(y=0, line_color="white", opacity=0.3)
    fig.update_layout(**_base_layout(
        title="롤링 24개월 초과수익률 vs KOSPI 200 (%)",
        yaxis_title="초과수익률 (%)",
        height=400,
    ))
    return fig


# ═══════════════════════════════════════════════════════
# Strategy Lab
# ═══════════════════════════════════════════════════════

def strategy_weight_chart(weights: dict) -> go.Figure:
    """팩터 가중치 카테고리별 가로 바 차트"""
    from lib.ai import FACTOR_CATEGORIES, FACTOR_LABELS

    cat_colors = {
        "밸류에이션": "#2196F3",
        "회귀 매력도": "#E91E63",
        "성장성": "#4CAF50",
        "차별화": "#FF9800",
        "커스텀": "#9C27B0",
    }

    labels = []
    values = []
    colors = []

    for cat, factors in FACTOR_CATEGORIES.items():
        for f in factors:
            w = weights.get(f, 0)
            if w > 0:
                labels.append(FACTOR_LABELS.get(f, f))
                values.append(w * 100)
                colors.append(cat_colors.get(cat, "#999"))

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.0f}%<extra></extra>",
    ))

    # 카테고리 합계 annotation
    total = sum(weights.get(f, 0) for f in weights)
    fig.update_layout(**_base_layout(
        title=f"팩터 가중치 구성 (합계 {total*100:.0f}%)",
        xaxis_title="가중치 (%)",
        height=max(len(labels) * 35 + 100, 300),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    ))
    return fig


def comparison_cumulative_chart(original: dict, modified: dict) -> go.Figure:
    """기존 vs 수정 전략 누적수익률 비교 차트"""
    fig = go.Figure()

    # 기존전략 (A0)
    orig_a0 = original.get("A0", {})
    if orig_a0:
        dates = orig_a0["rebalance_dates"]
        cum = [(v - 1) * 100 for v in orig_a0["portfolio_values"]]
        fig.add_trace(go.Scatter(
            x=dates, y=cum,
            name=STRATEGY_LABELS.get("A0", "기존전략"),
            line=dict(color="#9467bd", width=2.5),
            hovertemplate="%{y:+.1f}%<extra>" + STRATEGY_LABELS.get("A0", "기존전략") + "</extra>",
        ))

    # 수정 전략 (CUSTOM 또는 A0)
    mod_a0 = modified.get("CUSTOM", modified.get("A0", {}))
    if mod_a0:
        dates = mod_a0["rebalance_dates"]
        cum = [(v - 1) * 100 for v in mod_a0["portfolio_values"]]
        fig.add_trace(go.Scatter(
            x=dates, y=cum,
            name="수정 전략",
            line=dict(color="#FF5722", width=2.5, dash="dash"),
            hovertemplate="%{y:+.1f}%<extra>수정 전략</extra>",
        ))

    # BM
    bm = original.get("KOSPI", {})
    if bm:
        dates = bm["rebalance_dates"]
        cum = [(v - 1) * 100 for v in bm["portfolio_values"]]
        fig.add_trace(go.Scatter(
            x=dates, y=cum,
            name="KOSPI 200",
            line=dict(color="#7f7f7f", width=1.5, dash="dot"),
            hovertemplate="%{y:+.1f}%<extra>KOSPI 200</extra>",
        ))

    fig.update_layout(**_base_layout(
        title="누적 수익률 비교: 기존 vs 수정 (%)",
        yaxis_title="수익률 (%)",
        height=450,
    ))
    return fig
