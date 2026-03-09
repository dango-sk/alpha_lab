"""
대시보드 공통 스타일 — 다크/라이트 모드 자동 대응
"""
import streamlit as st

from lib.data import STRATEGY_COLORS  # 동적 전략 색상 (data.py에서 관리)


def inject_css():
    """전역 커스텀 CSS 주입 (다크/라이트 모두 대응)"""
    st.markdown("""
    <style>
    /* KPI 카드 — 반투명 배경으로 테마 자동 대응 */
    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .kpi-label {
        font-size: 13px;
        font-weight: 500;
        opacity: 0.7;
        margin-bottom: 4px;
        letter-spacing: 0.3px;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        margin: 4px 0;
        letter-spacing: -0.5px;
    }
    .kpi-sub {
        font-size: 12px;
        opacity: 0.6;
        margin-top: 6px;
    }
    .kpi-sub span {
        display: inline-block;
        margin: 0 6px;
    }
    .positive { color: #4CAF50; }
    .negative { color: #EF5350; }

    /* 섹션 헤더 */
    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 28px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .section-header h3 {
        margin: 0;
        font-size: 20px;
        font-weight: 600;
    }

    /* 로딩 프로그레스 바 — Streamlit 스피너 커스텀 */
    @keyframes progressSlide {
        0%   { left: -40%; }
        100% { left: 100%; }
    }
    .stSpinner > div {
        display: inline-flex !important;
        align-items: center;
        gap: 12px;
        padding: 12px 20px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        position: relative;
        overflow: hidden;
    }
    .stSpinner > div::before {
        content: "";
        position: absolute;
        bottom: 0;
        left: -40%;
        width: 40%;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.8), transparent);
        animation: progressSlide 1.2s ease-in-out infinite;
    }
    .stSpinner > div::after {
        content: none;
    }

    /* 커스텀 로딩 오버레이 */
    .loading-overlay {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 20px;
        opacity: 0.8;
    }
    .loading-bar-track {
        width: 200px;
        height: 3px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 2px;
        overflow: hidden;
        margin-bottom: 16px;
    }
    .loading-bar-fill {
        width: 40%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.8), transparent);
        animation: progressSlide 1.2s ease-in-out infinite;
        position: relative;
    }
    .loading-text {
        font-size: 14px;
        opacity: 0.6;
        letter-spacing: 0.5px;
    }
    .loading-dots::after {
        content: "";
        animation: dots 1.5s steps(4, end) infinite;
    }
    @keyframes dots {
        0%   { content: ""; }
        25%  { content: "."; }
        50%  { content: ".."; }
        75%  { content: "..."; }
    }

    /* 페이지 전환 시 프로그레스 바 오버레이 */
    @keyframes topBarSlide {
        0%   { left: -30%; }
        100% { left: 100%; }
    }

    /* Streamlit이 Running 상태일 때 상단 프로그레스 바 */
    .stApp[data-test-script-state="running"] [data-testid="stMain"]::after {
        content: "";
        position: fixed;
        top: 0;
        left: -30%;
        width: 30%;
        height: 3px;
        z-index: 9999;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.9), transparent);
        animation: topBarSlide 1s ease-in-out infinite;
        pointer-events: none;
    }

    /* Running 상태에서 기존 콘텐츠 살짝 흐리게 */
    .stApp[data-test-script-state="running"] [data-testid="stMain"] {
        opacity: 0.6;
        transition: opacity 0.15s ease-out;
    }

    /* 기본 상태에서는 부드럽게 복원 */
    .stApp [data-testid="stMain"] {
        transition: opacity 0.2s ease-in;
    }

    /* Streamlit 기본 Running 뱃지 숨기기 */
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* ═══ Weight Visualization (wv-*) ═══ */

    /* Overview stacked bar */
    .wv-overview {
        display: flex;
        height: 48px;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.2);
    }
    .wv-seg {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        cursor: default;
        transition: filter 0.2s, flex 0.4s;
        position: relative;
        overflow: hidden;
    }
    .wv-seg::after {
        content: "";
        position: absolute;
        top: 0; left: -100%; width: 60%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        animation: wv-shimmer 3s ease-in-out infinite;
    }
    @keyframes wv-shimmer {
        0%   { left: -100%; }
        100% { left: 200%; }
    }
    .wv-seg:hover { filter: brightness(1.25); }
    .wv-seg-label {
        font-size: 11px;
        font-weight: 600;
        color: #fff;
        text-shadow: 0 1px 3px rgba(0,0,0,0.4);
        white-space: nowrap;
        line-height: 1.2;
    }
    .wv-seg-pct {
        font-size: 15px;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 1px 3px rgba(0,0,0,0.4);
        line-height: 1.2;
    }

    /* Legend */
    .wv-legend {
        display: flex;
        gap: 16px;
        margin-bottom: 14px;
        font-size: 12px;
        opacity: 0.6;
    }
    .wv-legend-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 4px;
        vertical-align: middle;
    }

    /* Category card */
    .wv-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: background 0.2s;
    }
    .wv-card:hover {
        background: rgba(255,255,255,0.05);
    }
    .wv-card-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .wv-card-title {
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .wv-card-pct {
        font-size: 22px;
        font-weight: 800;
    }

    /* Factor row */
    .wv-factor {
        display: flex;
        align-items: center;
        padding: 5px 0;
    }
    .wv-fname {
        width: 210px;
        min-width: 210px;
        font-size: 13px;
        opacity: 0.85;
    }
    .wv-fbars {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .wv-fbar {
        display: flex;
        align-items: center;
        gap: 8px;
        height: 14px;
    }
    .wv-fbar-track {
        flex: 1;
        height: 7px;
        border-radius: 4px;
        background: rgba(255,255,255,0.04);
        overflow: hidden;
    }
    .wv-fbar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    .wv-fbar-val {
        width: 36px;
        font-size: 12px;
        font-weight: 600;
        text-align: right;
    }

    /* Exclusive badge */
    .wv-badge {
        display: inline-block;
        font-size: 10px;
        padding: 1px 6px;
        border-radius: 4px;
        margin-left: 6px;
        font-weight: 500;
        vertical-align: middle;
    }

    /* Total footer */
    .wv-total {
        text-align: right;
        font-size: 12px;
        opacity: 0.5;
        margin-top: 4px;
    }

    /* ═══ Scoring Visualization (wv-sc-*) ═══ */

    .wv-sc-wrap {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }
    .wv-sc-card {
        flex: 1;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 14px 16px 12px;
    }
    .wv-sc-head {
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .wv-sc-label {
        font-size: 10px;
        opacity: 0.4;
        margin-bottom: 4px;
        letter-spacing: 0.5px;
    }
    .wv-sc-bar {
        display: flex;
        gap: 3px;
        margin-bottom: 2px;
    }
    .wv-sc-cell {
        flex: 1;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 5px;
        font-size: 12px;
        font-weight: 700;
        color: #fff;
        text-shadow: 0 1px 2px rgba(0,0,0,0.4);
        transition: transform 0.15s, filter 0.15s;
    }
    .wv-sc-cell:hover {
        transform: scale(1.12);
        filter: brightness(1.3);
        z-index: 1;
    }
    .wv-sc-arrow {
        text-align: center;
        font-size: 12px;
        opacity: 0.45;
        padding: 6px 0;
        letter-spacing: 1px;
    }
    .wv-sc-norms {
        display: flex;
        gap: 3px;
        margin-bottom: 6px;
    }
    .wv-sc-norms > span {
        flex: 1;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
        color: #fff;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    .wv-sc-eq {
        text-align: center;
        font-size: 11px;
        font-weight: 500;
        opacity: 0.5;
        margin-top: 2px;
    }
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub_items: list[tuple[str, str]], color: str = "#42A5F5"):
    """
    커스텀 KPI 카드 HTML 렌더링.
    sub_items: [(label, value), ...] 예: [("MDD", "-36.3%"), ("Sharpe", "0.896")]
    """
    val_class = ""
    try:
        num = float(value.replace("%", "").replace("+", ""))
        val_class = "positive" if num > 0 else "negative" if num < 0 else ""
    except ValueError:
        pass

    sub_html = ""
    for s_label, s_val in sub_items:
        sub_html += f'<span>{s_label} <b>{s_val}</b></span>'

    html = f"""
    <div class="kpi-card" style="border-top: 3px solid {color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {val_class}">{value}</div>
        <div class="kpi-sub">{sub_html}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def section_header(title: str):
    """스타일된 섹션 헤더"""
    st.markdown(f"""
    <div class="section-header">
        <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)


def loading_placeholder(message: str = "데이터를 불러오는 중"):
    """고양이 로딩 플레이스홀더 표시. with 블록에서 사용."""
    return st.empty(), message


def show_loading(placeholder, message: str = "데이터를 불러오는 중"):
    """플레이스홀더에 로딩 애니메이션 표시"""
    placeholder.markdown(f"""
    <div class="loading-overlay">
        <div class="loading-bar-track"><div class="loading-bar-fill"></div></div>
        <div class="loading-text">{message}<span class="loading-dots"></span></div>
    </div>
    """, unsafe_allow_html=True)


def color_value(val, reverse=False):
    """숫자 문자열에 조건부 색상 적용."""
    if not isinstance(val, str):
        return ""
    clean = val.replace("%", "").replace("+", "").replace(",", "").strip()
    try:
        num = float(clean)
    except ValueError:
        return ""
    if reverse:
        num = -num
    if num > 0:
        return "color: #4CAF50; font-weight: 600"
    elif num < 0:
        return "color: #EF5350; font-weight: 600"
    return ""
