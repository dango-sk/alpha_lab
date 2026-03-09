"""
시황 스터디 — AI 기반 시황 학습 + 면접 대비 앱
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datetime import date, timedelta

import streamlit as st
from lib.ai import is_ai_available
from lib.market_ai import (
    generate_briefing,
    get_briefing_text,
    chat_qa,
    generate_quiz,
    grade_answer,
    chat_discussion,
    save_quiz_result,
    load_quiz_history,
    get_wrong_questions,
    extract_score,
    save_qa_history,
    load_qa_history,
    save_briefing,
    load_briefing,
    analyze_stock,
    chat_stock_qa,
    generate_stock_quiz,
    grade_stock_answer,
    save_stock_study,
    load_stock_history,
    get_stock_analysis_text,
    get_stock_picks,
    generate_global_quiz,
    get_weekly_trades,
    DIFFICULTY_LEVELS,
)

st.set_page_config(page_title="시황 스터디", page_icon="📰", layout="wide")

# ─── 커스텀 CSS ───
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }

    /* 지표 카드 */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        min-height: 90px;
    }
    .metric-name {
        font-size: 11px;
        font-weight: 500;
        opacity: 0.5;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 20px;
        font-weight: 700;
        margin: 2px 0;
    }
    .metric-change {
        font-size: 13px;
        font-weight: 600;
    }
    .metric-up { color: #ef5350; }
    .metric-down { color: #42a5f5; }
    .metric-flat { color: #888; }

    /* 센티먼트 뱃지 */
    .sentiment-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .sentiment-risk_on { background: rgba(239,83,80,0.15); color: #ef5350; }
    .sentiment-cautious { background: rgba(255,152,0,0.15); color: #FF9800; }
    .sentiment-risk_off { background: rgba(66,165,245,0.15); color: #42a5f5; }

    /* 핵심 포인트 */
    .key-point {
        background: rgba(255,152,0,0.06);
        border-left: 3px solid #FF9800;
        padding: 10px 16px;
        margin: 6px 0;
        border-radius: 0 8px 8px 0;
        font-size: 14px;
    }

    /* 섹션 카드 */
    .section-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
    }

    /* 퀴즈 */
    .quiz-question {
        background: rgba(255,152,0,0.08);
        border-left: 4px solid #FF9800;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin: 16px 0;
    }
    .grade-result {
        background: rgba(76,175,80,0.08);
        border-left: 4px solid #4CAF50;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin: 16px 0;
    }
    .mode-header {
        font-size: 14px;
        font-weight: 600;
        opacity: 0.6;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* 복습 카드 */
    .review-card {
        background: rgba(239,83,80,0.06);
        border: 1px solid rgba(239,83,80,0.15);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
    }
    .score-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 700;
    }
    .score-low { background: rgba(239,83,80,0.2); color: #ef5350; }
    .score-mid { background: rgba(255,152,0,0.2); color: #FF9800; }
    .score-high { background: rgba(76,175,80,0.2); color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# ─── API 키 체크 ───
if not is_ai_available():
    st.error("ANTHROPIC_API_KEY가 설정되지 않았습니다. `.env` 파일을 확인하세요.")
    st.stop()

# ─── 세션 초기화 (저장된 데이터 복원) ───
if "briefing" not in st.session_state:
    st.session_state.briefing = load_briefing()
if "qa_messages" not in st.session_state:
    st.session_state.qa_messages = load_qa_history()

defaults = {
    "quiz_current": None,
    "quiz_graded": None,
    "quiz_history": [],
    "discussion_messages": [],
    "quiz_difficulty": "기초",
    "review_target": None,
    "review_graded": None,
    "stock_analysis": None,
    "stock_name": "",
    "stock_qa_messages": [],
    "stock_quiz_current": None,
    "stock_quiz_graded": None,
    "stock_discussion_messages": [],
    "stock_sub_mode": "분석",
    "stock_picks": None,
    "global_quiz": None,
    "global_quiz_answered": False,
    "global_quiz_selected": None,
    "weekly_trades": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── 헬퍼 함수 ───
def _render_metric_card(name: str, value: str, change: str, up: bool):
    if up is None:
        color_class = "metric-flat"
    elif up:
        color_class = "metric-up"
    else:
        color_class = "metric-down"
    arrow = "▲" if up else "▼" if up is not None else ""
    return f"""
    <div class="metric-card">
        <div class="metric-name">{name}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-change {color_class}">{arrow} {change}</div>
    </div>
    """


def _render_indicators(indicators: list):
    categories = {}
    for ind in indicators:
        cat = ind.get("category", "기타")
        categories.setdefault(cat, []).append(ind)

    cat_icons = {"지수": "📊", "금리": "📈", "환율": "💱", "원자재": "🛢️", "크립토": "₿"}

    for cat, items in categories.items():
        icon = cat_icons.get(cat, "📌")
        st.markdown(f"**{icon} {cat}**")
        cols = st.columns(len(items))
        for col, item in zip(cols, items):
            with col:
                st.markdown(
                    _render_metric_card(
                        item["name"], item["value"],
                        item["change"], item.get("up"),
                    ),
                    unsafe_allow_html=True,
                )
        st.markdown("")


def _render_briefing(briefing):
    if isinstance(briefing, str):
        st.markdown(briefing)
        return
    if "error" in briefing:
        st.error(f"브리핑 생성 실패: {briefing['error']}")
        return

    sentiment = briefing.get("sentiment", "cautious")
    sentiment_labels = {"risk_on": "RISK ON", "cautious": "CAUTIOUS", "risk_off": "RISK OFF"}
    s_label = sentiment_labels.get(sentiment, sentiment.upper())
    key_points = briefing.get("key_points", [])

    top1, top2 = st.columns([1, 4])
    with top1:
        st.markdown(
            f'<span class="sentiment-badge sentiment-{sentiment}">{s_label}</span>',
            unsafe_allow_html=True,
        )
    with top2:
        st.caption("오늘의 시장 분위기")

    if key_points:
        st.markdown("")
        for pt in key_points:
            st.markdown(f'<div class="key-point">{pt}</div>', unsafe_allow_html=True)
        st.markdown("")

    st.divider()

    indicators = briefing.get("indicators", [])
    if indicators:
        _render_indicators(indicators)
        st.divider()

    sections = briefing.get("sections", {})
    if sections:
        tab_names = list(sections.keys())
        tabs = st.tabs(tab_names)
        for tab, name in zip(tabs, tab_names):
            with tab:
                st.markdown(f'<div class="section-card">', unsafe_allow_html=True)
                st.markdown(sections[name])
                st.markdown('</div>', unsafe_allow_html=True)
    elif briefing.get("full_text"):
        st.markdown(briefing["full_text"])


def _score_badge(score):
    if score is None:
        return '<span class="score-badge score-mid">?/10</span>'
    if score <= 4:
        cls = "score-low"
    elif score <= 6:
        cls = "score-mid"
    else:
        cls = "score-high"
    return f'<span class="score-badge {cls}">{score}/10</span>'


def _calc_streak(history: list) -> dict:
    """퀴즈 히스토리에서 학습 통계 계산"""
    # date, timedelta already imported at top

    if not history:
        return {"streak": 0, "total_days": 0, "today_count": 0, "avg_score": 0}

    today = date.today().isoformat()

    # 날짜별 그룹핑
    dates = sorted(set(h.get("date", "") for h in history if h.get("date")))
    total_days = len(dates)

    # 오늘 풀은 문제 수
    today_count = sum(1 for h in history if h.get("date") == today)

    # 연속 스트릭 계산 (오늘 또는 어제부터 역순)
    streak = 0
    check = date.today()
    date_set = set(dates)

    # 오늘 안 했으면 어제부터 체크
    if today not in date_set:
        check = check - timedelta(days=1)

    while check.isoformat() in date_set:
        streak += 1
        check = check - timedelta(days=1)

    # 평균 점수
    scores = [h["score"] for h in history if h.get("score") is not None]
    avg_score = sum(scores) / len(scores) if scores else 0

    # 최근 7일 점수 추이
    week_ago = (date.today() - timedelta(days=6)).isoformat()
    recent = [h for h in history if h.get("date", "") >= week_ago and h.get("score") is not None]

    return {
        "streak": streak,
        "total_days": total_days,
        "today_count": today_count,
        "avg_score": avg_score,
        "recent_scores": recent,
    }


def _render_stats_bar():
    """상단 학습 통계 바 렌더링"""
    history = load_quiz_history()
    stats = _calc_streak(history)

    streak = stats["streak"]
    # 스트릭 이모지
    if streak == 0:
        streak_display = "0일"
    elif streak < 3:
        streak_display = f"{streak}일"
    elif streak < 7:
        streak_display = f"{streak}일 🔥"
    elif streak < 30:
        streak_display = f"{streak}일 🔥🔥"
    else:
        streak_display = f"{streak}일 🔥🔥🔥"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-name">연속 학습</div>
            <div class="metric-value" style="font-size:24px;">{streak_display}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-name">오늘 푼 문제</div>
            <div class="metric-value" style="font-size:24px;">{stats['today_count']}문제</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-name">총 학습일</div>
            <div class="metric-value" style="font-size:24px;">{stats['total_days']}일</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        avg = stats["avg_score"]
        color = "#ef5350" if avg <= 4 else "#FF9800" if avg <= 6 else "#4CAF50"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-name">평균 점수</div>
            <div class="metric-value" style="font-size:24px; color:{color};">{avg:.1f}<span style="font-size:14px; opacity:0.5;">/10</span></div>
        </div>
        """, unsafe_allow_html=True)


# ─── 헤더 + 통계 + 모드 선택 ───
st.markdown("## 시황 스터디")
_render_stats_bar()
st.markdown("")
modes = ["브리핑", "Q&A", "퀴즈/토론", "종목 공부", "복습"]
mode = st.radio("", modes, horizontal=True, label_visibility="collapsed", key="study_mode")
st.divider()


# ═══════════════════════════════════════════════════════
# 브리핑 모드
# ═══════════════════════════════════════════════════════
if mode == "브리핑":
    st.markdown('<p class="mode-header">DAILY MARKET BRIEFING</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("오늘의 글로벌 금융시장을 AI가 웹서치로 수집하여 정리합니다.")
    with col2:
        gen_btn = st.button("시황 생성", type="primary", use_container_width=True)

    if gen_btn:
        with st.spinner("웹에서 최신 시황을 수집하는 중..."):
            result = generate_briefing()
            st.session_state.briefing = result
            save_briefing(result)

    if st.session_state.briefing:
        _render_briefing(st.session_state.briefing)
    else:
        st.info("'시황 생성' 버튼을 눌러 오늘의 시황 브리핑을 받아보세요.")

    # ── 주간 트레이딩 아이디어 ──
    st.divider()
    st.markdown("##### 주간 트레이딩 아이디어")
    if st.button("이번 주 롱/숏 아이디어 생성", key="weekly_trades_btn", use_container_width=True):
        with st.spinner("시장 분석 중... 롱/숏 종목을 찾고 있습니다"):
            st.session_state.weekly_trades = get_weekly_trades()
        st.rerun()

    if st.session_state.weekly_trades:
        trades = st.session_state.weekly_trades
        week_label = trades.get("week", "이번 주")
        st.caption(f"기간: {week_label}")

        # 주요 이벤트
        events = trades.get("weekly_events", [])
        if events:
            with st.expander("이번 주 주요 이벤트"):
                for ev in events:
                    st.markdown(f"- {ev}")

        picks = trades.get("picks", [])
        kr_picks = [p for p in picks if p.get("market") == "KR"]
        us_picks = [p for p in picks if p.get("market") == "US"]

        for label, market_picks in [("한국", kr_picks), ("미국", us_picks)]:
            if not market_picks:
                continue
            st.markdown(f"**{label}**")
            for p in market_picks:
                direction = p.get("direction", "")
                icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                name = p.get("name", "")
                ticker = p.get("ticker", "")
                price = p.get("current_price", "")
                target = p.get("target", "")
                stop = p.get("stop_loss", "")

                color = "rgba(76,175,80,0.08)" if direction == "LONG" else "rgba(239,83,80,0.08)"
                border_color = "#4CAF50" if direction == "LONG" else "#ef5350"

                st.markdown(f"""
                <div style="background:{color}; border-left:4px solid {border_color}; border-radius:0 12px 12px 0; padding:16px 20px; margin:8px 0;">
                    <div style="font-size:16px; font-weight:700;">{icon} {name} ({ticker})</div>
                    <div style="font-size:13px; opacity:0.7; margin:4px 0;">현재가 {price} | 목표 {target} | 손절 {stop}</div>
                    <div style="font-size:14px; margin-top:8px;">{p.get('reason', '')}</div>
                    <div style="font-size:12px; opacity:0.6; margin-top:6px;">리스크: {p.get('risk', '')}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("")


# ═══════════════════════════════════════════════════════
# Q&A 모드
# ═══════════════════════════════════════════════════════
elif mode == "Q&A":
    st.markdown('<p class="mode-header">MARKET Q&A</p>', unsafe_allow_html=True)

    briefing_text = get_briefing_text(st.session_state.briefing)

    if not briefing_text:
        st.warning("먼저 '브리핑' 탭에서 오늘의 시황을 생성하세요. 브리핑 내용을 기반으로 Q&A가 진행됩니다.")

    container = st.container(height=500)
    with container:
        if not st.session_state.qa_messages:
            st.caption("시황에 대해 궁금한 것을 물어보세요:")
            st.caption("• 오늘 미국 금리가 왜 올랐어?")
            st.caption("• 환율 상승이 한국 시장에 미치는 영향은?")
            st.caption("• 연준 발언의 의미를 자세히 설명해줘")
        for msg in st.session_state.qa_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    with st.form("qa_form", clear_on_submit=True):
        user_input = st.text_input("", placeholder="질문하기...", label_visibility="collapsed")
        cols = st.columns([5, 1])
        with cols[1]:
            submitted = st.form_submit_button("전송", use_container_width=True)

    if submitted and user_input:
        st.session_state.qa_messages.append({"role": "user", "content": user_input})
        with st.spinner("답변 생성 중..."):
            reply = chat_qa(st.session_state.qa_messages, briefing_text)
        st.session_state.qa_messages.append({"role": "assistant", "content": reply})
        save_qa_history(st.session_state.qa_messages)
        st.rerun()

    if st.session_state.qa_messages:
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.qa_messages = []
            save_qa_history([])
            st.rerun()


# ═══════════════════════════════════════════════════════
# 퀴즈/토론 모드
# ═══════════════════════════════════════════════════════
elif mode == "퀴즈/토론":
    st.markdown('<p class="mode-header">INTERVIEW PREP</p>', unsafe_allow_html=True)

    briefing_text = get_briefing_text(st.session_state.briefing)

    if not briefing_text:
        st.warning("먼저 '브리핑' 탭에서 오늘의 시황을 생성하세요. 시황 기반으로 면접 문제가 출제됩니다.")

    quiz_type = st.radio("", ["서술형 면접", "글로벌 연결 (객관식)"], horizontal=True, label_visibility="collapsed", key="quiz_type_radio")
    st.divider()

    if quiz_type == "글로벌 연결 (객관식)":
        st.markdown("자산 간, 국가 간 연결고리를 테스트하는 객관식 퀴즈입니다.")

        if st.button("문제 출제", type="primary", key="global_quiz_btn"):
            with st.spinner("글로벌 연결 문제 생성 중..."):
                q = generate_global_quiz(briefing_text)
            st.session_state.global_quiz = q
            st.session_state.global_quiz_answered = False
            st.session_state.global_quiz_selected = None
            st.rerun()

        if st.session_state.global_quiz:
            q = st.session_state.global_quiz

            # 배경 컨텍스트
            if q.get("context"):
                st.caption(q["context"])

            # 질문
            st.markdown(f'<div class="quiz-question">', unsafe_allow_html=True)
            st.markdown(f"**{q['question']}**")
            st.markdown('</div>', unsafe_allow_html=True)

            # 선택지
            choices = q.get("choices", [])
            if not st.session_state.global_quiz_answered:
                for i, choice in enumerate(choices):
                    letter = choice[0] if choice else chr(65 + i)
                    if st.button(choice, key=f"gq_choice_{i}", use_container_width=True):
                        st.session_state.global_quiz_selected = letter
                        st.session_state.global_quiz_answered = True
                        st.rerun()
            else:
                correct = q.get("answer", "")
                selected = st.session_state.global_quiz_selected

                for choice in choices:
                    letter = choice[0] if choice else ""
                    if letter == correct:
                        st.success(f"✅ {choice}")
                    elif letter == selected:
                        st.error(f"❌ {choice}")
                    else:
                        st.markdown(f"　{choice}")

                is_correct = selected == correct
                if is_correct:
                    st.balloons()
                    st.markdown("**정답입니다!**")
                else:
                    st.markdown(f"**오답** — 정답은 **{correct}**")

                # 해설
                st.markdown(f'<div class="grade-result">', unsafe_allow_html=True)
                st.markdown(f"**해설:** {q.get('explanation', '')}")
                st.markdown('</div>', unsafe_allow_html=True)

                # 결과 저장
                if not st.session_state.get("global_quiz_saved"):
                    save_quiz_result({
                        "type": "global_mc",
                        "difficulty": "글로벌 연결",
                        "question": q["question"],
                        "answer": selected,
                        "grade": f"**[점수]** {'10' if is_correct else '0'}/10\n\n{q.get('explanation', '')}",
                    })
                    st.session_state.global_quiz_saved = True

                if st.button("다음 문제", key="global_next", use_container_width=True):
                    st.session_state.global_quiz = None
                    st.session_state.global_quiz_answered = False
                    st.session_state.global_quiz_selected = None
                    st.session_state.global_quiz_saved = False
                    st.rerun()

    else:
        # 기존 서술형 퀴즈
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            difficulty = st.selectbox(
                "난이도",
                options=list(DIFFICULTY_LEVELS.keys()),
                format_func=lambda x: f"{x} — {DIFFICULTY_LEVELS[x]['description']}",
                key="quiz_difficulty_select",
            )
            st.session_state.quiz_difficulty = difficulty
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            quiz_btn = st.button("문제 출제", type="primary", use_container_width=True)

        if quiz_btn:
            with st.spinner("면접 질문 생성 중..."):
                quiz = generate_quiz(briefing_text, difficulty)
            st.session_state.quiz_current = quiz
            st.session_state.quiz_graded = None
            st.session_state.discussion_messages = []
            st.rerun()

        if st.session_state.quiz_current:
            st.markdown('<div class="quiz-question">', unsafe_allow_html=True)
            st.markdown(st.session_state.quiz_current)
            st.markdown('</div>', unsafe_allow_html=True)

            if not st.session_state.quiz_graded:
                with st.form("answer_form"):
                    answer = st.text_area(
                        "답변 작성",
                        height=200,
                        placeholder="서술형으로 답변을 작성하세요. 근거와 논리를 포함해주세요.",
                    )
                    submit_answer = st.form_submit_button("답변 제출", type="primary", use_container_width=True)

                if submit_answer and answer:
                    with st.spinner("답변 채점 중..."):
                        grade = grade_answer(
                            briefing_text,
                            st.session_state.quiz_difficulty,
                            st.session_state.quiz_current,
                            answer,
                        )
                    st.session_state.quiz_graded = grade
                    st.session_state.discussion_messages = [
                        {"role": "user", "content": f"[질문]\n{st.session_state.quiz_current}\n\n[내 답변]\n{answer}"},
                        {"role": "assistant", "content": grade},
                    ]
                    # 세션 히스토리
                    entry = {
                        "difficulty": st.session_state.quiz_difficulty,
                        "question": st.session_state.quiz_current,
                        "answer": answer,
                        "grade": grade,
                    }
                    st.session_state.quiz_history.append(entry)
                    # 파일에 영구 저장
                    save_quiz_result(entry.copy())
                    st.rerun()

            if st.session_state.quiz_graded:
                st.markdown('<div class="grade-result">', unsafe_allow_html=True)
                st.markdown(st.session_state.quiz_graded)
                st.markdown('</div>', unsafe_allow_html=True)

                st.divider()
                st.markdown("##### 토론 계속하기")
                st.caption("채점 결과의 후속질문에 답하거나, 반론을 제기해보세요.")

                if len(st.session_state.discussion_messages) > 2:
                    for msg in st.session_state.discussion_messages[2:]:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                with st.form("discussion_form", clear_on_submit=True):
                    disc_input = st.text_input("", placeholder="토론 계속...", label_visibility="collapsed")
                    disc_cols = st.columns([5, 1])
                    with disc_cols[1]:
                        disc_submit = st.form_submit_button("전송", use_container_width=True)

                if disc_submit and disc_input:
                    st.session_state.discussion_messages.append({"role": "user", "content": disc_input})
                    with st.spinner("응답 중..."):
                        reply = chat_discussion(
                            st.session_state.discussion_messages,
                            briefing_text,
                            st.session_state.quiz_difficulty,
                        )
                    st.session_state.discussion_messages.append({"role": "assistant", "content": reply})
                    st.rerun()

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.quiz_current:
                if st.button("새 문제 출제", use_container_width=True):
                    st.session_state.quiz_current = None
                    st.session_state.quiz_graded = None
                    st.session_state.discussion_messages = []
                    st.rerun()
        with col2:
            if st.session_state.quiz_history:
                with st.expander(f"세션 히스토리 ({len(st.session_state.quiz_history)}문제)"):
                    for i, h in enumerate(reversed(st.session_state.quiz_history), 1):
                        score = extract_score(h.get("grade", ""))
                        badge = _score_badge(score)
                        st.markdown(f"**{i}.** {badge} [{h['difficulty']}]", unsafe_allow_html=True)
                        st.caption(h["question"][:100] + "..." if len(h["question"]) > 100 else h["question"])


# ═══════════════════════════════════════════════════════
# 종목 공부 모드
# ═══════════════════════════════════════════════════════
elif mode == "종목 공부":
    st.markdown('<p class="mode-header">STOCK DEEP DIVE</p>', unsafe_allow_html=True)

    # 종목 입력 + 분석 버튼
    input_col, btn_col = st.columns([4, 1])
    with input_col:
        stock_input = st.text_input(
            "종목명",
            placeholder="종목명 또는 티커를 입력하세요 (예: 삼성전자, AAPL, SK하이닉스)",
            label_visibility="collapsed",
            key="stock_input",
        )
    with btn_col:
        analyze_btn = st.button("종목 분석", type="primary", use_container_width=True)

    if analyze_btn and stock_input:
        with st.spinner(f"'{stock_input}' 종목 분석 중..."):
            result = analyze_stock(stock_input)
        st.session_state.stock_analysis = result
        st.session_state.stock_name = stock_input
        st.session_state.stock_qa_messages = []
        st.session_state.stock_quiz_current = None
        st.session_state.stock_quiz_graded = None
        st.session_state.stock_discussion_messages = []
        save_stock_study(stock_input, result)
        st.rerun()

    # 종목 추천 + 최근 히스토리 (분석 전 상태에서만)
    if not st.session_state.stock_analysis:
        # AI 추천 종목
        picks_col, hist_col = st.columns(2)

        with picks_col:
            st.markdown("**오늘의 주목 종목**")
            if "stock_picks" not in st.session_state:
                st.session_state.stock_picks = None

            if st.button("AI 추천 받기", key="get_picks", use_container_width=True):
                with st.spinner("시장 동향 분석 중..."):
                    st.session_state.stock_picks = get_stock_picks()
                st.rerun()

            if st.session_state.stock_picks:
                for i, pick in enumerate(st.session_state.stock_picks):
                    cat = pick.get("category", "")
                    tags = " ".join(f"`{t}`" for t in pick.get("tags", []))
                    with st.container():
                        pc1, pc2 = st.columns([4, 1])
                        with pc1:
                            st.markdown(f"**{pick['name']}** · {cat}")
                            st.caption(f"{pick.get('reason', '')}  {tags}")
                        with pc2:
                            if st.button("분석", key=f"pick_{i}", use_container_width=True):
                                with st.spinner(f"'{pick['name']}' 분석 중..."):
                                    result = analyze_stock(pick["name"])
                                st.session_state.stock_analysis = result
                                st.session_state.stock_name = pick["name"]
                                st.session_state.stock_qa_messages = []
                                st.session_state.stock_quiz_current = None
                                st.session_state.stock_quiz_graded = None
                                st.session_state.stock_discussion_messages = []
                                save_stock_study(pick["name"], result)
                                st.rerun()

        with hist_col:
            stock_hist = load_stock_history()
            if stock_hist:
                st.markdown("**최근 공부한 종목**")
                recent = list(reversed(stock_hist))[:6]
                for i, item in enumerate(recent):
                    if st.button(
                        f"{item['stock_name']}  ({item['date']})",
                        key=f"stock_hist_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.stock_analysis = item["analysis"]
                        st.session_state.stock_name = item["stock_name"]
                        st.session_state.stock_qa_messages = []
                        st.session_state.stock_quiz_current = None
                        st.session_state.stock_quiz_graded = None
                        st.session_state.stock_discussion_messages = []
                        st.rerun()

    # 분석 결과가 있을 때
    if st.session_state.stock_analysis:
        analysis = st.session_state.stock_analysis
        sname = st.session_state.stock_name

        if isinstance(analysis, dict) and "error" in analysis and not analysis.get("sections"):
            st.error(f"분석 실패: {analysis['error']}")
        else:
            # 서브모드 선택
            sub_modes = ["분석", "Q&A", "퀴즈"]
            sub = st.radio("", sub_modes, horizontal=True, label_visibility="collapsed", key="stock_sub_radio")
            st.divider()

            if sub == "분석":
                # 종목 정보 카드
                info = analysis.get("stock_info", {}) if isinstance(analysis, dict) else {}
                if info:
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    with ic1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-name">종목</div>
                            <div class="metric-value" style="font-size:18px;">{info.get('stock_name', sname)}</div>
                            <div class="metric-change metric-flat">{info.get('sector', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with ic2:
                        up = info.get("up")
                        color_cls = "metric-up" if up else "metric-down" if up is not None else "metric-flat"
                        arrow = "▲" if up else "▼" if up is not None else ""
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-name">현재가</div>
                            <div class="metric-value" style="font-size:18px;">{info.get('current_price', '-')}</div>
                            <div class="metric-change {color_cls}">{arrow} {info.get('change', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with ic3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-name">PER / PBR</div>
                            <div class="metric-value" style="font-size:18px;">{info.get('pe_ratio', '-')} / {info.get('pb_ratio', '-')}</div>
                            <div class="metric-change metric-flat">시총 {info.get('market_cap', '-')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with ic4:
                        consensus = info.get("consensus", "-")
                        target = info.get("target_price", "-")
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-name">컨센서스</div>
                            <div class="metric-value" style="font-size:18px;">{consensus}</div>
                            <div class="metric-change metric-flat">목표가 {target}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("")

                # 섹션별 탭
                sections = analysis.get("sections", {}) if isinstance(analysis, dict) else {}
                if sections:
                    tab_names = list(sections.keys())
                    tabs = st.tabs(tab_names)
                    for tab, name in zip(tabs, tab_names):
                        with tab:
                            st.markdown(f'<div class="section-card">', unsafe_allow_html=True)
                            st.markdown(sections[name])
                            st.markdown('</div>', unsafe_allow_html=True)
                elif isinstance(analysis, dict) and analysis.get("full_text"):
                    st.markdown(analysis["full_text"])

            elif sub == "Q&A":
                st.markdown(f"**{sname}**에 대해 궁금한 것을 물어보세요.")
                analysis_text = get_stock_analysis_text(analysis)

                container = st.container(height=450)
                with container:
                    if not st.session_state.stock_qa_messages:
                        st.caption("예시:")
                        st.caption(f"• {sname}의 현재 밸류에이션이 적정한가?")
                        st.caption(f"• {sname}의 최대 리스크 요인은?")
                        st.caption(f"• 경쟁사 대비 {sname}의 강점은?")
                    for msg in st.session_state.stock_qa_messages:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                with st.form("stock_qa_form", clear_on_submit=True):
                    sq_input = st.text_input("", placeholder="종목에 대해 질문...", label_visibility="collapsed")
                    sq_cols = st.columns([5, 1])
                    with sq_cols[1]:
                        sq_submit = st.form_submit_button("전송", use_container_width=True)

                if sq_submit and sq_input:
                    st.session_state.stock_qa_messages.append({"role": "user", "content": sq_input})
                    with st.spinner("답변 생성 중..."):
                        reply = chat_stock_qa(st.session_state.stock_qa_messages, analysis_text)
                    st.session_state.stock_qa_messages.append({"role": "assistant", "content": reply})
                    st.rerun()

                if st.session_state.stock_qa_messages:
                    if st.button("대화 초기화", key="stock_qa_clear", use_container_width=True):
                        st.session_state.stock_qa_messages = []
                        st.rerun()

            elif sub == "퀴즈":
                st.markdown(f"**{sname}** 관련 면접 질문으로 연습하세요.")
                analysis_text = get_stock_analysis_text(analysis)

                sq_col1, sq_col2, sq_col3 = st.columns([2, 2, 1])
                with sq_col1:
                    s_diff = st.selectbox(
                        "난이도",
                        options=list(DIFFICULTY_LEVELS.keys()),
                        format_func=lambda x: f"{x} — {DIFFICULTY_LEVELS[x]['description']}",
                        key="stock_quiz_diff",
                    )
                with sq_col3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    sq_btn = st.button("문제 출제", type="primary", use_container_width=True, key="stock_quiz_btn")

                if sq_btn:
                    with st.spinner("종목 면접 질문 생성 중..."):
                        quiz = generate_stock_quiz(analysis_text, sname, s_diff)
                    st.session_state.stock_quiz_current = quiz
                    st.session_state.stock_quiz_graded = None
                    st.session_state.stock_discussion_messages = []
                    st.rerun()

                if st.session_state.stock_quiz_current:
                    st.markdown('<div class="quiz-question">', unsafe_allow_html=True)
                    st.markdown(st.session_state.stock_quiz_current)
                    st.markdown('</div>', unsafe_allow_html=True)

                    if not st.session_state.stock_quiz_graded:
                        with st.form("stock_answer_form"):
                            s_answer = st.text_area(
                                "답변 작성", height=200,
                                placeholder="서술형으로 답변을 작성하세요.",
                            )
                            s_submit = st.form_submit_button("답변 제출", type="primary", use_container_width=True)

                        if s_submit and s_answer:
                            with st.spinner("채점 중..."):
                                s_grade = grade_stock_answer(
                                    analysis_text, sname, s_diff,
                                    st.session_state.stock_quiz_current, s_answer,
                                )
                            st.session_state.stock_quiz_graded = s_grade
                            st.session_state.stock_discussion_messages = [
                                {"role": "user", "content": f"[질문]\n{st.session_state.stock_quiz_current}\n\n[내 답변]\n{s_answer}"},
                                {"role": "assistant", "content": s_grade},
                            ]
                            entry = {
                                "type": "stock",
                                "stock_name": sname,
                                "difficulty": s_diff,
                                "question": st.session_state.stock_quiz_current,
                                "answer": s_answer,
                                "grade": s_grade,
                            }
                            save_quiz_result(entry)
                            st.rerun()

                    if st.session_state.stock_quiz_graded:
                        st.markdown('<div class="grade-result">', unsafe_allow_html=True)
                        st.markdown(st.session_state.stock_quiz_graded)
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.divider()
                        st.markdown("##### 토론 계속하기")

                        if len(st.session_state.stock_discussion_messages) > 2:
                            for msg in st.session_state.stock_discussion_messages[2:]:
                                with st.chat_message(msg["role"]):
                                    st.markdown(msg["content"])

                        with st.form("stock_disc_form", clear_on_submit=True):
                            sd_input = st.text_input("", placeholder="토론 계속...", label_visibility="collapsed")
                            sd_cols = st.columns([5, 1])
                            with sd_cols[1]:
                                sd_submit = st.form_submit_button("전송", use_container_width=True)

                        if sd_submit and sd_input:
                            st.session_state.stock_discussion_messages.append({"role": "user", "content": sd_input})
                            with st.spinner("응답 중..."):
                                briefing_text = get_briefing_text(st.session_state.briefing)
                                reply = chat_discussion(
                                    st.session_state.stock_discussion_messages,
                                    analysis_text,
                                    s_diff,
                                )
                            st.session_state.stock_discussion_messages.append({"role": "assistant", "content": reply})
                            st.rerun()

                    if st.session_state.stock_quiz_current:
                        if st.button("새 문제", key="stock_new_quiz", use_container_width=True):
                            st.session_state.stock_quiz_current = None
                            st.session_state.stock_quiz_graded = None
                            st.session_state.stock_discussion_messages = []
                            st.rerun()

        # 다른 종목 분석 버튼
        st.divider()
        if st.button("다른 종목 분석하기", use_container_width=True):
            st.session_state.stock_analysis = None
            st.session_state.stock_name = ""
            st.session_state.stock_qa_messages = []
            st.session_state.stock_quiz_current = None
            st.session_state.stock_quiz_graded = None
            st.session_state.stock_discussion_messages = []
            st.rerun()


# ═══════════════════════════════════════════════════════
# 복습 모드
# ═══════════════════════════════════════════════════════
elif mode == "복습":
    st.markdown('<p class="mode-header">WRONG ANSWER REVIEW</p>', unsafe_allow_html=True)

    wrong = get_wrong_questions(threshold=6)
    all_history = load_quiz_history()

    # 상단 통계
    total = len(all_history)
    wrong_count = len(wrong)
    avg_score = sum(h["score"] for h in all_history if h.get("score") is not None) / max(1, len([h for h in all_history if h.get("score") is not None]))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("총 문제 수", f"{total}문제")
    with c2:
        st.metric("틀린 문제 (6점 이하)", f"{wrong_count}문제")
    with c3:
        st.metric("평균 점수", f"{avg_score:.1f}/10")

    st.divider()

    if not wrong:
        if total == 0:
            st.info("아직 풀어본 퀴즈가 없습니다. '퀴즈/토론' 탭에서 문제를 풀어보세요.")
        else:
            st.success("틀린 문제가 없습니다! 모든 문제에서 7점 이상을 받았어요.")
    else:
        st.markdown(f"**6점 이하 문제 {wrong_count}개** — 선택해서 다시 도전해보세요.")
        st.markdown("")

        for i, item in enumerate(reversed(wrong)):
            score = item.get("score", "?")
            date_str = item.get("date", "")
            diff = item.get("difficulty", "")

            with st.container():
                st.markdown(f'<div class="review-card">', unsafe_allow_html=True)

                hcol1, hcol2, hcol3 = st.columns([1, 1, 3])
                with hcol1:
                    st.markdown(_score_badge(score), unsafe_allow_html=True)
                with hcol2:
                    st.caption(f"{diff} · {date_str}")

                # 질문 (축약)
                q = item.get("question", "")
                st.markdown(q[:200] + "..." if len(q) > 200 else q)

                with st.expander("내 답변 & 채점 보기"):
                    st.markdown("**내 답변:**")
                    st.markdown(item.get("answer", ""))
                    st.divider()
                    st.markdown("**채점 결과:**")
                    st.markdown(item.get("grade", ""))

                # 재도전 버튼
                if st.button(f"이 문제 재도전", key=f"retry_{i}", use_container_width=True):
                    st.session_state.review_target = item
                    st.session_state.review_graded = None
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

        # 재도전 영역
        if st.session_state.review_target:
            st.divider()
            st.markdown("### 재도전")
            target = st.session_state.review_target

            st.markdown('<div class="quiz-question">', unsafe_allow_html=True)
            st.markdown(target["question"])
            st.markdown('</div>', unsafe_allow_html=True)

            if not st.session_state.review_graded:
                with st.form("review_answer_form"):
                    new_answer = st.text_area(
                        "다시 답변 작성",
                        height=200,
                        placeholder="이번에는 모범답안을 참고해서 더 나은 답변을 작성해보세요.",
                    )
                    retry_submit = st.form_submit_button("답변 제출", type="primary", use_container_width=True)

                if retry_submit and new_answer:
                    briefing_text = get_briefing_text(st.session_state.briefing)
                    with st.spinner("채점 중..."):
                        new_grade = grade_answer(
                            briefing_text,
                            target.get("difficulty", "기초"),
                            target["question"],
                            new_answer,
                        )
                    st.session_state.review_graded = new_grade
                    # 재도전 결과도 저장
                    save_quiz_result({
                        "difficulty": target.get("difficulty", "기초"),
                        "question": target["question"],
                        "answer": new_answer,
                        "grade": new_grade,
                        "is_retry": True,
                    })
                    st.rerun()

            if st.session_state.review_graded:
                new_score = extract_score(st.session_state.review_graded)
                old_score = target.get("score", 0)

                if new_score is not None and old_score is not None and new_score > old_score:
                    st.balloons()
                    st.success(f"점수 향상! {old_score}/10 → {new_score}/10")
                elif new_score is not None:
                    st.info(f"이번 점수: {new_score}/10 (이전: {old_score}/10)")

                st.markdown('<div class="grade-result">', unsafe_allow_html=True)
                st.markdown(st.session_state.review_graded)
                st.markdown('</div>', unsafe_allow_html=True)

                if st.button("복습 목록으로 돌아가기", use_container_width=True):
                    st.session_state.review_target = None
                    st.session_state.review_graded = None
                    st.rerun()
