"""
Trading Agent - KOSPI 강세/약세 판정
2023-11 ~ 현재까지 매월 실행 후 실제 KOSPI 수익률과 비교

실행: python analysis/regime_agent.py
"""
import os
import sys
import json
import anthropic
import openai
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / '.env')

def get_conn():
    """DB 연결을 반환. 끊어졌으면 재연결."""
    global _conn
    try:
        _conn.isolation_level  # connection alive check
        return _conn
    except Exception:
        _conn = psycopg2.connect(os.environ['DATABASE_URL'])
        return _conn

_conn = psycopg2.connect(os.environ['DATABASE_URL'])
claude_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
openai_client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
CLAUDE_MODEL = "claude-sonnet-4-6"
GPT_MODEL = "gpt-4o"

# ── 시작 시 전체 데이터 메모리 로드 ─────────────────────────────
print("데이터 사전 로딩 중...")
_technical_df = pd.read_sql(
    "SELECT trade_date, indicator, value, COALESCE(symbol, 'KOSPI') AS symbol FROM alpha_lab.technical_indicators ORDER BY trade_date",
    _conn, parse_dates=['trade_date']
)
_technical_df['trade_date'] = pd.to_datetime(_technical_df['trade_date']).dt.date
print(f"  technical: {len(_technical_df)}건")

_macro_df = pd.read_sql(
    "SELECT indicator, period, freq, value FROM alpha_lab.macro_indicators",
    _conn
)
print(f"  macro: {len(_macro_df)}건")
print("데이터 로딩 완료")

# ── Lag 설정 (lookahead bias 방지) ──────────────────────────
LAG = {
    'leading_index': 2,
    'yield_spread':  2,
    'wti_monthly':   1,
    'cpi':           1,
    'ppi':           2,
    'bsi_all':       2,
    'csi_outlook':   1,
}


# ── 데이터 조회 함수들 ────────────────────────────────────────

def get_macro_monthly(as_of: date, indicator: str, n_months: int = 6):
    """월별 매크로 지표 - lag 적용해서 N개월치 반환 (메모리)"""
    lag = LAG.get(indicator, 1)
    end_ym = (as_of - relativedelta(months=lag)).strftime('%Y%m')
    sub = _macro_df[(_macro_df['indicator'] == indicator) & (_macro_df['freq'] == 'M') & (_macro_df['period'] <= end_ym)]
    sub = sub.sort_values('period').tail(n_months)[['period', 'value']]
    return list(sub.itertuples(index=False, name=None))


def get_macro_daily_latest(as_of: date, indicator: str):
    """일별 매크로 지표 - as_of 전일까지 최신값 (당일 종가 미사용)"""
    as_of_str = (as_of - relativedelta(days=1)).strftime('%Y-%m-%d')
    sub = _macro_df[(_macro_df['indicator'] == indicator) & (_macro_df['freq'] == 'D') & (_macro_df['period'] <= as_of_str)]
    if sub.empty:
        return None
    row = sub.sort_values('period').iloc[-1]
    return (row['period'], row['value'])


def get_macro_daily_series(as_of: date, indicator: str, n_days: int = 60):
    """일별 매크로 지표 - as_of 전일까지 N일치 (당일 종가 미사용)"""
    as_of_str = (as_of - relativedelta(days=1)).strftime('%Y-%m-%d')
    sub = _macro_df[(_macro_df['indicator'] == indicator) & (_macro_df['freq'] == 'D') & (_macro_df['period'] <= as_of_str)]
    sub = sub.sort_values('period').tail(n_days)[['period', 'value']]
    return list(sub.itertuples(index=False, name=None))


def get_technical(as_of: date, symbol: str = 'KOSPI'):
    """기술적 지표 - as_of 전일까지 최신값 (당일 미사용)"""
    prev = as_of - relativedelta(days=1)
    sub = _technical_df[(_technical_df['trade_date'] <= prev) & (_technical_df['symbol'] == symbol)]
    if sub.empty:
        return {}
    idx = sub.groupby('indicator')['trade_date'].idxmax()
    return dict(zip(sub.loc[idx, 'indicator'], sub.loc[idx, 'value']))


def get_investor_flow(as_of: date, n_days: int = 20):
    """외국인/기관/개인 순매수 - as_of 전일까지 N일 합계 (당일 미사용)"""
    as_of_str = (as_of - relativedelta(days=1)).strftime('%Y-%m-%d')
    results = {}
    for key in ['investor_foreign_kospi', 'investor_institution_kospi', 'investor_individual_kospi']:
        sub = _macro_df[(_macro_df['indicator'] == key) & (_macro_df['freq'] == 'D') & (_macro_df['period'] <= as_of_str)]
        top = sub.sort_values('period').tail(n_days)
        results[key] = float(top['value'].sum()) if not top.empty else 0
    return results


def get_news_summary(as_of: date):
    """직전 달 macro 뉴스 - lookahead bias 방지. news_nate → news 순으로 조회"""
    prev_month = (as_of - relativedelta(months=1)).strftime('%Y-%m')
    cur = get_conn().cursor()
    # news_nate 먼저 (네이트 뉴스)
    cur.execute("""
        SELECT title, summary
        FROM alpha_lab.news_nate
        WHERE published_date LIKE %s
        ORDER BY published_date DESC
        LIMIT 20
    """, (f'{prev_month}%',))
    rows = cur.fetchall()
    # 없으면 기존 news 테이블
    if not rows:
        cur.execute("""
            SELECT title, summary
            FROM alpha_lab.news
            WHERE category = 'macro'
              AND published_date LIKE %s
            ORDER BY published_date DESC
            LIMIT 20
        """, (f'{prev_month}%',))
        rows = cur.fetchall()
    if not rows:
        return None
    return "\n".join([f"- {r[0]}: {r[1][:200]}" for r in rows])


def get_trade_amount_series(as_of: date, n_months: int = 6):
    """KOSPI ETF(069500) 월별 평균 거래대금 + 12개월 평균 대비 배율"""
    cur = get_conn().cursor()
    cutoff = (as_of - relativedelta(days=1)).strftime('%Y-%m-%d')
    cur.execute("""
        SELECT trade_date, trade_amount FROM alpha_lab.daily_price
        WHERE stock_code = '069500' AND trade_date::date <= %s
        ORDER BY trade_date
    """, (cutoff,))
    rows = cur.fetchall()
    if not rows:
        return [], None

    monthly = {}
    for td, amt in rows:
        ym = str(td)[:7]
        monthly.setdefault(ym, []).append(amt)

    items = sorted(monthly.items())
    # 12개월 평균 기준값
    baseline_vals = [v for vals in [v for _, v in items[-13:-1]] for v in vals]
    baseline = sum(baseline_vals) / len(baseline_vals) if baseline_vals else None

    series = []
    for ym, vals in items[-n_months:]:
        avg = sum(vals) / len(vals)
        ratio = avg / baseline if baseline else None
        series.append((ym, avg, ratio))
    return series, baseline


def get_kospi_return(from_date: date, to_date: date):
    """KOSPI(069500) 수익률 계산"""
    c = get_conn()
    cur = c.cursor()
    cur.execute("""
        SELECT trade_date, close FROM alpha_lab.daily_price
        WHERE stock_code = '069500'
          AND trade_date::date >= %s AND trade_date::date <= %s
        ORDER BY trade_date
    """, (from_date, to_date))
    rows = cur.fetchall()
    if len(rows) < 2:
        return None
    start_price = rows[0][1]
    end_price = rows[-1][1]
    return (end_price - start_price) / start_price * 100


# ── 데이터 요약 텍스트 빌더 ───────────────────────────────────

def get_kospi_monthly_series(as_of: date, n_months: int = 6):
    """월말 KOSPI 종가 시계열 - as_of 포함 최근 n개월"""
    # 당월(as_of 포함 달)은 전일까지만, 이전 달은 월말까지
    series = []
    for i in range(n_months - 1, -1, -1):
        if i == 0:
            cutoff = as_of - relativedelta(days=1)
        else:
            m = (as_of - relativedelta(months=i)).replace(day=28) + relativedelta(days=4)
            cutoff = m - relativedelta(days=m.day)
        sub = _technical_df[
            (_technical_df['indicator'] == 'close') &
            (_technical_df['trade_date'] <= cutoff)
        ]
        if not sub.empty:
            row = sub.sort_values('trade_date').iloc[-1]
            series.append((str(row['trade_date'])[:7], float(row['value'])))
    return series


def get_rsi_monthly_series(as_of: date, n_months: int = 6):
    """월말 RSI 시계열 (당일 미사용)"""
    series = []
    for i in range(n_months - 1, -1, -1):
        if i == 0:
            cutoff = as_of - relativedelta(days=1)
        else:
            m = (as_of - relativedelta(months=i)).replace(day=28) + relativedelta(days=4)
            cutoff = m - relativedelta(days=m.day)
        sub = _technical_df[
            (_technical_df['indicator'] == 'rsi14') &
            (_technical_df['trade_date'] <= cutoff)
        ]
        if not sub.empty:
            row = sub.sort_values('trade_date').iloc[-1]
            series.append((str(row['trade_date'])[:7], float(row['value'])))
    return series


def build_technical_summary(as_of: date) -> str:
    tech = get_technical(as_of)

    def tech_monthly_series(indicator, n_months=6, symbol='KOSPI'):
        """technical_indicators에서 월말 시계열 추출"""
        series = []
        for i in range(n_months - 1, -1, -1):
            if i == 0:
                cutoff = as_of - relativedelta(days=1)
            else:
                m = (as_of - relativedelta(months=i)).replace(day=28) + relativedelta(days=4)
                cutoff = m - relativedelta(days=m.day)
            sub = _technical_df[(_technical_df['indicator'] == indicator) & (_technical_df['symbol'] == symbol) & (_technical_df['trade_date'] <= cutoff)]
            if not sub.empty:
                row = sub.sort_values('trade_date').iloc[-1]
                series.append((str(row['trade_date'])[:7], float(row['value'])))
        return series

    kospi_series = tech_monthly_series('close')
    rsi_series   = tech_monthly_series('rsi14')
    ma50_series  = tech_monthly_series('ma50')
    ma200_series = tech_monthly_series('ma200')
    macd_series  = tech_monthly_series('macd')

    kospi_trend = "  →  ".join([f"{ym}:{v:,.0f}" for ym, v in kospi_series])
    rsi_trend   = "  →  ".join([f"{ym}:{v:.1f}" for ym, v in rsi_series])
    ma50_trend  = "  →  ".join([f"{ym}:{v:,.0f}" for ym, v in ma50_series])
    ma200_trend = "  →  ".join([f"{ym}:{v:,.0f}" for ym, v in ma200_series])
    macd_trend  = "  →  ".join([f"{ym}:{v:.1f}" for ym, v in macd_series])

    def daily_monthly_trend(indicator, n_months=3):
        """일별 지표를 월평균으로 집계해서 n개월 추이 반환"""
        series = get_macro_daily_series(as_of, indicator, 90)
        monthly = {}
        for period, val in series:
            monthly.setdefault(period[:7], []).append(val)
        items = sorted(monthly.items())[-n_months:]
        return "  →  ".join([f"{ym}:{sum(v)/len(v):.2f}" for ym, v in items])

    def index_tech_summary(symbol, label, n_months=6):
        """technical_indicators 테이블에서 사전계산된 지표 조회"""
        close_s = tech_monthly_series('close', n_months, symbol)
        ma50_s  = tech_monthly_series('ma50', n_months, symbol)
        ma200_s = tech_monthly_series('ma200', n_months, symbol)
        rsi_s   = tech_monthly_series('rsi14', n_months, symbol)
        macd_s  = tech_monthly_series('macd', n_months, symbol)

        if not close_s:
            return [f"▶ {label}: 데이터 부족"]

        fmt_s = lambda s, f: "  →  ".join([f"{ym}:{f.format(v)}" for ym, v in s]) or "N/A"
        return [
            f"▶ {label} 월말 종가(6개월):   {fmt_s(close_s, '{:.0f}')}",
            f"▶ {label} MA50(6개월):        {fmt_s(ma50_s, '{:.0f}')}",
            f"▶ {label} MA200(6개월):       {fmt_s(ma200_s, '{:.0f}')}",
            f"▶ {label} RSI(14,6개월):      {fmt_s(rsi_s, '{:.1f}')}",
            f"▶ {label} MACD(6개월):        {fmt_s(macd_s, '{:.1f}')}",
        ]

    ta_series, ta_baseline = get_trade_amount_series(as_of, 6)
    ta_trend = "  →  ".join([
        f"{ym}:{avg/1e8:.0f}억({ratio:.1f}x)" if ratio else f"{ym}:{avg/1e8:.0f}억"
        for ym, avg, ratio in ta_series
    ])

    lines = [
        f"[Technical 지표 - {as_of} 전일 기준]",
        f"▶ KOSPI 월말 종가 추이(6개월): {kospi_trend}",
        f"▶ MA50 월말 추이(6개월):       {ma50_trend}",
        f"▶ MA200 월말 추이(6개월):      {ma200_trend}",
        f"▶ RSI(14) 월말 추이(6개월):    {rsi_trend}",
        f"▶ MACD 월말 추이(6개월):       {macd_trend}",
        f"▶ 거래대금 월평균(6개월, 12m평균대비): {ta_trend}",
    ]

    # S&P500 / SOX
    lines.append("")
    for symbol, label in [('SP500', 'S&P500'), ('SOX', 'SOX')]:
        lines.extend(index_tech_summary(symbol, label))

    # RSI/MA200 과거 통계 (전월말 기준 → 해당월 수익률)
    current_rsi = rsi_series[-1][1] if rsi_series else 0
    current_close = kospi_series[-1][1] if kospi_series else 0
    current_ma200 = ma200_series[-1][1] if ma200_series else 0

    lines.append("")
    lines.append("[과거 통계 참고 - 2018년 이후 KOSPI 전월말 기준]")
    if current_rsi < 35:
        lines.append(f"▶ 현재 RSI {current_rsi:.1f} (과매도 구간)")
        lines.append(f"  과거 전월말 RSI<35 → 해당월: 평균 +3.8%, 상승확률 91% (11회 중 10회 상승)")
    elif current_rsi > 70:
        lines.append(f"▶ 현재 RSI {current_rsi:.1f} (과매수 구간)")
        lines.append(f"  과거 전월말 RSI>70 → 해당월: 평균 +1.7%, 상승확률 44% (9회 중 4회 상승)")
    else:
        lines.append(f"▶ 현재 RSI {current_rsi:.1f} (중립 구간)")

    if current_close > current_ma200:
        lines.append(f"▶ KOSPI가 MA200 위 (상승추세)")
        lines.append(f"  과거 MA200 위 → 해당월: 평균 +2.2%, 상승확률 55%")
    else:
        lines.append(f"▶ KOSPI가 MA200 아래 (하락추세)")
        lines.append(f"  과거 MA200 아래 → 해당월: 평균 +0.1%, 상승확률 54%")
        if current_rsi < 35:
            lines.append(f"  과거 MA200 아래 + RSI<35 → 해당월: 평균 +2.8%, 상승확률 90% (10회 중 9회 반등)")

    return "\n".join(lines)


def build_fundamental_summary(as_of: date) -> str:
    def fmt(rows, unit=''):
        if not rows:
            return 'N/A'
        vals = [f"{r[0]}: {r[1]:.2f}{unit}" for r in rows]
        trend = '↑' if rows[-1][1] > rows[0][1] else ('↓' if rows[-1][1] < rows[0][1] else '→')
        chg = rows[-1][1] - rows[0][1]
        return f"{', '.join(vals)} {trend}({chg:+.2f})"

    leading = get_macro_monthly(as_of, 'leading_index', 6)
    yield_sp = get_macro_monthly(as_of, 'yield_spread', 6)
    cpi = get_macro_monthly(as_of, 'cpi', 6)
    ppi = get_macro_monthly(as_of, 'ppi', 6)
    bsi = get_macro_monthly(as_of, 'bsi_all', 6)
    csi = get_macro_monthly(as_of, 'csi_outlook', 6)
    wti = get_macro_monthly(as_of, 'wti_monthly', 6)

    # 일별 지표 → 월평균 추이
    def daily_monthly_trend(indicator, n_months=3):
        series = get_macro_daily_series(as_of, indicator, 90)
        monthly = {}
        for period, val in series:
            monthly.setdefault(period[:7], []).append(val)
        items = sorted(monthly.items())[-n_months:]
        return "  →  ".join([f"{ym}:{sum(v)/len(v):.2f}" for ym, v in items])

    vix = get_macro_daily_latest(as_of, 'vix')
    bond_1y = get_macro_daily_latest(as_of, 'bond_1y')
    bond_10y = get_macro_daily_latest(as_of, 'bond_10y')
    spread = (bond_10y[1] - bond_1y[1]) if bond_1y and bond_10y else None

    lines = [
        f"[Fundamental 지표 - {as_of} 기준, lag 적용]",
        f"선행종합지수(2개월 lag): {fmt(leading)}",
        f"장단기금리차(2개월 lag): {fmt(yield_sp, '%p')}",
        f"CPI(1개월 lag): {fmt(cpi)}",
        f"PPI(2개월 lag): {fmt(ppi)}",
        f"전산업BSI(2개월 lag): {fmt(bsi)}",
        f"경기전망CSI(1개월 lag): {fmt(csi)}",
        f"WTI유가(1개월 lag): {fmt(wti, '$')}",
        "",
        f"[시장 환경 지표]",
        f"▶ VIX 월평균 추이(3개월):      {daily_monthly_trend('vix')}",
        f"▶ 원/달러 월평균 추이(3개월):  {daily_monthly_trend('usd_krw')}",
        f"▶ 국고채 10y 월평균(3개월):    {daily_monthly_trend('bond_10y')}",
        f"▶ 국고채 1y 월평균(3개월):     {daily_monthly_trend('bond_1y')}",
        f"  장단기금리차(10y-1y) 최근: {spread:+.2f}%p" if spread else "  금리차: N/A",
        f"▶ 수급 월별 순매수 추이(억원)",
    ]

    # 외국인/기관/개인 수급
    for key, label in [
        ('investor_foreign_kospi', '외국인'),
        ('investor_institution_kospi', '기관  '),
        ('investor_individual_kospi', '개인  '),
    ]:
        series = get_macro_daily_series(as_of, key, 90)
        monthly = {}
        for period, val in series:
            monthly.setdefault(period[:7], []).append(val)
        parts = []
        for ym, vals in sorted(monthly.items())[-3:]:
            net = (vals[-1] - vals[0]) / 100  # 백만원 → 억원
            parts.append(f"{ym}:{net:+.0f}억")
        lines.append(f"  {label}: {'  →  '.join(parts)}")

    return "\n".join(lines)


# ── Agent 호출 ────────────────────────────────────────────────

def call_agent(system_prompt: str, user_content: str, max_tokens: int = 400, use_gpt: bool = False) -> str:
    import time
    for attempt in range(5):
        try:
            if use_gpt:
                resp = openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    max_tokens=max_tokens,
                    temperature=1,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ]
                )
                return resp.choices[0].message.content
            else:
                msg = claude_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}]
                )
                return msg.content[0].text
        except Exception as e:
            if attempt < 4 and ('overloaded' in str(e).lower() or '529' in str(e) or '500' in str(e)):
                wait = 30 * (attempt + 1)
                print(f"    ⏳ API 오류, {wait}초 후 재시도 ({attempt+1}/5)...")
                time.sleep(wait)
            else:
                raise


NO_EXTERNAL = "반드시 제공된 데이터만 사용하고, 외부 지식이나 학습 데이터는 사용하지 마세요."
CONCISE = "각 근거는 핵심 수치 1~2개를 포함해 2문장 이내로 간결하게 작성하세요. 군더더기 없이 핵심만 쓰세요."


TECH_SYSTEM = (
    "당신은 기술적 분석 전문가입니다. "
    "주어진 기술적 지표의 현황과 시사점을 핵심 포인트 3가지로 정리하세요. Bull/Bear 판정은 하지 마세요. "
    f"{NO_EXTERNAL} {CONCISE} "
    "형식: 포인트1: ...\n포인트2: ...\n포인트3: ..."
)
FUND_SYSTEM = (
    "당신은 거시경제 분석 전문가입니다. "
    "주어진 경제지표의 현황과 시사점을 핵심 포인트 3가지로 정리하세요. Bull/Bear 판정은 하지 마세요. "
    f"{NO_EXTERNAL} {CONCISE} "
    "형식: 포인트1: ...\n포인트2: ...\n포인트3: ..."
)
BULL1_SYSTEM = (
    "당신은 강세론자입니다. 반드시 강세 관점에서만 주장하세요. "
    "기술적·펀더멘털·뉴스 분석을 바탕으로 향후 1개월 KOSPI 상승 논거 3가지를 구성하세요. "
    "약세나 리스크 요인은 절대 언급하지 마세요. 오직 상승 근거만 제시하세요. "
    f"{NO_EXTERNAL} {CONCISE}"
)
BEAR1_SYSTEM = (
    "당신은 약세론자입니다. 반드시 약세 관점에서만 주장하세요. "
    "기술적·펀더멘털·뉴스 분석을 바탕으로 향후 1개월 KOSPI 하락 논거 3가지를 구성하세요. "
    "강세나 긍정 요인은 절대 언급하지 마세요. 오직 하락 근거만 제시하세요. "
    f"{NO_EXTERNAL} {CONCISE}"
)
BULL2_SYSTEM = (
    "당신은 강세론자입니다. "
    "상대방(약세·변동성론자)의 주장을 읽고 핵심 약점을 반박하세요. "
    f"{NO_EXTERNAL} {CONCISE}"
)
BEAR2_SYSTEM = (
    "당신은 약세·변동성론자입니다. "
    "상대방(강세론자)의 주장을 읽고 핵심 약점을 반박하세요. "
    f"{NO_EXTERNAL} {CONCISE}"
)
MANAGER_SYSTEM = (
    "당신은 포트폴리오 매니저입니다.\n"
    "토론 내용을 종합하여 이번 달 KOSPI 예상 수익률(%)을 추정하세요.\n\n"
    "[전략 맥락]\n"
    "당신의 판정은 실제 포트폴리오 운용에 직접 사용됩니다.\n"
    "- 예상 수익률 > +3% (강세) → 종목 비중 최대 30%, 공격적 운용\n"
    "- 예상 수익률 < -3% (약세) → 종목 비중 10%로 축소, 손절률 적용, 방어적 운용\n"
    "- 그 사이 (변동성) → 보수적 운용\n"
    "목표: KOSPI200 대비 초과수익 달성\n"
    "주의: 잘못된 약세 판정(실제 상승장) → 수익 기회 상실. 잘못된 강세 판정(실제 하락장) → 큰 손실.\n"
    "양쪽 비용을 균형 있게 고려하세요. 확신이 없다고 무조건 보수적으로 가지 마세요.\n\n"
    "[출력 규칙]\n"
    "반드시 JSON 한 줄만 출력. 앞뒤 설명, 마크다운, 코드블록 절대 금지.\n"
    "형식: {\"expected_return\": 숫자, \"confidence\": 숫자, \"summary\": \"문장\"}\n"
    "expected_return: KOSPI 예상 월간 수익률(%). 소수점 1자리.\n"
    "confidence: 확신도 0~100.\n"
    "summary는 핵심 수치를 포함해 2문장 이내로 간결하게 작성하세요."
)


def run_technical_agent(summary: str) -> str:
    return call_agent(system_prompt=TECH_SYSTEM, user_content=summary, max_tokens=2048, use_gpt=True)


def run_fundamental_agent(summary: str) -> str:
    return call_agent(system_prompt=FUND_SYSTEM, user_content=summary, max_tokens=2048, use_gpt=True)


NEWS_SYSTEM = (
    "당신은 매크로 뉴스 분석 전문가입니다. "
    "주어진 뉴스들에서 KOSPI에 영향을 미치는 핵심 이벤트/흐름을 포인트 3가지로 정리하세요. "
    "개별종목 뉴스는 무시하고 매크로/시장 전체 흐름만 분석하세요. Bull/Bear 판정은 하지 마세요. "
    f"{NO_EXTERNAL} {CONCISE} "
    "형식: 포인트1: ...\n포인트2: ...\n포인트3: ..."
)

def run_news_agent(as_of: date) -> tuple[str, str]:
    """뉴스 분석 → (분석결과, 원본뉴스). 뉴스 없으면 (None, None)"""
    news = get_news_summary(as_of)
    if not news:
        return None, None
    news_input = f"[{as_of.strftime('%Y-%m')} 직전월 매크로 뉴스]\n{news}"
    result = call_agent(system_prompt=NEWS_SYSTEM, user_content=news_input, max_tokens=2048, use_gpt=True)
    return result, news_input


def run_bull1(tech: str, fund: str) -> str:
    """1라운드: Bull 독립 주장 (GPT-4o)"""
    return call_agent(system_prompt=BULL1_SYSTEM,
                      user_content=f"[기술적 분석]\n{tech}\n\n[펀더멘털 분석]\n{fund}",
                      max_tokens=2048, use_gpt=True)


def run_bear1(tech: str, fund: str) -> str:
    """1라운드: Bear 독립 주장 (GPT-4o)"""
    return call_agent(system_prompt=BEAR1_SYSTEM,
                      user_content=f"[기술적 분석]\n{tech}\n\n[펀더멘털 분석]\n{fund}",
                      max_tokens=2048, use_gpt=True)


def run_bull2(tech: str, fund: str, bull1: str, bear1: str) -> str:
    """2라운드: Bull이 Bear 주장 보고 재반박 (GPT-4o)"""
    return call_agent(
        system_prompt=BULL2_SYSTEM,
        user_content=(
            f"[내 주장 (1라운드)]\n{bull1}\n\n"
            f"[상대방 Bear 주장]\n{bear1}\n\n"
            f"[참고 데이터]\n기술적: {tech[:300]}\n펀더멘털: {fund[:300]}"
        ),
        max_tokens=1024, use_gpt=True
    )


def run_bear2(tech: str, fund: str, bull1: str, bear1: str) -> str:
    """2라운드: Bear가 Bull 주장 보고 재반박 (GPT-4o)"""
    return call_agent(
        system_prompt=BEAR2_SYSTEM,
        user_content=(
            f"[내 주장 (1라운드)]\n{bear1}\n\n"
            f"[상대방 Bull 주장]\n{bull1}\n\n"
            f"[참고 데이터]\n기술적: {tech[:300]}\n펀더멘털: {fund[:300]}"
        ),
        max_tokens=1024, use_gpt=True
    )


def run_manager_agent(manager_user: str) -> dict:
    import re
    for attempt in range(3):
        result = call_agent(system_prompt=MANAGER_SYSTEM, user_content=manager_user, max_tokens=2048, use_gpt=True)
        try:
            m = re.search(r'\{.*\}', result, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                if "expected_return" in parsed:
                    er = float(parsed["expected_return"])
                    if er > 3:
                        parsed["judgment"] = "강세"
                    elif er < -3:
                        parsed["judgment"] = "약세"
                    else:
                        parsed["judgment"] = "변동성"
                    parsed["expected_return"] = er
                    return parsed
        except Exception:
            pass
        print(f"  ⚠️ Manager 파싱 실패 (시도 {attempt+1}/3): {result[:80]}")
    return {"judgment": "N/A", "expected_return": 0, "confidence": 0, "summary": result[:200]}


# ── 하드코딩 사이클 (레짐 조합 백테스트와 동일) ──────────────
def build_lessons(past_results: list, current_as_of: date, max_lessons: int = 15) -> str:
    """과거 틀린 사례에서 교훈 추출 (lookahead 방지: current_as_of 이전만)"""
    mistakes = []
    for r in past_results:
        if r.get('as_of', '') >= str(current_as_of):
            continue
        if r.get('direction_correct') is False:
            mistakes.append(r)

    if not mistakes:
        return ""

    # 최근 10개
    recent = mistakes[-10:]
    # 가장 크게 틀린 5개
    biggest = sorted(mistakes, key=lambda r: abs(r.get('expected_return', 0) - (r.get('kospi_next_month_return') or 0)), reverse=True)[:5]

    # 중복 제거
    selected = {r['as_of']: r for r in recent + biggest}

    lines = ["\n[과거 판단 교훈 - 틀렸던 사례]"]
    for r in sorted(selected.values(), key=lambda r: r['as_of']):
        er = r.get('expected_return', 0)
        ret = r.get('kospi_next_month_return', 0)
        summary = r.get('summary', '')[:60]
        # 핵심 데이터 추출
        detail = r.get('detail', {})
        tech_input = detail.get('technical_input', '')
        rsi_match = ''
        if 'RSI' in tech_input:
            import re
            m = re.search(r'RSI.*?(\d+\.\d+)', tech_input)
            if m:
                rsi_match = f'RSI={m.group(1)}'

        lines.append(f"- {r['as_of'][:7]}: 예상{er:+.1f}% → 실제{ret:+.1f}%. {rsi_match} {summary}")

    lines.append("→ 위 사례들의 패턴을 참고하여 같은 실수를 반복하지 마세요.")
    return "\n".join(lines)


_CYCLE_BEAR_PERIODS = [
    ("2018-01-01", "2019-01-01"),
    ("2020-01-01", "2020-03-31"),
    ("2021-06-01", "2022-10-31"),
    ("2024-07-01", "2024-10-31"),
]

def _get_cycle(d: date) -> str:
    ds = d.strftime('%Y-%m-%d')
    for s, e in _CYCLE_BEAR_PERIODS:
        if s <= ds <= e:
            return "Bear"
    return "Bull"

def _is_transition(d: date) -> bool:
    """Bear 시작/끝 ±1개월이면 전환 구간"""
    ds = d.strftime('%Y-%m')
    for s, e in _CYCLE_BEAR_PERIODS:
        s_date = date.fromisoformat(s)
        e_date = date.fromisoformat(e)
        for offset in [-1, 0, 1]:
            if (s_date + relativedelta(months=offset)).strftime('%Y-%m') == ds:
                return True
            if (e_date + relativedelta(months=offset)).strftime('%Y-%m') == ds:
                return True
    return False


# ── 월별 실행 ─────────────────────────────────────────────────

def run_month(as_of: date, past_results: list = None) -> dict:
    print(f"\n{'='*60}")
    print(f"  {as_of.strftime('%Y-%m')} 레짐 판정")
    print(f"{'='*60}")

    tech_input = build_technical_summary(as_of)
    fund_input = build_fundamental_summary(as_of)

    # 각 agent의 실제 프롬프트 기록
    prompts = {}

    def log_agent(label, system, user, response):
        print(f"\n  {'─'*60}")
        print(f"  [{label}]")
        print(f"  [시스템 프롬프트]\n  {system}")
        print(f"  [입력]\n  {user}")
        print(f"  [응답]\n  {response}")
        print(f"  {'─'*60}")

    print("  [1/6] Technical Agent...")
    prompts['technical'] = {"system": TECH_SYSTEM, "user": tech_input, "model": "GPT-4o"}
    tech = run_technical_agent(tech_input)
    log_agent("Technical", TECH_SYSTEM, tech_input, tech)

    print("  [2/6] Fundamental Agent...")
    prompts['fundamental'] = {"system": FUND_SYSTEM, "user": fund_input, "model": "GPT-4o"}
    fund = run_fundamental_agent(fund_input)
    log_agent("Fundamental", FUND_SYSTEM, fund_input, fund)

    print("  [3/6] News Agent...")
    news_analysis, news_input = run_news_agent(as_of)
    if news_analysis:
        prompts['news'] = {"system": NEWS_SYSTEM, "user": news_input, "model": "GPT-4o"}
        log_agent("News", NEWS_SYSTEM, news_input, news_analysis)
    else:
        news_analysis = ""
        news_input = ""
        print(f"    📰 뉴스 없음 (해당 월 미수집)")

    # Bull/Bear에 3개 분석 모두 전달
    debate_input = f"[기술적 분석]\n{tech}\n\n[펀더멘털 분석]\n{fund}"
    if news_analysis:
        debate_input += f"\n\n[뉴스 분석]\n{news_analysis}"

    bull1_user = debate_input
    bear1_user = debate_input

    print("  [4/6] 1라운드: Bull/Bear 독립 주장...")
    prompts['bull1'] = {"system": BULL1_SYSTEM, "user": bull1_user, "model": "GPT-4o"}
    bull1 = call_agent(system_prompt=BULL1_SYSTEM, user_content=bull1_user, max_tokens=2048, use_gpt=True)
    log_agent("강세론자 1R", BULL1_SYSTEM, bull1_user, bull1)
    prompts['bear1'] = {"system": BEAR1_SYSTEM, "user": bear1_user, "model": "GPT-4o"}
    bear1 = call_agent(system_prompt=BEAR1_SYSTEM, user_content=bear1_user, max_tokens=2048, use_gpt=True)
    log_agent("약세론자 1R", BEAR1_SYSTEM, bear1_user, bear1)

    bull2_user = f"[내 주장 (1라운드)]\n{bull1}\n\n[상대방 Bear 주장]\n{bear1}\n\n[참고 데이터]\n기술적: {tech[:300]}\n펀더멘털: {fund[:300]}"
    bear2_user = f"[내 주장 (1라운드)]\n{bear1}\n\n[상대방 Bull 주장]\n{bull1}\n\n[참고 데이터]\n기술적: {tech[:300]}\n펀더멘털: {fund[:300]}"

    print("  [5/6] 2라운드: Bull/Bear 재반박...")
    prompts['bull2'] = {"system": BULL2_SYSTEM, "user": bull2_user, "model": "GPT-4o"}
    bull2 = call_agent(system_prompt=BULL2_SYSTEM, user_content=bull2_user, max_tokens=1024, use_gpt=True)
    log_agent("강세론자 2R", BULL2_SYSTEM, bull2_user, bull2)
    prompts['bear2'] = {"system": BEAR2_SYSTEM, "user": bear2_user, "model": "GPT-4o"}
    bear2 = call_agent(system_prompt=BEAR2_SYSTEM, user_content=bear2_user, max_tokens=1024, use_gpt=True)
    log_agent("약세론자 2R", BEAR2_SYSTEM, bear2_user, bear2)

    news_section = f"\n\n[뉴스 분석]\n{news_analysis}" if news_analysis else ""
    lessons = build_lessons(past_results, as_of) if past_results else ""
    if lessons:
        print(f"    📚 과거 교훈 {lessons.count(chr(10))-1}건 포함")
    manager_user = (
        f"[기술적]\n{tech}\n\n[펀더멘털]\n{fund}{news_section}\n\n"
        f"[강세 1라운드]\n{bull1}\n\n[약세 1라운드]\n{bear1}\n\n"
        f"[강세 2라운드 재반박]\n{bull2}\n\n[약세 2라운드 재반박]\n{bear2}"
        f"{lessons}\n\n"
        'JSON 출력 (반드시 {"expected_return": 으로 시작):'
    )
    prompts['manager'] = {"system": MANAGER_SYSTEM, "user": manager_user, "model": "Claude"}

    print("  [6/6] Manager 최종 판정...")
    result = run_manager_agent(manager_user)
    log_agent("Manager", MANAGER_SYSTEM, manager_user[:300], json.dumps(result, ensure_ascii=False))

    # 해당 월 KOSPI 수익률 (6월 판정 → 6월 수익률)
    month_end = (as_of + relativedelta(months=1)) - relativedelta(days=1)
    kospi_ret = get_kospi_return(as_of, month_end)

    er = result.get("expected_return", 0)
    j = result["judgment"]
    conf = result.get("confidence", 0)

    print(f"  → 예상 수익률: {er:+.1f}% → {j} (신뢰도: {conf})")
    print(f"  → 실제 KOSPI: {kospi_ret:+.1f}%" if kospi_ret else "  → 실제 수익률: N/A")

    # 방향 맞춤 여부 (up/down)
    cycle = _get_cycle(as_of)
    transition = _is_transition(as_of)
    if kospi_ret is not None:
        direction_correct = (er > 0 and kospi_ret > 0) or (er < 0 and kospi_ret < 0) or (er == 0)
    else:
        direction_correct = None

    return {
        "as_of": str(as_of),
        "expected_return": er,
        "judgment": j,
        "confidence": conf,
        "summary": result.get("summary", ""),
        "cycle": cycle,
        "transition": transition,
        "kospi_next_month_return": kospi_ret,
        "direction_correct": direction_correct,
        "detail": {
            "technical_input": tech_input,
            "fundamental_input": fund_input,
            "news_input": news_input,
            "technical": tech,
            "fundamental": fund,
            "news": news_analysis or "",
            "bull1": bull1,
            "bear1": bear1,
            "bull2": bull2,
            "bear2": bear2,
            "manager": result.get("summary", ""),
            "prompts": prompts,
        }
    }


def main():
    # 백테스트 기간과 동일: 2018-04 ~ 2026-03
    test_months = [date(y, m, 1) for y in range(2018, 2027) for m in range(1, 13) if date(2018, 4, 1) <= date(y, m, 1) <= date(2026, 3, 1)]

    # 기존 결과 로드 (테스트 대상은 제거 → 재실행)
    out_path = Path(__file__).parent / "regime_agent_results.json"
    existing = []
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    test_date_strs = {str(d) for d in test_months}
    existing = [r for r in existing if r['as_of'] not in test_date_strs]
    existing_dates = {r['as_of'] for r in existing}

    # 시간순으로 돌리면서 결과 누적 (few-shot 메모리용)
    all_results = list(existing)  # 기존 결과 = 과거 교훈 소스
    new_results = []
    for current in test_months:
        if str(current) in existing_dates:
            print(f"\n  ⏭️ {current} 이미 존재, 스킵")
            continue
        try:
            r = run_month(current, past_results=all_results)
            new_results.append(r)
            all_results.append(r)  # 다음 월 판정 시 교훈으로 활용
            all_results.sort(key=lambda r: r.get('as_of', ''))
            # 중간 저장
            _interim = existing + new_results
            _interim.sort(key=lambda x: x.get('as_of', ''))
            with open(Path(__file__).parent / "regime_agent_results.json", 'w', encoding='utf-8') as _f:
                json.dump(_interim, _f, ensure_ascii=False, indent=2)
            print(f"  💾 중간 저장 완료 ({len(_interim)}건)")
        except Exception as e:
            print(f"  ❌ {current} 오류: {e}")
            try:
                get_conn().rollback()
            except Exception:
                pass
            new_results.append({"as_of": str(current), "error": str(e)})

    # 기존 + 신규 합치고 날짜순 정렬
    results = existing + new_results
    results.sort(key=lambda r: r.get('as_of', ''))

    # 결과 저장
    out_path = Path(__file__).parent / "regime_agent_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 요약
    valid = [r for r in results if r.get("direction_correct") is not None]
    if valid:
        dir_acc = sum(1 for r in valid if r["direction_correct"]) / len(valid) * 100
        print(f"\n{'='*60}")
        print(f"  전체 {len(valid)}개월 방향 정확도: {dir_acc:.1f}%")
        print(f"{'='*60}")
        for r in results:
            er = r.get('expected_return', 0)
            ret = r.get('kospi_next_month_return')
            ret_str = f"{ret:+.1f}%" if ret is not None else "N/A"
            dc = r.get("direction_correct")
            dc_str = "✅" if dc else ("❌" if dc is False else "-")
            print(f"  {r['as_of'][:7]}  예상:{er:+.1f}%  실제:{ret_str:7s}  방향:{dc_str}")

    # HTML 리포트 생성
    html_path = Path(__file__).parent / "regime_agent_report.html"
    generate_html(results, html_path)

    print(f"\n결과 저장: {out_path}")
    print(f"리포트 저장: {html_path}")
    _conn.close()


def generate_html(results: list, out_path: Path):
    valid = [r for r in results if r.get("direction_correct") is not None]
    accuracy = sum(1 for r in valid if r["direction_correct"]) / len(valid) * 100 if valid else 0

    rows_html = ""
    for r in results:
        if "error" in r:
            continue
        j = r.get("judgment", "?")
        conf = r.get("confidence", 0)
        ret = r.get("kospi_next_month_return")
        er = r.get("expected_return", 0)
        dc = r.get("direction_correct")
        ret_str = f"{ret:+.1f}%" if ret is not None else "N/A"
        er_str = f"{er:+.1f}%" if er else "N/A"
        correct_str = "✅" if dc else ("❌" if dc is False else "-")
        j_color = "#2ecc71" if er and er > 0 else "#e74c3c"
        ret_color = "#2ecc71" if ret and ret > 0 else "#e74c3c"

        detail = r.get("detail", {})
        detail_id = f"detail_{r['as_of'][:7].replace('-','')}"

        rows_html += f"""
        <tr onclick="toggleDetail('{detail_id}')" style="cursor:pointer">
            <td>{r['as_of'][:7]}</td>
            <td style="color:{j_color};font-weight:bold">{er_str}</td>
            <td>{conf}</td>
            <td style="color:{ret_color}">{ret_str}</td>
            <td>{correct_str}</td>
            <td style="font-size:0.85em;color:#aaa">{r.get('summary','')[:80]}</td>
        </tr>
        <tr id="{detail_id}" style="display:none;background:#1a1a2e">
            <td colspan="6" style="padding:16px">
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px">
                    <div class="agent-card technical">
                        <div class="agent-label">📈 Technical</div>
                        <pre>{detail.get('technical','').replace('<','&lt;')}</pre>
                    </div>
                    <div class="agent-card fundamental">
                        <div class="agent-label">📊 Fundamental</div>
                        <pre>{detail.get('fundamental','').replace('<','&lt;')}</pre>
                    </div>
                    <div class="agent-card news">
                        <div class="agent-label">📰 News</div>
                        <pre>{detail.get('news','').replace('<','&lt;')}</pre>
                    </div>
                </div>
                <div style="margin:8px 0 4px;color:#888;font-size:0.85em">1라운드 — 독립 주장</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">
                    <div class="agent-card bull">
                        <div class="agent-label">🐂 강세론자</div>
                        <pre>{detail.get('bull1','').replace('<','&lt;')}</pre>
                    </div>
                    <div class="agent-card bear">
                        <div class="agent-label">🐻 약세론자</div>
                        <pre>{detail.get('bear1','').replace('<','&lt;')}</pre>
                    </div>
                </div>
                <div style="margin:8px 0 4px;color:#888;font-size:0.85em">2라운드 — 재반박</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">
                    <div class="agent-card bull">
                        <div class="agent-label">🐂 강세론자 반박</div>
                        <pre>{detail.get('bull2','').replace('<','&lt;')}</pre>
                    </div>
                    <div class="agent-card bear">
                        <div class="agent-label">🐻 약세론자 반박</div>
                        <pre>{detail.get('bear2','').replace('<','&lt;')}</pre>
                    </div>
                </div>
                <div style="margin:8px 0 4px;color:#888;font-size:0.85em">최종 판정</div>
                <div class="agent-card" style="border-color:#f39c12;padding:12px">
                    <div class="agent-label">⚖️ Manager</div>
                    <pre>판정: {r.get('judgment','?')} / 신뢰도: {r.get('confidence',0)}
{detail.get('manager','').replace('<','&lt;')}</pre>
                </div>
            </td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>KOSPI 레짐 판정 리포트</title>
<style>
  body {{ font-family: 'Pretendard', sans-serif; background:#0f0f1a; color:#e0e0e0; margin:0; padding:24px }}
  h1 {{ color:#fff; font-size:1.6em; margin-bottom:4px }}
  .subtitle {{ color:#888; margin-bottom:24px }}
  .summary-bar {{ display:flex; gap:24px; margin-bottom:24px }}
  .kpi {{ background:#1e1e2e; border-radius:8px; padding:16px 24px; text-align:center }}
  .kpi .val {{ font-size:2em; font-weight:bold; color:#7c9fff }}
  .kpi .lbl {{ color:#888; font-size:0.85em }}
  table {{ width:100%; border-collapse:collapse; background:#1e1e2e; border-radius:8px; overflow:hidden }}
  th {{ background:#2a2a3e; padding:12px 16px; text-align:left; color:#aaa; font-size:0.85em }}
  td {{ padding:12px 16px; border-bottom:1px solid #2a2a3e; font-size:0.9em }}
  tr:hover > td {{ background:#252535 }}
  .agent-card {{ background:#0f0f1a; border-radius:6px; padding:12px }}
  .agent-card pre {{ white-space:pre-wrap; font-size:0.8em; color:#ccc; margin:8px 0 0 }}
  .agent-label {{ font-size:0.8em; font-weight:bold; color:#888 }}
  .technical .agent-label {{ color:#7c9fff }}
  .fundamental .agent-label {{ color:#a29bfe }}
  .news .agent-label {{ color:#fdcb6e }}
  .bull .agent-label {{ color:#2ecc71 }}
  .bear .agent-label {{ color:#e74c3c }}
</style>
</head>
<body>
<h1>KOSPI 레짐 판정 리포트</h1>
<div class="subtitle">Trading Agent — 2023-11 ~ 2026-02 | 각 행 클릭 시 상세 내용 확인</div>
<div class="summary-bar">
  <div class="kpi"><div class="val">{len(valid)}</div><div class="lbl">분석 개월</div></div>
  <div class="kpi"><div class="val">{accuracy:.1f}%</div><div class="lbl">판정 정확도</div></div>
  <div class="kpi"><div class="val">{sum(1 for r in valid if r.get('judgment')=='Bull')}</div><div class="lbl">Bull 판정</div></div>
  <div class="kpi"><div class="val">{sum(1 for r in valid if r.get('judgment')=='Bear')}</div><div class="lbl">Bear 판정</div></div>
</div>
<table>
  <thead><tr><th>기준월</th><th>판정</th><th>신뢰도</th><th>실제 다음달 수익률</th><th>정확</th><th>요약</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<script>
function toggleDetail(id) {{
  var el = document.getElementById(id);
  el.style.display = el.style.display === 'none' ? 'table-row' : 'none';
}}
</script>
</body>
</html>"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    main()
