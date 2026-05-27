"""
Trading Agent Multimodel - KOSPI regime prediction.

This version is intentionally isolated from regime_agent.py / regime_agent_v2.py:
- default output: analysis/regime_agent_multimodel_results.json
- default report: analysis/regime_agent_multimodel_report.html
- backtest news default: already-collected Nate/news DB rows only
- optional Gemini grounded-search test mode writes separate raw/citation files

실행:
  python analysis/regime_agent_multimodel.py --months 2018-04,2020-03,2024-08
  python analysis/regime_agent_multimodel.py --start 2018-04 --end 2026-05
  python analysis/regime_agent_multimodel.py --grounding-test 2018-04,2020-03,2024-08
"""
import argparse
import os
import re
import sys
import json
import anthropic
import openai
from google import genai
from google.genai import types as genai_types
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / '.env')

claude_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
openai_client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
CLAUDE_MODEL = os.environ.get("REGIME_CLAUDE_MODEL", "claude-sonnet-4-6")
GEMINI_MODEL = os.environ.get("REGIME_GEMINI_MODEL", "gemini-2.5-flash")
GPT_MODEL = os.environ.get("REGIME_OPENAI_MODEL", "gpt-4o")
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

OUTPUT_PATH = Path(os.environ.get(
    "REGIME_AGENT_OUTPUT_PATH",
    Path(__file__).parent / "regime_agent_multimodel_results.json",
))
REPORT_PATH = Path(os.environ.get(
    "REGIME_AGENT_REPORT_PATH",
    Path(__file__).parent / "regime_agent_multimodel_report.html",
))
GROUNDING_TEST_DIR = Path(__file__).parent / "regime_news_grounding_test"

TRUSTED_NEWS_DOMAINS = {
    # 통신사
    "yna.co.kr", "yonhapnews.co.kr", "newsis.com", "news1.kr",
    # 종합 경제지
    "hankyung.com", "mk.co.kr", "mt.co.kr", "edaily.co.kr",
    "fnnews.com", "sedaily.com", "asiae.co.kr", "heraldcorp.com",
    "biz.chosun.com", "moneys.co.kr", "wowtv.co.kr", "ebn.co.kr",
    "thebell.co.kr", "businesspost.co.kr", "einfomax.co.kr",
    # 종합지
    "chosun.com", "donga.com", "joongang.co.kr", "hani.co.kr", "khan.co.kr",
    # IT/산업
    "etnews.com", "zdnet.co.kr", "ddaily.co.kr",
    # 방송
    "kbs.co.kr", "sbs.co.kr", "imbc.com", "ytn.co.kr", "mbn.co.kr", "jtbc.co.kr",
    # 기타 일간/온라인
    "dailian.co.kr", "nocutnews.co.kr", "pressian.com", "ohmynews.com",
    # 공공/연구
    "bok.or.kr", "fss.or.kr", "moef.go.kr", "kdi.re.kr", "hri.co.kr",
    "krx.co.kr", "kosis.kr", "kostat.go.kr",
}


_PCT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
_BPS_PATTERN = re.compile(r"\d+\s*(?:bp|bps|베이시스\s*포인트?)", re.IGNORECASE)
_MONEY_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:조\s*원|억\s*원|만\s*원|조\s*달러|억\s*달러|만\s*달러|달러|원)"
)
_INDEX_PATTERN = re.compile(
    r"(?:KOSPI|코스피|S&P\s*500|나스닥|다우|지수|포인트|pt)\D{0,15}?(\d[\d,\.]{2,})",
    re.IGNORECASE,
)
_RAW_NUM_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")


def _score_specificity(text: str) -> dict:
    """bull/bear 토론 재료 적합성 — 수치/단위 밀도 측정."""
    if not text:
        return {"percent": 0, "bps": 0, "money": 0, "index_level": 0, "raw_numbers": 0, "verdict": "empty"}
    counts = {
        "percent": len(_PCT_PATTERN.findall(text)),
        "bps": len(_BPS_PATTERN.findall(text)),
        "money": len(_MONEY_PATTERN.findall(text)),
        "index_level": len(_INDEX_PATTERN.findall(text)),
        "raw_numbers": len(_RAW_NUM_PATTERN.findall(text)),
    }
    typed = counts["percent"] + counts["bps"] + counts["money"] + counts["index_level"]
    if typed >= 8:
        verdict = "rich"
    elif typed >= 4:
        verdict = "ok"
    elif typed >= 1:
        verdict = "thin"
    else:
        verdict = "no_numbers"
    counts["typed_total"] = typed
    counts["verdict"] = verdict
    return counts


def _classify_grounding_chunks(chunks: list) -> dict:
    """화이트리스트 도메인 기준으로 grounding chunks를 분류."""
    trusted, flagged = [], []
    for idx, c in enumerate(chunks or []):
        web = (c or {}).get("web") or {}
        title = (web.get("title") or "").strip().lower()
        domain = title.lstrip("www.")
        is_trusted = any(
            domain == td or domain.endswith("." + td)
            for td in TRUSTED_NEWS_DOMAINS
        )
        entry = {"index": idx, "domain": title, "uri": web.get("uri")}
        (trusted if is_trusted else flagged).append(entry)
    total = len(trusted) + len(flagged)
    return {
        "trusted": trusted,
        "flagged": flagged,
        "trust_ratio": round(len(trusted) / total, 3) if total else 0.0,
        "total_sources": total,
    }

ROLE_MODELS = {
    "technical": os.environ.get("REGIME_MODEL_TECHNICAL", "claude"),
    "fundamental": os.environ.get("REGIME_MODEL_FUNDAMENTAL", "gemini"),
    "news": os.environ.get("REGIME_MODEL_NEWS", "gemini"),
    "bull": os.environ.get("REGIME_MODEL_BULL", "claude"),
    "bear": os.environ.get("REGIME_MODEL_BEAR", "gemini"),
    "bull_rebuttal": os.environ.get("REGIME_MODEL_BULL_REBUTTAL", "claude"),
    "bear_rebuttal": os.environ.get("REGIME_MODEL_BEAR_REBUTTAL", "gemini"),
    "manager": os.environ.get("REGIME_MODEL_MANAGER", "claude"),
}

# 모델 비교 실험용 preset. --config로 선택하면 모든 역할을 일괄 덮어씀.
# "mixed"는 위 ROLE_MODELS 디폴트 그대로(빈 dict = override 없음).
PRESETS = {
    "mixed":  {},
    "claude": {k: "claude" for k in ROLE_MODELS},
    "gemini": {k: "gemini" for k in ROLE_MODELS},
    "gpt":    {k: "gpt"    for k in ROLE_MODELS},
}

conn = None
_technical_df = None
_macro_df = None


def init_data():
    """Connect to DB and preload market data. Deferred so --help and grounding tests stay light."""
    global conn, _technical_df, _macro_df
    if conn is not None and _technical_df is not None and _macro_df is not None:
        return
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    print("데이터 사전 로딩 중...")
    _technical_df = pd.read_sql(
        "SELECT trade_date, indicator, value, COALESCE(symbol, 'KOSPI') AS symbol FROM alpha_lab.technical_indicators ORDER BY trade_date",
        conn, parse_dates=['trade_date']
    )
    _technical_df['trade_date'] = pd.to_datetime(_technical_df['trade_date']).dt.date
    print(f"  technical: {len(_technical_df)}건")

    _macro_df = pd.read_sql(
        "SELECT indicator, period, freq, value FROM alpha_lab.macro_indicators",
        conn
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
    cur = conn.cursor()
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
    cur = conn.cursor()
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
    cur = conn.cursor()
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
                # as_of가 월초일 때 i=0과 겹치지 않도록, i개월 전 월초 -1일 = (i-1)개월 전 월말
                m = as_of - relativedelta(months=i)
                cutoff = m - relativedelta(days=1)
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
    lines.append("[현재 지표 상태]")
    if current_rsi < 35:
        lines.append(f"▶ 현재 RSI {current_rsi:.1f} (과매도 구간)")
    elif current_rsi > 70:
        lines.append(f"▶ 현재 RSI {current_rsi:.1f} (과매수 구간)")
    else:
        lines.append(f"▶ 현재 RSI {current_rsi:.1f} (중립 구간)")

    if current_close > current_ma200:
        lines.append(f"▶ KOSPI가 MA200 위 (상승추세)")
    else:
        lines.append(f"▶ KOSPI가 MA200 아래 (하락추세)")

    # ── V2 추가: 변화율 Feature ──────────────────────────────
    lines.append("")
    lines.append("[변화율 Feature - 전환 감지용]")

    # MA50 기울기 (전월 대비 변화)
    if len(ma50_series) >= 2:
        ma50_slope = ma50_series[-1][1] - ma50_series[-2][1]
        lines.append(f"▶ MA50 기울기(전월대비): {ma50_slope:+.0f} ({'상승' if ma50_slope > 0 else '하락'})")

    # MA200 기울기
    if len(ma200_series) >= 2:
        ma200_slope = ma200_series[-1][1] - ma200_series[-2][1]
        lines.append(f"▶ MA200 기울기(전월대비): {ma200_slope:+.0f} ({'상승' if ma200_slope > 0 else '하락'})")

    # RSI 변화량
    if len(rsi_series) >= 2:
        rsi_delta = rsi_series[-1][1] - rsi_series[-2][1]
        lines.append(f"▶ RSI 변화량(전월대비): {rsi_delta:+.1f}")

    # RSI 3개월 변화량 (추세)
    if len(rsi_series) >= 3:
        rsi_delta_3m = rsi_series[-1][1] - rsi_series[-3][1]
        lines.append(f"▶ RSI 3개월 변화량: {rsi_delta_3m:+.1f}")

    # MACD 변화량 (모멘텀 변화)
    if len(macd_series) >= 2:
        macd_delta = macd_series[-1][1] - macd_series[-2][1]
        lines.append(f"▶ MACD 변화량(전월대비): {macd_delta:+.1f} ({'모멘텀 개선' if macd_delta > 0 else '모멘텀 악화'})")

    # KOSPI 월간 수익률 (최근 3개월)
    if len(kospi_series) >= 2:
        for i in range(-1, max(-4, -len(kospi_series)), -1):
            ret = (kospi_series[i][1] - kospi_series[i-1][1]) / kospi_series[i-1][1] * 100
            lines.append(f"▶ KOSPI 월간수익률 {kospi_series[i][0]}: {ret:+.1f}%")

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

    # ── V2 추가: 매크로 변화율 Feature ────────────────────────
    lines.append("")
    lines.append("[매크로 변화율 Feature - 전환 감지용]")

    # VIX 변화율
    def daily_monthly_avg(indicator, n_months=3):
        series = get_macro_daily_series(as_of, indicator, 90)
        monthly = {}
        for period, val in series:
            monthly.setdefault(period[:7], []).append(val)
        items = sorted(monthly.items())[-n_months:]
        return [(ym, sum(v)/len(v)) for ym, v in items]

    vix_monthly = daily_monthly_avg('vix')
    if len(vix_monthly) >= 2:
        vix_chg = vix_monthly[-1][1] - vix_monthly[-2][1]
        vix_pct = vix_chg / vix_monthly[-2][1] * 100 if vix_monthly[-2][1] else 0
        lines.append(f"▶ VIX 변화율(전월대비): {vix_chg:+.1f} ({vix_pct:+.1f}%) {'급등' if vix_pct > 20 else '급락' if vix_pct < -20 else ''}")

    # 환율 변화율
    usd_monthly = daily_monthly_avg('usd_krw')
    if len(usd_monthly) >= 2:
        usd_chg = usd_monthly[-1][1] - usd_monthly[-2][1]
        lines.append(f"▶ 원/달러 변화(전월대비): {usd_chg:+.1f}원 ({'원화약세' if usd_chg > 0 else '원화강세'})")

    # 금리 변화율 (10년물)
    bond_monthly = daily_monthly_avg('bond_10y')
    if len(bond_monthly) >= 2:
        bond_chg = bond_monthly[-1][1] - bond_monthly[-2][1]
        lines.append(f"▶ 국고채10y 변화(전월대비): {bond_chg:+.2f}%p")

    # 수급 가속도 (외국인 전월 대비 변화)
    for key, label in [('investor_foreign_kospi', '외국인'), ('investor_institution_kospi', '기관')]:
        series = get_macro_daily_series(as_of, key, 120)
        monthly = {}
        for period, val in series:
            monthly.setdefault(period[:7], []).append(val)
        items = sorted(monthly.items())[-3:]
        if len(items) >= 2:
            net_prev = (items[-2][1][-1] - items[-2][1][0]) / 100
            net_curr = (items[-1][1][-1] - items[-1][1][0]) / 100
            accel = net_curr - net_prev
            lines.append(f"▶ {label} 수급 가속도: {accel:+.0f}억 ({'매수 가속' if accel > 0 else '매도 가속'})")

    return "\n".join(lines)


# ── Agent 호출 ────────────────────────────────────────────────

def model_label(model: str) -> str:
    labels = {
        "claude": f"Claude ({CLAUDE_MODEL})",
        "gemini": f"Gemini ({GEMINI_MODEL})",
        "gpt": f"OpenAI ({GPT_MODEL})",
    }
    return labels.get(model, model)


def call_agent(system_prompt: str, user_content: str, max_tokens: int = 400,
               model: str = "gemini", temperature: float = 1.0) -> str:
    """멀티모델 Agent 호출. model: 'gemini', 'gpt', 'claude'"""
    import time
    for attempt in range(5):
        try:
            if model == "gpt":
                resp = openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ]
                )
                return resp.choices[0].message.content
            elif model == "claude":
                msg = claude_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}]
                )
                return msg.content[0].text
            elif model == "gemini":
                prompt = f"[System]\n{system_prompt}\n\n[User]\n{user_content}"
                resp = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                    )
                )
                return resp.text
            else:
                raise ValueError(f"Unknown model: {model}")
        except Exception as e:
            if attempt < 4 and ('overloaded' in str(e).lower() or '529' in str(e) or '500' in str(e) or '503' in str(e)):
                wait = 30 * (attempt + 1)
                print(f"    ⏳ API 오류 ({model}), {wait}초 후 재시도 ({attempt+1}/5)...")
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
    "토론 내용을 종합하여 이번 달 KOSPI 레짐을 5단계로 분류하고 각 레짐의 확률을 추정하세요.\n\n"
    "[레짐 정의 - 월간 KOSPI 수익률 구간]\n"
    "- 크래시 : 월수익률 <  -7%\n"
    "- 약세   : -7% ~ -3%\n"
    "- 중립   : -3% ~ +3%\n"
    "- 강세   : +3% ~ +7%\n"
    "- 랠리   : > +7%\n\n"
    "[판단 원칙]\n"
    "1. 5개 레짐 확률의 합은 정확히 1.00. 가장 확률 높은 레짐을 regime 필드에 명시.\n"
    "2. 확신이 없을 때 중립에 70% 이상 몰아주지 마라. 토론에서 약세 신호가 강하면 크래시/약세에 정직하게 확률을 부여.\n"
    "3. KOSPI는 양극단(크래시·랠리)도 정기적으로 발생하는 시장이다. 안전선 잡지 말고 토론에서 본 신호 강도대로 확률을 분산하라.\n"
    "4. 중립이 가장 흔한 결과이긴 하지만 압도적으로 흔하지는 않다. 신호가 분명하면 적극적으로 한쪽으로 기울여라.\n\n"
    "[전략 맥락]\n"
    "당신의 판정은 실제 포트폴리오 운용에 직접 사용됩니다. 비중은 다음 가중치로 계산됩니다:\n"
    "- 크래시:10%, 약세:15%, 중립:20%, 강세:27%, 랠리:30%\n"
    "주의: 잘못된 약세 판정 → 수익 기회 상실. 잘못된 강세 판정 → 큰 손실. 양쪽 비용 균형 고려.\n\n"
    "[출력 규칙]\n"
    "반드시 JSON 한 줄만 출력. 앞뒤 설명, 마크다운, 코드블록 절대 금지.\n"
    "형식: {\"regime\": \"크래시|약세|중립|강세|랠리\", \"probabilities\": {\"크래시\": 0.xx, \"약세\": 0.xx, \"중립\": 0.xx, \"강세\": 0.xx, \"랠리\": 0.xx}, \"confidence\": 0~100, \"summary\": \"문장\"}\n"
    "regime: 가장 확률 높은 레짐 한글명.\n"
    "probabilities: 5개 레짐 확률, 합=1.00, 소수점 2자리.\n"
    "confidence: 가장 높은 확률 레짐에 대한 확신도 0~100.\n"
    "summary: 핵심 수치를 포함해 2문장 이내로 간결하게."
)

def run_technical_agent(summary: str) -> str:
    return call_agent(
        system_prompt=TECH_SYSTEM,
        user_content=summary,
        max_tokens=8192,
        model=ROLE_MODELS["technical"],
    )


def run_fundamental_agent(summary: str) -> str:
    return call_agent(
        system_prompt=FUND_SYSTEM,
        user_content=summary,
        max_tokens=8192,
        model=ROLE_MODELS["fundamental"],
    )


NEWS_SYSTEM = (
    "당신은 매크로 뉴스 분석 전문가입니다. "
    "주어진 뉴스들에서 KOSPI에 영향을 미치는 핵심 이벤트/흐름을 포인트 3가지로 정리하세요. "
    "개별종목 뉴스는 무시하고 매크로/시장 전체 흐름만 분석하세요. Bull/Bear 판정은 하지 마세요. "
    f"{NO_EXTERNAL} {CONCISE} "
    "형식: 포인트1: ...\n포인트2: ...\n포인트3: ..."
)

def run_news_agent(as_of: date) -> tuple[str, str]:
    """뉴스 이벤트 추출. 백테스트 기본값은 DB에 저장된 시점 격리 뉴스만 사용."""
    news = get_news_summary(as_of)
    if not news:
        return None, None
    news_input = f"[{as_of.strftime('%Y-%m')} 직전월 매크로 뉴스]\n{news}"
    result = call_agent(
        system_prompt=NEWS_SYSTEM,
        user_content=news_input,
        max_tokens=8192,
        model=ROLE_MODELS["news"],
    )
    return result, news_input


def run_bull1(tech: str, fund: str) -> str:
    return call_agent(system_prompt=BULL1_SYSTEM,
                      user_content=f"[기술적 분석]\n{tech}\n\n[펀더멘털 분석]\n{fund}",
                      max_tokens=8192, model=ROLE_MODELS["bull"])


def run_bear1(tech: str, fund: str) -> str:
    return call_agent(system_prompt=BEAR1_SYSTEM,
                      user_content=f"[기술적 분석]\n{tech}\n\n[펀더멘털 분석]\n{fund}",
                      max_tokens=8192, model=ROLE_MODELS["bear"])


def run_bull2(tech: str, fund: str, bull1: str, bear1: str) -> str:
    return call_agent(
        system_prompt=BULL2_SYSTEM,
        user_content=(
            f"[내 주장 (1라운드)]\n{bull1}\n\n"
            f"[상대방 Bear 주장]\n{bear1}\n\n"
            f"[참고 데이터]\n기술적: {tech[:300]}\n펀더멘털: {fund[:300]}"
        ),
        max_tokens=4096, model=ROLE_MODELS["bull_rebuttal"]
    )


def run_bear2(tech: str, fund: str, bull1: str, bear1: str) -> str:
    return call_agent(
        system_prompt=BEAR2_SYSTEM,
        user_content=(
            f"[내 주장 (1라운드)]\n{bear1}\n\n"
            f"[상대방 Bull 주장]\n{bull1}\n\n"
            f"[참고 데이터]\n기술적: {tech[:300]}\n펀더멘털: {fund[:300]}"
        ),
        max_tokens=4096, model=ROLE_MODELS["bear_rebuttal"]
    )


# 5-class 레짐 → midpoint (확률 가중 ER 계산용)
REGIME_MIDPOINTS = {"크래시": -10.0, "약세": -5.0, "중립": 0.0, "강세": +5.0, "랠리": +10.0}
REGIME_KEYS = ["크래시", "약세", "중립", "강세", "랠리"]


def classify_actual_regime(kospi_ret: float) -> str:
    """실제 KOSPI 수익률을 5-class bucket으로 분류."""
    if kospi_ret < -7:
        return "크래시"
    elif kospi_ret < -3:
        return "약세"
    elif kospi_ret < 3:
        return "중립"
    elif kospi_ret < 7:
        return "강세"
    else:
        return "랠리"


def compute_regime_prior(as_of: date) -> dict | None:
    """as_of 이전까지의 KOSPI 월수익률 분포 — rolling backward, lookahead-free."""
    if _technical_df is None:
        return None
    sub = _technical_df[
        (_technical_df['indicator'] == 'close') &
        (_technical_df['symbol'] == 'KOSPI') &
        (_technical_df['trade_date'] < as_of)
    ].sort_values('trade_date')
    if sub.empty:
        return None

    # 월별 마지막 거래일 종가
    sub = sub.copy()
    sub['ym'] = sub['trade_date'].apply(lambda d: (d.year, d.month))
    monthly = sub.groupby('ym', as_index=False).last().sort_values('trade_date')
    closes = monthly['value'].tolist()
    if len(closes) < 2:
        return None

    # 월수익률 → 5-class 버킷
    buckets = {k: 0 for k in REGIME_KEYS}
    for i in range(1, len(closes)):
        ret = (closes[i] - closes[i-1]) / closes[i-1] * 100
        if ret < -7:
            buckets["크래시"] += 1
        elif ret < -3:
            buckets["약세"] += 1
        elif ret < 3:
            buckets["중립"] += 1
        elif ret < 7:
            buckets["강세"] += 1
        else:
            buckets["랠리"] += 1

    total = sum(buckets.values())
    if total == 0:
        return None
    return {"counts": buckets, "total": total}


def parse_prediction(result: str, fallback_key: str = "summary") -> dict:
    import re
    try:
        m = re.search(r'\{.*\}', result, re.DOTALL)
        if not m:
            raise ValueError("JSON not found")
        parsed = json.loads(m.group())

        # 새 schema: regime + probabilities
        if "regime" in parsed and "probabilities" in parsed:
            probs = parsed["probabilities"]
            # 키 정규화 + 합 정규화
            probs = {k: float(probs.get(k, 0)) for k in REGIME_KEYS}
            total = sum(probs.values()) or 1.0
            probs = {k: v / total for k, v in probs.items()}
            parsed["probabilities"] = {k: round(v, 4) for k, v in probs.items()}
            # 일관성 보장: regime은 argmax(probabilities)로 강제 (모델 self-inconsistency 차단)
            stated = parsed.get("regime", "")
            argmax = max(probs, key=probs.get)
            parsed["regime_stated"] = stated
            parsed["regime"] = argmax
            # 확률 가중 ER
            er = sum(probs[k] * REGIME_MIDPOINTS[k] for k in REGIME_KEYS)
            parsed["expected_return"] = round(er, 2)
            # judgment 매핑 (기존 ±3% 임계 유지 + 5-class 직접 매핑)
            regime = parsed.get("regime", "")
            if regime in ("크래시", "약세"):
                parsed["judgment"] = "약세"
            elif regime in ("강세", "랠리"):
                parsed["judgment"] = "강세"
            else:
                parsed["judgment"] = "변동성"
            return parsed

        # 구 schema fallback (expected_return만 있는 경우)
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
    return {"judgment": "N/A", "expected_return": 0, "confidence": 0, fallback_key: result[:200]}


def run_manager_agent(manager_user: str) -> dict:
    for attempt in range(3):
        result = call_agent(
            system_prompt=MANAGER_SYSTEM,
            user_content=manager_user,
            max_tokens=8192,
            model=ROLE_MODELS["manager"],
            temperature=1.0,
        )
        parsed = parse_prediction(result)
        if parsed.get("judgment") != "N/A":
            return parsed
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

def run_month(as_of: date, past_results: list = None, debate_mode: str = "full") -> dict:
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

    tech_model = ROLE_MODELS["technical"]
    fund_model = ROLE_MODELS["fundamental"]
    news_model = ROLE_MODELS["news"]
    bull_model = ROLE_MODELS["bull"]
    bear_model = ROLE_MODELS["bear"]
    bull_rebuttal_model = ROLE_MODELS["bull_rebuttal"]
    bear_rebuttal_model = ROLE_MODELS["bear_rebuttal"]
    manager_model = ROLE_MODELS["manager"]

    n_steps = {"full": 6, "single": 5, "none": 4}[debate_mode]
    print(f"  [1/{n_steps}] Technical Agent ({model_label(tech_model)})...")
    prompts['technical'] = {"system": TECH_SYSTEM, "user": tech_input, "model": model_label(tech_model)}
    tech = run_technical_agent(tech_input)
    log_agent(f"Technical [{model_label(tech_model)}]", TECH_SYSTEM, tech_input, tech)

    print(f"  [2/{n_steps}] Fundamental Agent ({model_label(fund_model)})...")
    prompts['fundamental'] = {"system": FUND_SYSTEM, "user": fund_input, "model": model_label(fund_model)}
    fund = run_fundamental_agent(fund_input)
    log_agent(f"Fundamental [{model_label(fund_model)}]", FUND_SYSTEM, fund_input, fund)

    print(f"  [3/{n_steps}] News Event Agent ({model_label(news_model)})...")
    news_analysis, news_input = run_news_agent(as_of)
    if news_analysis:
        prompts['news'] = {"system": NEWS_SYSTEM, "user": news_input, "model": model_label(news_model)}
        log_agent(f"News Event [{model_label(news_model)}]", NEWS_SYSTEM, news_input, news_analysis)
    else:
        news_analysis = ""
        news_input = ""
        print(f"    뉴스 없음 (해당 월 미수집)")

    bull1 = bear1 = bull2 = bear2 = ""
    if debate_mode in ("full", "single"):
        debate_input = f"[기술적 분석]\n{tech}\n\n[펀더멘털 분석]\n{fund}"
        if news_analysis:
            debate_input += f"\n\n[뉴스 이벤트]\n{news_analysis}"

        bull1_user = debate_input
        bear1_user = debate_input

        print(f"  [4/{n_steps}] 1라운드: Bull({model_label(bull_model)}) / Bear({model_label(bear_model)}) 독립 주장...")
        prompts['bull1'] = {"system": BULL1_SYSTEM, "user": bull1_user, "model": model_label(bull_model)}
        bull1 = call_agent(system_prompt=BULL1_SYSTEM, user_content=bull1_user, max_tokens=8192, model=bull_model)
        log_agent(f"강세론자 1R [{model_label(bull_model)}]", BULL1_SYSTEM, bull1_user, bull1)
        prompts['bear1'] = {"system": BEAR1_SYSTEM, "user": bear1_user, "model": model_label(bear_model)}
        bear1 = call_agent(system_prompt=BEAR1_SYSTEM, user_content=bear1_user, max_tokens=8192, model=bear_model)
        log_agent(f"약세론자 1R [{model_label(bear_model)}]", BEAR1_SYSTEM, bear1_user, bear1)

    if debate_mode == "full":
        bull2_user = f"[내 주장 (1라운드)]\n{bull1}\n\n[상대방 Bear 주장]\n{bear1}\n\n[참고 데이터]\n기술적: {tech[:300]}\n펀더멘털: {fund[:300]}"
        bear2_user = f"[내 주장 (1라운드)]\n{bear1}\n\n[상대방 Bull 주장]\n{bull1}\n\n[참고 데이터]\n기술적: {tech[:300]}\n펀더멘털: {fund[:300]}"

        print(f"  [5/{n_steps}] 2라운드: Bull({model_label(bull_rebuttal_model)}) / Bear({model_label(bear_rebuttal_model)}) 재반박...")
        prompts['bull2'] = {"system": BULL2_SYSTEM, "user": bull2_user, "model": model_label(bull_rebuttal_model)}
        bull2 = call_agent(system_prompt=BULL2_SYSTEM, user_content=bull2_user, max_tokens=4096, model=bull_rebuttal_model)
        log_agent(f"강세론자 2R [{model_label(bull_rebuttal_model)}]", BULL2_SYSTEM, bull2_user, bull2)
        prompts['bear2'] = {"system": BEAR2_SYSTEM, "user": bear2_user, "model": model_label(bear_rebuttal_model)}
        bear2 = call_agent(system_prompt=BEAR2_SYSTEM, user_content=bear2_user, max_tokens=4096, model=bear_rebuttal_model)
        log_agent(f"약세론자 2R [{model_label(bear_rebuttal_model)}]", BEAR2_SYSTEM, bear2_user, bear2)

    news_section = f"\n\n[뉴스 이벤트]\n{news_analysis}" if news_analysis else ""
    lessons = build_lessons(past_results, as_of) if past_results else ""
    if lessons:
        print(f"    과거 교훈 {lessons.count(chr(10))-1}건 포함")

    # 분석 시점 이전까지의 KOSPI 월수익률 분포 (lookahead-free prior)
    prior = compute_regime_prior(as_of)
    prior_section = ""
    if prior:
        lines = [f"\n\n[참고 - 분석 시점 이전 {prior['total']}개월 KOSPI 월수익률 분포]"]
        for k in REGIME_KEYS:
            cnt = prior['counts'][k]
            pct = cnt / prior['total'] * 100
            lines.append(f"  {k}: {cnt}건 ({pct:.0f}%)")
        lines.append("위 분포는 분석 시점까지의 평시 prior. 현재 신호가 분명하면 적극적으로 조정하라.")
        prior_section = "\n".join(lines)

    debate_block = ""
    if debate_mode == "full":
        debate_block = (
            f"\n\n[강세 1라운드]\n{bull1}\n\n[약세 1라운드]\n{bear1}\n\n"
            f"[강세 2라운드 재반박]\n{bull2}\n\n[약세 2라운드 재반박]\n{bear2}"
        )
    elif debate_mode == "single":
        debate_block = f"\n\n[강세 의견]\n{bull1}\n\n[약세 의견]\n{bear1}"
    manager_user = (
        f"[기술적]\n{tech}\n\n[펀더멘털]\n{fund}{news_section}"
        f"{debate_block}"
        f"{lessons}"
        f"{prior_section}\n\n"
        'JSON 출력 (반드시 {"regime": 으로 시작):'
    )
    prompts['manager'] = {"system": MANAGER_SYSTEM, "user": manager_user, "model": model_label(manager_model)}

    print(f"  [{n_steps}/{n_steps}] Manager 최종 판정 ({model_label(manager_model)})...")
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
    predicted_regime = result.get("regime")
    actual_regime = classify_actual_regime(kospi_ret) if kospi_ret is not None else None
    if kospi_ret is not None:
        direction_correct = (er > 0 and kospi_ret > 0) or (er < 0 and kospi_ret < 0) or (er == 0)
        if predicted_regime in REGIME_KEYS and actual_regime in REGIME_KEYS:
            regime_correct = predicted_regime == actual_regime
            regime_distance = abs(REGIME_KEYS.index(predicted_regime) - REGIME_KEYS.index(actual_regime))
        else:
            regime_correct = None
            regime_distance = None
    else:
        direction_correct = None
        regime_correct = None
        regime_distance = None

    if actual_regime:
        print(f"  → 실제 레짐: {actual_regime} (예측: {predicted_regime}, 거리: {regime_distance})")

    return {
        "as_of": str(as_of),
        "expected_return": er,
        "judgment": j,
        "confidence": conf,
        "regime": predicted_regime,
        "regime_stated": result.get("regime_stated"),
        "probabilities": result.get("probabilities"),
        "summary": result.get("summary", ""),
        "cycle": cycle,
        "transition": transition,
        "kospi_next_month_return": kospi_ret,
        "actual_regime": actual_regime,
        "regime_correct": regime_correct,
        "regime_distance": regime_distance,
        "direction_correct": direction_correct,
        "prior": prior,
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
            "role_models": {k: model_label(v) for k, v in ROLE_MODELS.items()},
        }
    }


def _json_default(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def run_grounded_news_search(as_of: date) -> dict:
    """Gemini grounded search for lookahead-bias audit. Not used in backtests."""
    as_of_text = as_of.strftime("%Y-%m-%d")
    prev_month = (as_of - relativedelta(months=1)).strftime("%Y-%m")
    target_month = as_of.strftime("%Y-%m")
    prompt = f"""
역할: 너는 KOSPI 백테스트의 lookahead-bias 감사관이다.
본 작업은 미래 예측이 아니라, "{as_of_text} 시점의 정보만으로 의사결정 재료를 정리할 수 있는지"를 검증하는 절차다.

【시간 정의 — 절대 혼동 금지】
- 정보 컷오프(knowledge cutoff): {as_of_text}
    · 이 날짜 이전(< {as_of_text})에 공개된 뉴스만 사용 가능.
    · 이 날짜 이후(> {as_of_text})에 게재·업데이트된 모든 기사·블로그·회고·실적 발표·통계 자료는 사용 금지.
- 재료(material) 기간: {prev_month} 한 달간 발간된 한국 시장·거시 경제 뉴스.
- 예측 대상(prediction target) 기간: {target_month} 한 달간의 KOSPI 흐름.
    · 너는 예측을 수행하지 않는다. 오직 {target_month} 흐름에 영향을 줄 만한 {prev_month} 뉴스 재료를 정리할 뿐이다.
    · {target_month} 또는 그 이후의 실제 시세·실적·결과·회고 분석은 절대 인용 금지.

【출력 — 반드시 한국어, 아래 양식 그대로】

(1) 첫 문장은 반드시 다음 양식으로 시작:
"정보 컷오프 {as_of_text} 기준, {prev_month} 발간 뉴스 중 {target_month} KOSPI 흐름에 영향을 줄 수 있는 항목:"

(2) 핵심 이벤트 3~5개 — bull/bear 토론 재료로 쓰일 수 있도록 구체적으로 작성
    · 각 항목 3~5줄, 발간일(YYYY-MM-DD) 명시.
    · **각 이벤트마다 검증 가능한 수치 최소 2개 이상 본문에 포함**.
        예: "기준금리 1.50% → 1.75% (+25bp)", "KOSPI 2,400pt 돌파", "외국인 순매수 1.2조원",
            "달러/원 1,080원", "수출 전년동기비 +6.1%", "CPI 1.3%", "유가 WTI 65달러".
    · "양호" "급등" "우려 확산" 같은 정성 표현만 있고 수치가 빠지면 안 됨.
    · 이벤트가 {target_month} KOSPI에 어떤 채널(금리·환율·수급·심리·정책 등)로 영향을 줄 수 있는지 한 줄 부연.

(3) 각 이벤트의 출처
    · 매체명 · 기사 제목 · 발간일(YYYY-MM-DD) · URL.
    · 발간일이 모호하면 "발간일 불확실"로 명시.
    · 본문 (2)에 적은 핵심 수치의 1차 출처를 가능하면 같은 항목 안에 함께 표기.

(4) LOOKAHEAD_CONTAMINATION 의심 항목
    · grounded search가 끌어온 출처 중 발간/업데이트 일자가 {as_of_text}보다 이후인 모든 항목을 빠짐없이 나열.
    · 각 항목에 (게재일, 컷오프 대비 며칠/몇 달 미래인지) 함께 표기.
    · 의심 항목이 없으면 "해당 없음"이라고만 적는다.
"""
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            temperature=0.1,
            max_output_tokens=8192,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = json.loads(json.dumps(response, default=_json_default, ensure_ascii=False))
    chunks = (
        ((raw.get("candidates") or [{}])[0].get("grounding_metadata") or {}).get("grounding_chunks")
        or []
    )
    text = getattr(response, "text", "")
    return {
        "as_of": as_of_text,
        "prompt": prompt,
        "text": text,
        "audit": _classify_grounding_chunks(chunks),
        "specificity": _score_specificity(text),
        "raw": raw,
    }


def run_grounding_tests(months: list[date]):
    GROUNDING_TEST_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for as_of in months:
        print(f"\n[Grounding Test] {as_of}")
        payload = run_grounded_news_search(as_of)
        out_path = GROUNDING_TEST_DIR / f"{as_of.strftime('%Y-%m')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        audit = payload.get("audit", {})
        spec = payload.get("specificity", {})
        index.append({
            "as_of": payload["as_of"],
            "path": str(out_path),
            "trust_ratio": audit.get("trust_ratio"),
            "trusted_domains": [t["domain"] for t in audit.get("trusted", [])],
            "flagged_domains": [t["domain"] for t in audit.get("flagged", [])],
            "specificity_verdict": spec.get("verdict"),
            "specificity": {k: spec.get(k) for k in ("percent", "bps", "money", "index_level", "typed_total")},
            "manual_check": "Verify every citation URL publication date <= as_of.",
        })
        print(
            f"  저장: {out_path}  "
            f"(trusted {len(audit.get('trusted', []))}/{audit.get('total_sources', 0)}, "
            f"specificity={spec.get('verdict')} "
            f"[%×{spec.get('percent')}, bp×{spec.get('bps')}, ₩×{spec.get('money')}, idx×{spec.get('index_level')}])"
        )
    index_path = GROUNDING_TEST_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\nGrounding audit index: {index_path}")


def parse_month(value: str) -> date:
    return datetime.strptime(value, "%Y-%m").date().replace(day=1)


def month_range(start: date, end: date) -> list[date]:
    months = []
    current = start
    while current <= end:
        months.append(current)
        current = current + relativedelta(months=1)
    return months


def parse_args():
    parser = argparse.ArgumentParser(description="Run isolated multimodel KOSPI regime agent.")
    parser.add_argument("--start", default="2018-04", help="Backtest start month, YYYY-MM.")
    parser.add_argument("--end", default="2026-05", help="Backtest end month, YYYY-MM.")
    parser.add_argument("--months", help="Comma-separated months to run, e.g. 2018-04,2020-03.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Result JSON path.")
    parser.add_argument("--report", default=str(REPORT_PATH), help="HTML report path.")
    parser.add_argument("--force", action="store_true", help="Re-run months already present in output.")
    parser.add_argument("--grounding-test", help="Comma-separated months for Gemini grounded-search lookahead test only.")
    parser.add_argument("--config", choices=list(PRESETS.keys()), default="mixed",
                        help="모델 preset. mixed=현행 디폴트, claude/gemini/gpt=전체 단일 모델.")
    parser.add_argument("--debate-mode", choices=["full", "single", "none"], default="full",
                        help="full=2라운드(기본), single=1라운드만(bull1/bear1, 재반박 생략), none=토론 전부 생략.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.grounding_test:
        months = [parse_month(m.strip()) for m in args.grounding_test.split(",") if m.strip()]
        run_grounding_tests(months)
        return

    # --config preset 적용: ROLE_MODELS 전체를 일괄 덮어씀.
    overrides = PRESETS.get(args.config, {})
    if overrides:
        ROLE_MODELS.update(overrides)
    # --output/--report가 디폴트면 config suffix 자동 추가.
    debate_suffix = {"full": "", "single": "_single", "none": "_nodebate"}[args.debate_mode]
    suffix = args.config + debate_suffix
    if args.output == str(OUTPUT_PATH):
        args.output = str(OUTPUT_PATH).replace(".json", f"_{suffix}.json")
    if args.report == str(REPORT_PATH):
        args.report = str(REPORT_PATH).replace(".html", f"_{suffix}.html")
    print(f"[Config: {args.config}] ROLE_MODELS = {ROLE_MODELS}")
    print(f"  → output: {args.output}")
    print(f"  → report: {args.report}")

    init_data()

    if args.months:
        test_months = [parse_month(m.strip()) for m in args.months.split(",") if m.strip()]
    else:
        test_months = month_range(parse_month(args.start), parse_month(args.end))

    out_path = Path(args.output)
    report_path = Path(args.report)
    existing = []
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    test_date_strs = {str(d) for d in test_months}
    if args.force:
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
            r = run_month(current, past_results=all_results, debate_mode=args.debate_mode)
            new_results.append(r)
            all_results.append(r)  # 다음 월 판정 시 교훈으로 활용
            all_results.sort(key=lambda r: r.get('as_of', ''))
            interim = existing + new_results
            interim.sort(key=lambda r: r.get('as_of', ''))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(interim, f, ensure_ascii=False, indent=2)
            print(f"  💾 중간 저장 완료: {out_path} ({len(interim)}건)")
        except Exception as e:
            print(f"  ❌ {current} 오류: {e}")
            conn.rollback()
            new_results.append({"as_of": str(current), "error": str(e)})

    # 기존 + 신규 합치고 날짜순 정렬
    results = existing + new_results
    results.sort(key=lambda r: r.get('as_of', ''))

    # 결과 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_html(results, report_path)

    print(f"\n결과 저장: {out_path}")
    print(f"리포트 저장: {report_path}")
    print(f"웹뷰 실험 연결: AI_REGIME_RESULTS_PATH={out_path} 로 실행")
    if conn is not None:
        conn.close()


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
