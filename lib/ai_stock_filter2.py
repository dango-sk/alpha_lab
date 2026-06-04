"""
AI 종목 필터: 3-에이전트 합의 시스템

팩터 전략 상위 30종목 → AI 필터링 → 최종 10종목 + 비중

에이전트 구조:
  1. 기술적 분석 AI — 기술적 지표 기반 → Top 10
  2. 뉴스 분석 AI   — Gemini Search Grounding (뉴스+기술적 리포트) → Top 10
  3. 합의 AI       — 1,2 결과 + 팩터 원점수 → 최종 10종목 + 비중
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import ANTHROPIC_API_KEY, CACHE_DIR
from lib.db import get_conn, read_sql
from lib.technical_indicators import calc_all_indicators

logger = logging.getLogger(__name__)

# ─── API 클라이언트 ───
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"

LOG_DIR = CACHE_DIR / "ai_filter_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════
# 데이터 수집
# ═══════════════════════════════════════════════════════

def _load_price_data(conn, stock_code: str, calc_date: str, lookback: int = 300) -> pd.DataFrame:
    """종목의 최근 N일 OHLCV 데이터 로드."""
    sql = """
        SELECT trade_date, open, high, low, close, volume
        FROM daily_price
        WHERE stock_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
    """
    df = read_sql(sql, conn, params=(stock_code, calc_date, lookback))
    if df.empty:
        return df
    df = df.sort_values("trade_date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _get_stock_name(conn, stock_code: str) -> str:
    """종목코드 → 종목명."""
    code_a = f"A{stock_code}" if not stock_code.startswith("A") else stock_code
    sql = "SELECT stock_name FROM fnspace_master WHERE stock_code = ? LIMIT 1"
    df = read_sql(sql, conn, params=(code_a,))
    return df.iloc[0]["stock_name"] if not df.empty else stock_code


def collect_technical_data(conn, stocks: list[tuple[str, float]], calc_date: str) -> dict:
    """30종목의 기술적 지표 일괄 계산.

    Parameters
    ----------
    stocks : list of (stock_code, factor_score)
    calc_date : 계산 기준일

    Returns
    -------
    dict : {stock_code: {"name": str, "factor_score": float, "indicators": dict}}
    """
    result = {}
    for code, score in stocks:
        name = _get_stock_name(conn, code)
        df = _load_price_data(conn, code, calc_date)
        if df.empty or len(df) < 60:
            logger.warning(f"[{code}] {name}: 가격 데이터 부족, 건너뜀")
            continue
        indicators = calc_all_indicators(df)
        result[code] = {
            "name": name,
            "factor_score": round(score, 4),
            "indicators": indicators,
        }
    return result


# ═══════════════════════════════════════════════════════
# Gemini Search Grounding: 뉴스 + 기술적 분석 수집
# ═══════════════════════════════════════════════════════

def _init_gemini():
    """Gemini 클라이언트 초기화."""
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)


def search_stock_news(stock_name: str, stock_code: str, calc_date: str = None) -> str:
    """Gemini Search Grounding으로 종목 뉴스 + 기술적 분석 검색.

    Returns
    -------
    str : 검색 결과 요약 텍스트
    """
    if not GEMINI_API_KEY:
        return "(Gemini API 키 미설정)"

    from google import genai
    from google.genai import types

    client = _init_gemini()
    date_constraint = ""
    if calc_date:
        from datetime import datetime, timedelta
        calc_dt = datetime.strptime(calc_date, "%Y-%m-%d")
        start_date = (calc_dt - timedelta(days=30)).strftime("%Y-%m-%d")
        date_constraint = f"\n중요: {start_date}부터 {calc_date} 사이에 발생한 정보만 수집하세요. 이 범위를 벗어난 항목은 절대 포함하지 마세요."

    prompt = f"""당신은 한국 주식시장 뉴스 리서치 수집가입니다.
종목 "{stock_name}" (종목코드: {stock_code})에 대한 투자 관련 정보를 검색하여 '객관적 사실'로만 수집합니다.
{date_constraint}

## 절대 규칙
- 당신의 역할은 '사실 수집'이다. 호재/악재/중립 같은 투자 판단·평결을 내리지 마라. 판단은 다음 단계에서 한다.
- 반드시 실제 검색 결과에 근거하라. 확인되지 않은 수치·사건을 지어내지 마라.
- 각 항목에 보도/발생 날짜(YYYY-MM-DD)를 반드시 기록하라.
- 추측성·루머성 정보는 source_type을 "루머"로, 확정 공시는 "공시"로 명시하라.
- 해당 종목 고유의 의미 있는 정보만 담아라. 일반 시황·업종 일반론은 제외하라.
- 검색에서 찾은 항목은 모두 기재하라. news_items 개수 제한 없음.
- 의미 있는 정보가 없으면 news_items를 빈 배열로 두라. 억지로 채우지 마라.

## 수집 대상 (최근 1개월 이내)
1) 뉴스/이벤트: 실적, 소송, 규제, M&A, 신사업, 경영진 변동 등 — 모든 항목 각각 기재
2) 증권사 의견: 목표가(기존→변경), 투자의견 및 변경 여부, 증권사명 — 모든 증권사 각각 기재
3) 수급: 외국인/기관 순매수·순매도 방향과 기간

## 각 항목 기록 형식
- date: YYYY-MM-DD
- source_type: 공시 | 언론 | 루머
- source_name: 출처명 (예: 전자공시, 한경, OO증권 리포트)
- category: 실적 | 소송 | M&A | 규제 | 신사업 | 목표가변경 | 수급 | 경영진 | 기타
- fact: 무슨 일이 있었는지 사실만 서술 (판단·형용사 배제)
- figures: 관련 수치만 (실적이면 실제치+컨센서스/가이던스, 목표가면 증권사/기존→변경/의견, 수급이면 주체/방향/기간)
- price_reaction: 보도 전후 주가 반응이 자료에 있으면 기재, 없으면 "불명"

## 출력 (JSON만, 그 외 텍스트·마크다운·인용주석 금지)
{{
  "stock_code": "{stock_code}",
  "stock_name": "{stock_name}",
  "news_items": [
    {{
      "date": "2025-05-12",
      "source_type": "언론",
      "source_name": "한경",
      "category": "목표가변경",
      "fact": "OO증권이 목표주가를 상향 조정",
      "figures": "OO증권 / 80,000→100,000 / 매수 유지",
      "price_reaction": "발표 당일 +2% 후 횡보"
    }}
  ]
}}
news_items는 date 내림차순(최신 우선)."""

    import time, re
    attempt = 0
    while True:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.3,
                    max_output_tokens=8192,
                ),
            )
            if not response.text:
                attempt += 1
                wait = min(5 * attempt, 60)
                print(f"  응답 없음, 재시도 {attempt}회... {wait}초 대기")
                time.sleep(wait)
                continue

            raw = response.text
            # 인용 주석 제거 (e.g. [cite: 1], [1])
            cleaned = re.sub(r'\[cite:?\s*\d+\]|\[\d+\]', '', raw).strip()
            # 코드펜스 제거
            cleaned = re.sub(r'```(?:json)?\s*', '', cleaned).strip()
            # 첫 번째 완전한 JSON 객체 추출 (중괄호 depth 추적)
            start = cleaned.find('{')
            if start != -1:
                depth, end = 0, -1
                for i, ch in enumerate(cleaned[start:], start):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                if end != -1:
                    try:
                        parsed = json.loads(cleaned[start:end + 1])
                        return json.dumps(parsed, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        pass
            # 파싱 실패 시 cleaned 텍스트 그대로 반환
            return cleaned

        except Exception as e:
            attempt += 1
            wait = min(5 * attempt, 60)
            print(f"  재시도 {attempt}회... ({e}) {wait}초 대기")
            time.sleep(wait)


def collect_news_data(conn, stocks: dict, calc_date: str = None) -> dict:
    """30종목 뉴스 일괄 수집.

    Parameters
    ----------
    stocks : {stock_code: {"name": str, ...}} from collect_technical_data
    calc_date : 기준일 (이후 뉴스 제외)

    Returns
    -------
    dict : {stock_code: news_summary_text}
    """
    news = {}
    for code, info in stocks.items():
        logger.info(f"뉴스 검색: {info['name']} ({code})")
        result = search_stock_news(info["name"], code, calc_date)
        news[code] = result
        print(f"\n{'─'*60}")
        print(f"[뉴스] {info['name']} ({code})")
        print(f"{'─'*60}")
        print(result)
    return news


# ═══════════════════════════════════════════════════════
# 에이전트 1: 기술적 분석 AI
# ═══════════════════════════════════════════════════════

TECHNICAL_AGENT_PROMPT = """당신은 한국 주식시장 전문 기술적 분석가입니다.
팩터 전략으로 선정된 30종목의 기술적 지표를 분석하여, 기술적으로 가장 매력적인 10종목을 선정해야 합니다.

## 중요: 미래 데이터 사용 금지
- 제공된 기술적 지표는 특정 기준일까지의 데이터로 계산된 것입니다.
- 기준일 이후의 주가 변동, 뉴스, 이벤트 등 미래 정보를 절대 참조하지 마세요.
- 오직 제공된 지표 데이터만으로 판단하세요.

## 평가 기준
- **추세**: 이동평균 배열, 골든/데드크로스, ADX 추세 강도
- **모멘텀**: RSI 과매수/과매도, MACD 히스토그램 방향, 스토캐스틱
- **변동성**: 볼린저밴드 위치, ATR, 역사적 변동성
- **거래량**: 거래량 변화율, OBV 기울기, MFI, CMF
- **가격 위치**: 52주 고점/저점 대비, 피보나치 위치
- **캔들 패턴**: 도지, 망치형, 장악형 등

## 선정 원칙
- 상승 추세 + 적절한 진입 타이밍 (과매수 아닌) 종목 우선
- 거래량 뒷받침이 있는 상승 신호 우선
- 하락 추세 + 과매도 반등 기대도 고려 (단, 근거 명확할 때만)
- 변동성 과도한 종목은 비중 하향

## 출력 형식 (반드시 JSON)
```json
{
  "top_10": [
    {
      "stock_code": "XXXXXX",
      "stock_name": "종목A",
      "score": 85,
      "reason": "20/60일 골든크로스, RSI 55 중립, 거래량 증가세"
    }
  ],
  "excluded_notable": [
    {
      "stock_code": "YYYYYY",
      "stock_name": "종목B",
      "reason": "RSI 78 과매수, 볼린저 상단 이탈"
    }
  ]
}
```
score는 0-100 (기술적 매력도). top_10은 score 내림차순."""


def run_technical_agent(tech_data: dict) -> dict:
    """에이전트 1: 기술적 분석 AI 실행."""
    import anthropic

    # 종목별 지표 요약 텍스트 생성
    stock_summaries = []
    for code, info in tech_data.items():
        ind = info["indicators"]
        summary = f"""### {info['name']} ({code}) | 팩터점수: {info['factor_score']}
- RSI(14): {ind.get('rsi_14')}, Stoch K/D: {ind.get('stoch_k')}/{ind.get('stoch_d')}
- MACD: {ind.get('macd')}, Signal: {ind.get('macd_signal')}, Hist: {ind.get('macd_histogram')}
- ADX: {ind.get('adx_14')}, CCI: {ind.get('cci_20')}, Williams%R: {ind.get('williams_r_14')}
- MA이격도: 5일={ind.get('ma5_gap')}%, 20일={ind.get('ma20_gap')}%, 60일={ind.get('ma60_gap')}%, 120일={ind.get('ma120_gap')}%, 200일={ind.get('ma200_gap')}%
- BB %b: {ind.get('bb_pct_b')}, BW: {ind.get('bb_bandwidth')}, ATR: {ind.get('atr_14')}, HV: {ind.get('hv_20')}
- 거래량비: {ind.get('vol_change_ratio')}, OBV기울기: {ind.get('obv_slope')}, MFI: {ind.get('mfi_14')}, CMF: {ind.get('cmf_20')}
- 52주고점비: {ind.get('week52_high_ratio')}%, 52주저점비: {ind.get('week52_low_ratio')}%, Fib위치: {ind.get('fib_position')}
- 크로스(5/20): {ind.get('cross_5_20')}, (20/60): {ind.get('cross_20_60')}, (60/120): {ind.get('cross_60_120')}
- 신고가: {ind.get('new_high')}, 신저가: {ind.get('new_low')}, 연속일수: {ind.get('consecutive_days')}
- 캔들: 도지={ind.get('doji')}, 망치형={ind.get('hammer')}, 장악형={ind.get('bullish_engulfing')}"""
        stock_summaries.append(summary)

    user_msg = f"""다음 30종목의 기술적 지표를 분석하고 Top 10을 선정해주세요.

{chr(10).join(stock_summaries)}

JSON 형식으로만 응답해주세요."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        system=TECHNICAL_AGENT_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw_text = response.content[0].text
    result = _parse_json_response(raw_text)
    result["_raw"] = {
        "system_prompt": TECHNICAL_AGENT_PROMPT,
        "user_prompt": user_msg,
        "response": raw_text,
        "model": CLAUDE_MODEL,
    }
    return result


# ═══════════════════════════════════════════════════════
# 에이전트 2: 뉴스 분석 AI
# ═══════════════════════════════════════════════════════

NEWS_AGENT_PROMPT = """당신은 한국 주식시장 전문 뉴스/센티먼트 분석가입니다.
팩터 전략으로 선정된 30종목의 최신 뉴스와 시장 의견을 분석하여, 뉴스 관점에서 가장 투자 매력적인 10종목을 선정해야 합니다.

## 중요: 미래 데이터 사용 금지
- 제공된 뉴스/분석 자료는 특정 기준일까지의 정보입니다.
- 기준일 이후에 발생한 실적 발표, 주가 변동, 뉴스 등 미래 정보를 절대 참조하지 마세요.
- 본인이 알고 있는 미래 정보를 판단에 반영하지 마세요. 오직 제공된 자료만 사용하세요.

## 평가 기준
- **악재 여부**: 소송, 횡령, 실적 쇼크, 규제 리스크, 경영진 리스크
- **호재 여부**: 실적 서프라이즈, 신사업 진출, M&A, 정책 수혜
- **증권사 의견**: 목표가 상향/하향, 투자의견 변경
- **업종 흐름**: 섹터 전체 모멘텀과 해당 종목 위치
- **센티먼트**: 시장 참여자들의 전반적 분위기

## 선정 원칙
- 중대 악재 (소송, 횡령, 실적쇼크) 종목은 반드시 제외
- 뉴스 없는 종목은 중립 (점수 50)
- 호재 + 업종 모멘텀 종목 우선 선정
- 단기 이벤트보다 구조적 변화 우선 고려

## 출력 형식 (반드시 JSON)
```json
{
  "top_10": [
    {
      "stock_code": "XXXXXX",
      "stock_name": "종목A",
      "score": 80,
      "reason": "실적 서프라이즈, 목표가 상향 다수"
    }
  ],
  "excluded_notable": [
    {
      "stock_code": "YYYYYY",
      "stock_name": "종목B",
      "reason": "대규모 소송 진행 중, 실적 하향 조정"
    }
  ]
}
```
score는 0-100 (뉴스 관점 투자 매력도). top_10은 score 내림차순."""


def run_news_agent(tech_data: dict, news_data: dict) -> dict:
    """에이전트 2: 뉴스 분석 AI 실행."""
    import anthropic

    stock_summaries = []
    for code, info in tech_data.items():
        news = news_data.get(code, "(뉴스 없음)")
        summary = f"""### {info['name']} ({code}) | 팩터점수: {info['factor_score']}
{news}"""
        stock_summaries.append(summary)

    user_msg = f"""다음 30종목의 최신 뉴스/시장 의견을 분석하고 Top 10을 선정해주세요.

{chr(10).join(stock_summaries)}

JSON 형식으로만 응답해주세요."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        system=NEWS_AGENT_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw_text = response.content[0].text
    result = _parse_json_response(raw_text)
    result["_raw"] = {
        "system_prompt": NEWS_AGENT_PROMPT,
        "user_prompt": user_msg,
        "response": raw_text,
        "model": CLAUDE_MODEL,
    }
    return result


# ═══════════════════════════════════════════════════════
# 에이전트 3: 합의 AI
# ═══════════════════════════════════════════════════════

CONSENSUS_AGENT_PROMPT = """당신은 자산운용사의 CIO(최고투자책임자)입니다.
두 명의 전문 분석가(기술적 분석가, 뉴스 분석가)의 의견과 퀀트 팩터 점수를 종합하여 최종 10종목과 비중을 결정해야 합니다.

## 중요: 미래 데이터 사용 금지
- 모든 판단은 기준일까지의 데이터만으로 이루어져야 합니다.
- 기준일 이후의 주가, 실적, 뉴스 등 미래 정보를 절대 참조하지 마세요.
- 본인이 알고 있는 미래 정보를 판단에 반영하지 마세요.

## 의사결정 원칙
1. **두 에이전트 모두 선정한 종목**: 높은 확신 → 비중 상향
2. **한 에이전트만 선정한 종목**: 선정 이유의 강도에 따라 판단
3. **두 에이전트 모두 제외한 종목**: 원칙적으로 제외 (팩터 점수가 압도적이면 예외)
4. **팩터 점수**: 기본 순위의 근거. 에이전트 의견과 배치될 경우 팩터 점수 높은 쪽 우선
5. **섹터 분산**: 특정 섹터 3종목 이상 편입 지양
6. **비중 배분**: 확신도에 비례 (총합 100%)

## 출력 형식 (반드시 JSON)
```json
{
  "final_portfolio": [
    {
      "stock_code": "XXXXXX",
      "stock_name": "종목A",
      "weight_pct": 15.0,
      "confidence": "high",
      "tech_selected": true,
      "news_selected": true,
      "factor_rank": 1,
      "reason": "기술적+뉴스 모두 긍정, 팩터 상위권"
    }
  ],
  "decision_summary": "양측 합의 N종목 고비중 + 단독 선정 종목 보완. 섹터 분산 및 과매수 종목 제외 고려."
}
```
final_portfolio는 weight_pct 내림차순, 총합 100%."""


def run_consensus_agent(tech_data: dict, tech_result: dict, news_result: dict) -> dict:
    """에이전트 3: 합의 AI 실행."""
    import anthropic

    # 팩터 점수 순위
    factor_ranking = sorted(tech_data.items(), key=lambda x: x[1]["factor_score"], reverse=True)
    factor_rank_text = "\n".join(
        f"{i+1}위. {info['name']} ({code}) — 팩터점수: {info['factor_score']}"
        for i, (code, info) in enumerate(factor_ranking)
    )

    user_msg = f"""## 팩터 점수 순위 (30종목)
{factor_rank_text}

## 기술적 분석 AI 결과
{json.dumps(tech_result, ensure_ascii=False, indent=2)}

## 뉴스 분석 AI 결과
{json.dumps(news_result, ensure_ascii=False, indent=2)}

위 정보를 종합하여 최종 10종목 포트폴리오(비중 합 100%)를 결정해주세요.
JSON 형식으로만 응답해주세요."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        system=CONSENSUS_AGENT_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw_text = response.content[0].text
    result = _parse_json_response(raw_text)
    result["_raw"] = {
        "system_prompt": CONSENSUS_AGENT_PROMPT,
        "user_prompt": user_msg,
        "response": raw_text,
        "model": CLAUDE_MODEL,
    }
    return result


# ═══════════════════════════════════════════════════════
# 뉴스 순위 에이전트 (30종목 전체 점수 + 순위)
# ═══════════════════════════════════════════════════════

NEWS_RANKING_PROMPT = """당신은 한국 주식시장 전문 뉴스/센티먼트 분석가입니다.
팩터 전략으로 이미 선별된 종목들의 뉴스를 분석해, 각 종목에 '뉴스 기반 향후 초과수익 가능성' 점수를 부여하고 횡단(cross-sectional) 순위를 매깁니다.

────────────────────────────────────────
## 0. 절대 규칙 (위반 시 분석 무효)
- 오직 입력으로 제공된 자료에 '명시된' 사실만 사용한다. 자료에 없는 수치·사건을 추론하거나 너의 학습 지식으로 보충하지 마라.
- 기준일 이후의 실적·주가·뉴스 등 미래 정보를 절대 참조하지 마라. 결과를 아는 상태로 역산하지 마라.
- 출처 신뢰도 위계: 전자공시(DART 등 확정 공시) > 언론 보도 > 추측성 기사·루머. 루머성 정보는 점수에 거의 반영하지 마라.
- 입력에는 각 종목의 뉴스 목록과 기준일(as_of)이 포함된다. 최신성 판단은 as_of를 기준으로 한다.
- **입력으로 받은 모든 종목(### 로 시작하는 종목 블록)을 단 하나도 빠짐없이 rankings에 포함하라.** 입력 종목 수 N과 출력 rankings 길이는 반드시 같아야 한다. 뉴스가 없거나 부실한 종목도 절대 생략하지 말고 50점(중립, confidence '하')으로 반드시 포함하라. 토큰을 아끼려고 종목을 누락하는 것은 분석 무효 사유다.

## 1. 채점 철학
- 점수는 '뉴스가 얼마나 좋은가'가 아니라 '아직 가격에 반영 안 된 초과수익 여지가 있는가'다.
- 호재의 절대 강도가 아니라 시장 기대/컨센서스 대비 '서프라이즈'를 평가한다. 서프라이즈는 부호뿐 아니라 '크기'로 가중하라 (컨센서스 소폭 상회와 대폭 상회를 구분).
- 이미 그 뉴스로 주가가 크게 움직였다면 호재여도 감점한다 (사후 추격 방지). price_reaction이 '불명'이면, 최근(수일 내) 뉴스는 priced_in=false로 두되 기준일에서 멀면 priced_in=true 쪽으로 보수적으로 추정하라. 근거 없는 과도한 감점은 피한다.
- 목표가·이익추정치는 '수준'이 아니라 상향/하향 '변화 방향과 폭, 동의 폭(애널리스트 수)'에 가중한다.
- 테마성 급등·과열된 관심은 가점이 아니라 주의 신호로 처리한다 (리테일 주도·공매도 제약 시장에서 후행 언더퍼폼 경향).
- 뉴스 에이전트의 1차 역할은 '지뢰 제거'다. 뉴스는 악재 변별에 강하고 호재 예측에 약하다. 호재로 순위를 크게 흔들지 마라.

## 2. 팩터와의 직교성 (중복 계상 금지)
- 이 종목들은 이미 정량 팩터(모멘텀·밸류 등)로 선별됐다. '최근 주가가 올랐다' 같은 가격 정보 자체에는 가점하지 마라. 그건 팩터가 이미 포착했다.
- 뉴스 에이전트는 '가격에 아직 드러나지 않은 이벤트성·구조적 정보'만 추가한다.

## 3. 업종 베타 vs 종목 알파
- 정책·금리·환율처럼 업종 전체에 작용하는 뉴스는 개별 종목 알파가 아니라 베타다. 업종 공통 호재는 종목 간 변별에 쓰지 마라.
- 같은 업종 호재 속에서도 '그 종목 고유의(idiosyncratic)' 차별적 뉴스가 있는지로 변별하라.
- 입력 앞부분에 [시장 매크로 배경]이 주어지면, 이는 모든 종목에 공통으로 작용하는 시장 환경(금리·환율·유가·지정학)이다.
- 이를 개별 종목 점수의 직접 가/감점 사유로 쓰지 마라. 매크로 민감도로 점수를 조정하는 것(예: '금리 오르니 차입 많은 종목 감점')은 베타이며, 이미 정량 팩터가 포착하므로 중복 계상이다.
- 매크로는 오직 '그 종목의 고유 뉴스 한 건'을 해석하고 중요도를 판단할 때의 배경으로만 쓴다 (예: 한 기업이 회사채 차환 성공을 공시했을 때, 금리 급등 국면이면 그 호재의 가치를 더 높게 평가).
- 매크로 적용에 필요한 기업 특성(차입 규모, 수출 비중 등)이 제공 자료에 없으면, 학습 지식으로 추론하지 말고 적용하지 마라.

## 4. 이벤트 평가 기준
- 악재(점수 하향): 소송·횡령·분식, 실적 쇼크(기대 대비 하회), 규제·경영진 리스크, 추정치 하향. 중대 악재는 사실상 제외 후보.
- 호재(점수 상향, 보수적): 컨센서스 상회 서프라이즈, 추정치·목표가 상향 모멘텀, 구조적 신사업·정책 수혜 중 '종목 고유'인 것.
- 실적 구분: 컨센서스 대비 -10% 이상 하회는 '실적 쇼크'(중대 악재, ≤15), 소폭 하회는 '실적 하회'(25 부근)로 구분한다.
- 중요도(materiality): 실적·지배구조·대형 M&A는 크게, 사소한 단일 계약·IR성 보도는 점수를 거의 움직이지 마라.
- 최신성: 각 뉴스의 date를 기준일(as_of)과 비교해 판단하라. as_of에서 멀수록 이미 반영됐을 가능성이 높으니 가중을 낮춘다.
- 상충 처리: 한 종목에 호재·악재가 공존하면 중요도와 최신성으로 net하라. 단 중대 악재(횡령·분식·대규모 소송·실적 쇼크)가 하나라도 있으면 다른 호재와 무관하게 점수 상한을 15로 제한한다.

## 5. 점수 앵커 (업사이드 보수적, 다운사이드 변별력 넓게 / 정수)
- 75: 컨센서스 크게 상회 + 다수 목표가 상향 + 발표 후 주가 반응 미미(미반영) + 구조적
- 65: 분명한 호재이나 일부 이미 반영, 추정치 소폭 상향
- 55: 약한 호재 또는 호재가 대부분 반영됨
- 50: 뉴스 없음 또는 중립 (기본값. 이미 팩터 통과했으므로 페널티 없음)
- 40: 경미한 악재 (소송 피소 초기, 추정치 소폭 하향)
- 25: 실적 하회 + 다수 목표가 하향
- 15 이하: 횡령·분식·대규모 소송·실적 쇼크 (제외 후보)
- 상단(75 초과)은 정말 예외적인 강한 미반영 서프라이즈에만 허용한다.

## 6. 불확실성 수축
- 근거가 빈약하거나 출처가 루머성이면 confidence를 '하'로 두고 점수를 50 쪽으로 끌어당겨라(shrink). 확정 공시 기반이면 '상'.

## 7. 채점 절차 (반드시 이 순서)
0) 먼저 입력에서 ### 로 시작하는 종목 블록을 모두 세어 입력 종목 수 N을 확인한다. 채점 대상 종목 코드 목록을 작성한다.
1) 각 종목의 뉴스에서 개별 이벤트를 추출·분류한다 (events).
2) 위 기준으로 이벤트를 종합해 점수를 산정한다.
3) N개 종목을 서로 비교해 상대적으로 점수가 일관되는지 재조정한다 (횡단 캘리브레이션).
4) score 내림차순으로 정렬해 rank를 부여한다. 동일 점수는 공동 순위(같은 rank)로 둔다. 정보가 없어 구분이 안 되는 종목을 인위적으로 줄 세우지 마라.
5) 0)에서 작성한 종목 목록과 rankings를 대조해 빠진 종목이 없는지 확인한다. 빠진 종목이 있으면 50점으로 추가한 뒤 출력한다.

## 8. 출력 형식 (JSON만 출력, 그 외 텍스트·마크다운 금지)
{
  "rankings": [
    {
      "rank": 1,
      "stock_code": "XXXXXX",
      "stock_name": "종목A",
      "events": [
        {
          "type": "호재|악재|중립",
          "category": "실적|소송|M&A|규제|신사업|목표가변경|수급|기타",
          "surprise": "상회|부합|하회|불명",
          "priced_in": true,
          "materiality": "상|중|하",
          "source_type": "공시|기사|루머",
          "summary": "한 줄 요약 (자료에 명시된 사실만)"
        }
      ],
      "score": 68,
      "confidence": "상|중|하",
      "already_priced_in": false,
      "reason": "기대 대비 서프라이즈 여부와 반영 여부를 반드시 포함한 1~2문장"
    }
  ],
  "data_quality": "뉴스 없음 종목 수, 루머성 출처 비중 등 데이터 품질 플래그 1~2문장"
}

## 9. 출력 전 자가검증 (체크 후 출력)
- **rankings 길이 == 입력 종목 수 N. 입력 종목 코드와 출력 종목 코드를 1:1 대조해 누락·중복이 없음을 반드시 확인하라. 하나라도 빠지면 출력하지 말고 채워서 다시 검증하라.**
- rank는 score 내림차순과 일관되며, 동일 점수는 공동 순위(억지로 가르지 않음)
- score는 정수, reason에 서프라이즈·반영 여부 명시
- 자료에 없는 수치·사건 없음, 미래 정보 없음
- JSON 외 텍스트·코드펜스 없음"""


def run_news_ranking_agent(tech_data: dict, news_data: dict, calc_date: str = None,
                           macro_context: str = None) -> dict:
    """뉴스 기반 전체 종목 점수 + 순위 산출 (30종목 → 30개 score).

    Parameters
    ----------
    tech_data : {stock_code: {"name": str, "factor_score": float}}
    news_data : {stock_code: news_summary_text}
    calc_date : 기준일(as_of). 최신성 판단 기준으로 프롬프트에 전달.
    macro_context : 시장 전체 매크로 뉴스 배경(금리/환율/유가/지정학). 공통 배경 참고용.

    Returns
    -------
    dict : {"rankings": [...], "data_quality": str}
    """
    import anthropic

    stock_summaries = []
    for code, info in tech_data.items():
        news = news_data.get(code, "(뉴스 없음)")
        summary = f"### {info['name']} ({code}) | 팩터점수: {info['factor_score']}\n{news}"
        stock_summaries.append(summary)

    as_of_line = f"[기준일(as_of)] {calc_date}\n\n" if calc_date else ""
    macro_block = ""
    if macro_context:
        macro_block = (
            "[시장 매크로 배경 — 공통 참고용]\n"
            "아래는 기준일 직전 약 1개월간의 시장 전체 매크로 뉴스 제목입니다. "
            "개별 종목 점수의 직접 가/감점 사유로 쓰지 말고(업종 베타이므로 종목 간 변별·매크로 민감도 조정에 사용 금지), "
            "종목 고유 뉴스 한 건을 해석·중요도 판단할 때의 시장 환경 맥락으로만 참고하세요. "
            "매크로 적용에 필요한 기업 특성이 종목 자료에 없으면 추론하지 말고 적용하지 마세요.\n"
            f"{macro_context}\n\n"
        )
    user_msg = (
        f"{as_of_line}"
        f"{macro_block}"
        f"다음 {len(tech_data)}종목의 최신 뉴스/시장 의견을 분석하고 "
        f"각 종목에 점수를 부여한 뒤 순위를 매겨주세요.\n\n"
        f"{chr(10).join(stock_summaries)}\n\n"
        f"JSON 형식으로만 응답해주세요."
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=NEWS_RANKING_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw_text = response.content[0].text
    result = _parse_json_response(raw_text)

    # factor_score 보완 (프롬프트에서 누락된 경우)
    for item in result.get("rankings", []):
        if item.get("factor_score") is None:
            code = item.get("stock_code", "")
            if code in tech_data:
                item["factor_score"] = tech_data[code]["factor_score"]

    result["_raw"] = {
        "system_prompt": NEWS_RANKING_PROMPT,
        "user_prompt": user_msg,
        "response": raw_text,
        "model": CLAUDE_MODEL,
    }
    return result


# ═══════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════

def _parse_json_response(text: str) -> dict:
    """AI 응답에서 JSON 추출."""
    # ```json ... ``` 블록 추출
    import re
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        logger.error(f"JSON 파싱 실패: {text[:200]}")
        return {"error": "JSON 파싱 실패", "raw": text}


def _save_log(calc_date: str, data: dict):
    """AI 필터 결과 로그 저장."""
    log_path = LOG_DIR / f"ai_filter_{calc_date}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"로그 저장: {log_path}")


# ═══════════════════════════════════════════════════════
# 메인 파이프라인
# ═══════════════════════════════════════════════════════

def run_ai_filter(
    stocks: list[tuple[str, float]],
    calc_date: str,
    conn=None,
) -> dict:
    """AI 종목 필터 전체 파이프라인 실행.

    Parameters
    ----------
    stocks : list of (stock_code, factor_score)
        팩터 전략 상위 30종목 (score 내림차순)
    calc_date : str
        리밸런싱 기준일 (YYYY-MM-DD)
    conn : DB connection (None이면 자동 생성)

    Returns
    -------
    dict : {
        "final_portfolio": [...],  # 최종 10종목 + 비중
        "tech_result": {...},
        "news_result": {...},
        "consensus_result": {...},
    }
    """
    close_conn = False
    if conn is None:
        conn = get_conn()
        close_conn = True

    try:
        logger.info(f"=== AI 종목 필터 시작 ({calc_date}) ===")

        # Step 1: 기술적 지표 계산
        logger.info("Step 1/4: 기술적 지표 계산...")
        tech_data = collect_technical_data(conn, stocks, calc_date)
        logger.info(f"  {len(tech_data)}종목 지표 계산 완료")

        # Step 2: 뉴스 수집 (Gemini Search Grounding)
        logger.info("Step 2/4: 뉴스 수집 (Gemini Search)...")
        news_data = collect_news_data(conn, tech_data, calc_date)
        logger.info(f"  {len(news_data)}종목 뉴스 수집 완료")

        # Step 3: 에이전트 1,2 병렬 실행 (현재는 순차)
        logger.info("Step 3/4: AI 에이전트 분석...")
        tech_result = run_technical_agent(tech_data)
        logger.info("  기술적 분석 AI 완료")
        news_result = run_news_agent(tech_data, news_data)
        logger.info("  뉴스 분석 AI 완료")

        # Step 4: 합의 AI
        logger.info("Step 4/4: 합의 AI 최종 결정...")
        consensus_result = run_consensus_agent(tech_data, tech_result, news_result)
        logger.info("  합의 AI 완료")

        result = {
            "calc_date": calc_date,
            "input_stocks": len(stocks),
            "tech_result": tech_result,
            "news_result": news_result,
            "consensus_result": consensus_result,
            "final_portfolio": consensus_result.get("final_portfolio", []),
        }

        # 로그 저장
        _save_log(calc_date, result)

        logger.info(f"=== AI 종목 필터 완료: {len(result['final_portfolio'])}종목 선정 ===")
        return result

    finally:
        if close_conn:
            conn.close()


# ═══════════════════════════════════════════════════════
# CLI 실행
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="AI 종목 필터 실행")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="기준일 (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=30, help="입력 종목 수")
    parser.add_argument("--strategy", default=None, help="전략 파일 경로")
    args = parser.parse_args()

    conn = get_conn()

    # 전략에서 상위 30종목 가져오기
    from lib.factor_engine import score_stocks_from_strategy, load_strategy_module, code_to_module, DEFAULT_STRATEGY_CODE

    if args.strategy:
        strategy = load_strategy_module(args.strategy)
    else:
        strategy = code_to_module(DEFAULT_STRATEGY_CODE)

    stocks = score_stocks_from_strategy(conn, args.date, strategy)
    stocks = stocks[:args.top_n]

    print(f"\n팩터 상위 {len(stocks)}종목 → AI 필터 시작\n")
    result = run_ai_filter(stocks, args.date, conn)

    # ── 에이전트별 프롬프트 & 응답 출력 ──
    for agent_name, key in [("기술적 분석 AI", "tech_result"), ("뉴스 분석 AI", "news_result"), ("합의 AI", "consensus_result")]:
        agent_data = result.get(key, {})
        raw = agent_data.get("_raw", {})
        print(f"\n{'='*80}")
        print(f"  [{agent_name}] 모델: {raw.get('model', '?')}")
        print(f"{'='*80}")
        print(f"\n--- SYSTEM PROMPT ---")
        print(raw.get("system_prompt", "(없음)"))
        print(f"\n--- USER PROMPT ---")
        print(raw.get("user_prompt", "(없음)")[:3000])
        if len(raw.get("user_prompt", "")) > 3000:
            print(f"  ... (총 {len(raw['user_prompt'])}자, 3000자까지 표시)")
        print(f"\n--- AI 응답 ---")
        print(raw.get("response", "(없음)"))

    # ── 최종 포트폴리오 ──
    print(f"\n{'='*80}")
    print("  최종 포트폴리오")
    print(f"{'='*80}")
    for item in result["final_portfolio"]:
        print(f"  {item.get('stock_name', '?'):>10s} ({item['stock_code']}) "
              f"| 비중: {item.get('weight_pct', 0):5.1f}% "
              f"| {item.get('reason', '')}")

    conn.close()
