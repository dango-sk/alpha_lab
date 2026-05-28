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
        SELECT trade_date, open, high, low, close, volume, trade_amount
        FROM daily_price
        WHERE stock_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
    """
    df = read_sql(sql, conn, params=(stock_code, calc_date, lookback))
    if df.empty:
        return df
    df = df.sort_values("trade_date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "trade_amount"]:
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
        date_constraint = f"\n중요: {calc_date} 이후에 발생한 뉴스, 실적, 주가 변동은 절대 포함하지 마세요. {calc_date} 이전 정보만 사용하세요."

    prompt = f"""한국 주식 종목 "{stock_name}" (종목코드: {stock_code})에 대한 최근 투자 관련 정보를 검색하고 요약해주세요.
{date_constraint}

다음 항목을 각각 정리해주세요:

1. **최근 주요 뉴스 (1개월 이내)**
   - 실적 발표, 소송, 규제, 인수합병, 경영진 변동 등 주가에 영향을 줄 수 있는 이벤트
   - 각 뉴스의 주가 영향 방향 (호재/악재/중립) 표시

2. **증권사 리포트 및 목표가**
   - 최근 증권사 리포트 요약, 투자의견, 목표가
   - 지지선/저항선, 추세 전환 여부

3. **시장 센티먼트**
   - 업종 전체의 흐름과 해당 종목의 위치
   - 수급 동향 (외국인/기관 매매 추이)

각 항목별로 간결하게 3-5줄로 요약해주세요. 모든 문장은 반드시 끝까지 완성해주세요. 중간에 잘리는 문장이 없도록 해주세요. 한국어로 답변해주세요."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3,
                max_output_tokens=4096,
            ),
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini 검색 실패 [{stock_name}]: {e}")
        return f"(검색 실패: {e})"


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

## 가산점 신호
다른 지표들과 종합 판단하되, 아래 신호가 있는 종목은 기술적 매력도에 가산점을 부여할 것.
- 골든상태유지(5>20)가 1인 종목 (5일선이 20일선 위에 있는 상태 유지 중): 단기 상승 추세 지속 신호로 가산점 부여
- 거래대금(60일평균대비배율)이 2.75 이상인 종목: 강한 수급 유입 신호로 가산점 부여

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
- 크로스(5/20): {ind.get('cross_5_20')}, (20/60): {ind.get('cross_20_60')}, (60/120): {ind.get('cross_60_120')}, 골든상태유지(5>20): {ind.get('golden_state_5_20')}
- 신고가: {ind.get('new_high')}, 신저가: {ind.get('new_low')}, 연속일수: {ind.get('consecutive_days')}
- 캔들: 도지={ind.get('doji')}, 망치형={ind.get('hammer')}, 장악형={ind.get('bullish_engulfing')}
- 거래대금(60일평균대비배율): {ind.get('trade_amount_ratio_60d')}"""
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

        # final_portfolio에 가산점 지표 추가
        final_portfolio = consensus_result.get("final_portfolio", [])
        for item in final_portfolio:
            code = item.get("stock_code", "")
            if code in tech_data:
                ind = tech_data[code].get("indicators", {})
                item["golden_state_5_20"] = ind.get("golden_state_5_20")
                item["trade_amount_ratio_60d"] = ind.get("trade_amount_ratio_60d")

        result = {
            "calc_date": calc_date,
            "input_stocks": len(stocks),
            "tech_result": tech_result,
            "news_result": news_result,
            "consensus_result": consensus_result,
            "final_portfolio": final_portfolio,
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
