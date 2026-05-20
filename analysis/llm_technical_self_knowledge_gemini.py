"""
LLM 기술적 분석 자체지식 검증 (Gemini 버전)
  방식 A (baseline): calc_all_indicators() 30+ 지표 텍스트 → LLM
  방식 B: 종목명 + 300일 OHLCV CSV 원본 → LLM
  방식 C: 종목명 + 기준일만 → LLM (hallucination 확인용)

실행: python analysis/llm_technical_self_knowledge_gemini.py [--date YYYY-MM-DD]
결과: analysis/results/llm_self_knowledge_gemini_{stamp}.xlsx
"""
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai
from google.genai import types
import pandas as pd

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"

from lib.db import get_conn
from lib.ai_stock_filter import (
    _load_price_data,
    _get_stock_name,
    collect_technical_data,
    _parse_json_response,
    TECHNICAL_AGENT_PROMPT,
)
from lib.factor_engine import (
    score_stocks_from_strategy,
    code_to_module,
    DEFAULT_STRATEGY_CODE,
)


# ═══════════════════════════════════════════════════════
# 공통 출력 형식 (A/B/C 동일)
# ═══════════════════════════════════════════════════════

OUTPUT_FORMAT = """
## 출력 형식 (반드시 JSON)
```json
{
  "top_10": [
    {
      "stock_code": "XXXXXX",
      "stock_name": "종목A",
      "score": 85,
      "reason": "판단 근거"
    }
  ],
  "excluded_notable": [
    {
      "stock_code": "YYYYYY",
      "stock_name": "종목B",
      "reason": "제외 근거"
    }
  ]
}
```
score는 0-100 (기술적 매력도). top_10은 score 내림차순."""


# ═══════════════════════════════════════════════════════
# 방식 B 프롬프트
# ═══════════════════════════════════════════════════════

OHLCV_AGENT_PROMPT = """당신은 한국 주식시장 전문 기술적 분석가입니다.
팩터 전략으로 선정된 30종목의 OHLCV(시가/고가/저가/종가/거래량) 원본 데이터를 분석하여,
기술적으로 가장 매력적인 10종목을 선정해야 합니다.

## 중요: 미래 데이터 사용 금지
- 제공된 OHLCV 데이터는 특정 기준일까지의 데이터입니다.
- 기준일 이후의 주가 변동, 뉴스 등 미래 정보를 절대 참조하지 마세요.
- 오직 제공된 OHLCV 데이터만으로 판단하세요.

## 평가 기준
- **추세**: 가격의 방향성, 최근 고점/저점 흐름
- **모멘텀**: 최근 가격 변화 속도와 강도
- **거래량**: 가격 상승 시 거래량 뒷받침 여부, 거래량 추세
- **변동성**: 가격 진폭 안정성
- **가격 위치**: 최근 구간 내 현재 위치 (고점 근처 vs 저점 근처)

## 선정 원칙
- 상승 추세 + 거래량 뒷받침 종목 우선
- 과도한 급등 후 거래량 없이 고점 유지 중인 종목 주의
- 하락 추세 종목은 명확한 반등 근거(거래량 급증 + 저점 상승) 있을 때만 포함
""" + OUTPUT_FORMAT


# ═══════════════════════════════════════════════════════
# 방식 C 프롬프트
# ═══════════════════════════════════════════════════════

NAME_ONLY_AGENT_PROMPT = """당신은 한국 주식시장 전문 기술적 분석가입니다.
팩터 전략으로 선정된 30종목의 종목명과 기준일을 보고,
기술적으로 가장 매력적인 10종목을 선정해야 합니다.

## 중요: 기준일 이전 정보만 사용
- 기준일 이후에 발생한 주가 변동, 뉴스, 실적 등은 절대 참조하지 마세요.
- 기준일 시점까지 알고 있는 정보만으로 판단하세요.

## 평가 기준
- 기준일 시점의 주가 추세 및 모멘텀
- 업종 흐름 및 해당 종목의 기술적 위치
- 최근 수급 및 시장 센티먼트

## 선정 원칙
- 상승 추세 + 모멘텀 강한 종목 우선
- 확실한 근거 없는 종목은 중립(점수 50) 처리
- 정보가 불충분한 경우 솔직하게 reason에 명시
""" + OUTPUT_FORMAT


# ═══════════════════════════════════════════════════════
# 공통 Gemini 호출
# ═══════════════════════════════════════════════════════

def _call_gemini(system_prompt: str, user_msg: str, max_retries: int = 5) -> dict:
    """Gemini API 호출 + 토큰/시간 측정. 503 시 지수 백오프 재시도."""
    from google.genai import errors as genai_errors

    client = genai.Client(api_key=GEMINI_API_KEY)

    t0 = time.time()
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=4096,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            break
        except genai_errors.ServerError:
            wait = 15 * (2 ** attempt)
            print(f"  [Gemini] 503 재시도 {attempt + 1}/{max_retries} — {wait}초 대기...")
            if attempt == max_retries - 1:
                raise
            time.sleep(wait)
    elapsed = round(time.time() - t0, 2)

    # thinking 파트를 건너뛰고 실제 텍스트 파트만 추출
    raw_text = None
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if not getattr(part, "thought", False) and part.text:
                raw_text = part.text
                break
    if raw_text is None:
        raw_text = response.text or ""

    result   = _parse_json_response(raw_text)
    result["_meta"] = {
        "elapsed_sec":   elapsed,
        "input_tokens":  response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
        "model":         GEMINI_MODEL,
    }
    result["_raw"] = {
        "system_prompt": system_prompt,
        "response":      raw_text,
    }
    return result


# ═══════════════════════════════════════════════════════
# 방식 A 함수 (Gemini)
# ═══════════════════════════════════════════════════════

def run_technical_agent_gemini(tech_data: dict) -> dict:
    """방식 A: 계산된 지표 텍스트 → Gemini."""
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

    user_msg = (
        f"다음 30종목의 기술적 지표를 분석하고 Top 10을 선정해주세요.\n\n"
        + "\n\n".join(stock_summaries)
        + "\n\nJSON 형식으로만 응답해주세요."
    )

    return _call_gemini(TECHNICAL_AGENT_PROMPT, user_msg)


# ═══════════════════════════════════════════════════════
# 방식 B 함수
# ═══════════════════════════════════════════════════════

def _format_ohlcv_csv(df: pd.DataFrame, stock_name: str, stock_code: str,
                      factor_score: float) -> str:
    """OHLCV DataFrame을 LLM 입력용 텍스트로 변환."""
    lines = [f"### {stock_name} ({stock_code}) | 팩터점수: {round(factor_score, 4)}"]
    lines.append("date,open,high,low,close,volume")
    for _, row in df.iterrows():
        try:
            lines.append(
                f"{row['trade_date']},"
                f"{int(row['open'])},{int(row['high'])},"
                f"{int(row['low'])},{int(row['close'])},"
                f"{int(row['volume'])}"
            )
        except (ValueError, TypeError):
            continue
    return "\n".join(lines)


def run_ohlcv_agent(conn, stocks: list[tuple[str, float]], calc_date: str) -> dict:
    """방식 B: OHLCV CSV 원본 → Gemini."""
    stock_csvs = []
    skipped = []

    for code, score in stocks:
        name = _get_stock_name(conn, code)
        df   = _load_price_data(conn, code, calc_date)
        if df.empty or len(df) < 60:
            skipped.append(code)
            continue
        stock_csvs.append(_format_ohlcv_csv(df, name, code, score))

    if skipped:
        print(f"  [B] 데이터 부족 스킵: {skipped}")

    sep = "\n" + "─" * 40 + "\n"
    user_msg = (
        f"다음 {len(stock_csvs)}종목의 OHLCV 데이터를 분석하고 "
        f"기술적으로 가장 매력적인 Top 10을 선정해주세요.\n"
        f"기준일: {calc_date}\n\n"
        + sep.join(stock_csvs)
        + "\n\nJSON 형식으로만 응답해주세요."
    )

    return _call_gemini(OHLCV_AGENT_PROMPT, user_msg)


# ═══════════════════════════════════════════════════════
# 방식 C 함수
# ═══════════════════════════════════════════════════════

def run_name_only_agent(conn, stocks: list[tuple[str, float]], calc_date: str) -> dict:
    """방식 C: 종목명 + 기준일만 → Gemini."""
    lines = []
    for i, (code, score) in enumerate(stocks, 1):
        name = _get_stock_name(conn, code)
        lines.append(f"{i}. {name} ({code}) | 팩터점수: {round(score, 4)}")

    user_msg = (
        f"기준일: {calc_date}\n\n"
        f"다음 {len(lines)}종목 중 기술적으로 가장 매력적인 Top 10을 선정해주세요.\n\n"
        + "\n".join(lines)
        + "\n\nJSON 형식으로만 응답해주세요."
    )

    return _call_gemini(NAME_ONLY_AGENT_PROMPT, user_msg)


# ═══════════════════════════════════════════════════════
# 비교
# ═══════════════════════════════════════════════════════

def _compare(result_a: dict, result_x: dict, label: str) -> dict:
    """A방식 vs 다른 방식 Top 10 일치율 계산."""
    codes_a = {item["stock_code"] for item in result_a.get("top_10", [])}
    codes_x = {item["stock_code"] for item in result_x.get("top_10", [])}
    overlap  = codes_a & codes_x

    print(f"\n  [A vs {label}]")
    print(f"  A Top10: {sorted(codes_a)}")
    print(f"  {label} Top10: {sorted(codes_x)}")
    print(f"  겹치는 종목 ({len(overlap)}개): {sorted(overlap)}")
    print(f"  일치율: {len(overlap) / max(len(codes_a), 1):.1%}")

    return {
        "a_top10":        sorted(codes_a),
        f"{label.lower()}_top10": sorted(codes_x),
        "overlap":        sorted(overlap),
        "overlap_count":  len(overlap),
        "agreement_rate": len(overlap) / max(len(codes_a), 1),
    }


# ═══════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════

def run_comparison(calc_date: str = None):
    calc_date = calc_date or datetime.now().strftime("%Y-%m-%d")
    conn = get_conn()

    print(f"\n{'='*60}")
    print(f"  LLM 기술적 분석 자체지식 검증 (Gemini)")
    print(f"  기준일: {calc_date}  모델: {GEMINI_MODEL}")
    print(f"{'='*60}")

    # ── 팩터 상위 30종목 ──
    print("\n[1/5] 팩터 상위 30종목 계산...")
    strategy = code_to_module(DEFAULT_STRATEGY_CODE)
    stocks   = score_stocks_from_strategy(conn, calc_date, strategy)[:30]
    print(f"  {len(stocks)}종목 선정 완료")

    # ── 방식 A ──
    print("\n[2/5] 방식 A 실행 (계산된 지표 텍스트 → Gemini)...")
    t0        = time.time()
    tech_data = collect_technical_data(conn, stocks, calc_date)
    result_a  = run_technical_agent_gemini(tech_data)
    elapsed_a = round(time.time() - t0, 1)
    meta_a    = result_a.get("_meta", {})
    print(f"  완료 ({elapsed_a}초)")
    print(f"  입력 토큰: {meta_a.get('input_tokens'):,}  출력 토큰: {meta_a.get('output_tokens'):,}")
    print(f"  Top 10: {[x['stock_code'] for x in result_a.get('top_10', [])]}")

    # ── 방식 B ──
    print("\n[3/5] 방식 B 실행 (OHLCV CSV → Gemini)...")
    result_b = run_ohlcv_agent(conn, stocks, calc_date)
    meta_b   = result_b.get("_meta", {})
    print(f"  완료 ({meta_b.get('elapsed_sec')}초)")
    print(f"  입력 토큰: {meta_b.get('input_tokens'):,}  출력 토큰: {meta_b.get('output_tokens'):,}")
    print(f"  Top 10: {[x['stock_code'] for x in result_b.get('top_10', [])]}")

    # ── 방식 C ──
    print("\n[4/5] 방식 C 실행 (종목명 + 기준일만 → Gemini)...")
    result_c = run_name_only_agent(conn, stocks, calc_date)
    meta_c   = result_c.get("_meta", {})
    print(f"  완료 ({meta_c.get('elapsed_sec')}초)")
    print(f"  입력 토큰: {meta_c.get('input_tokens'):,}  출력 토큰: {meta_c.get('output_tokens'):,}")
    print(f"  Top 10: {[x['stock_code'] for x in result_c.get('top_10', [])]}")

    # ── 비교 ──
    print(f"\n[5/5] 결과 비교...")
    comp_ab = _compare(result_a, result_b, "B")
    comp_ac = _compare(result_a, result_c, "C")

    # ── xlsx 저장 ──
    out_dir  = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"llm_self_knowledge_gemini_{stamp}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # 요약
        pd.DataFrame({
            "항목": [
                "기준일", "모델",
                "A 소요시간(초)", "B 소요시간(초)", "C 소요시간(초)",
                "A 입력토큰", "A 출력토큰",
                "B 입력토큰", "B 출력토큰",
                "C 입력토큰", "C 출력토큰",
                "A vs B 일치 종목수", "A vs B 일치율",
                "A vs C 일치 종목수", "A vs C 일치율",
            ],
            "값": [
                calc_date, GEMINI_MODEL,
                elapsed_a, meta_b.get("elapsed_sec"), meta_c.get("elapsed_sec"),
                meta_a.get("input_tokens"), meta_a.get("output_tokens"),
                meta_b.get("input_tokens"), meta_b.get("output_tokens"),
                meta_c.get("input_tokens"), meta_c.get("output_tokens"),
                comp_ab["overlap_count"], f"{comp_ab['agreement_rate']:.1%}",
                comp_ac["overlap_count"], f"{comp_ac['agreement_rate']:.1%}",
            ],
        }).to_excel(writer, sheet_name="요약", index=False)

        # 각 방식 Top10
        pd.DataFrame(result_a.get("top_10", [])).to_excel(writer, sheet_name="A_Top10", index=False)
        pd.DataFrame(result_b.get("top_10", [])).to_excel(writer, sheet_name="B_Top10", index=False)
        pd.DataFrame(result_c.get("top_10", [])).to_excel(writer, sheet_name="C_Top10", index=False)

        # 종목별 A vs B vs C 비교
        codes_a = set(comp_ab["a_top10"])
        codes_b = set(comp_ab.get("b_top10", []))
        codes_c = set(comp_ac.get("c_top10", []))
        all_codes = sorted(codes_a | codes_b | codes_c)
        rows = [
            {
                "종목코드": code,
                "A 선정": "O" if code in codes_a else "-",
                "B 선정": "O" if code in codes_b else "-",
                "C 선정": "O" if code in codes_c else "-",
                "A=B":    "O" if code in codes_a and code in codes_b else "-",
                "A=C":    "O" if code in codes_a and code in codes_c else "-",
            }
            for code in all_codes
        ]
        pd.DataFrame(rows).to_excel(writer, sheet_name="비교", index=False)

    print(f"\n  결과 저장: {out_path.relative_to(Path(__file__).parent.parent)}")
    conn.close()
    return {"ab": comp_ab, "ac": comp_ac}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 기술적 분석 자체지식 검증 (Gemini)")
    parser.add_argument("--date", default=None, help="기준일 YYYY-MM-DD (기본: 오늘)")
    args = parser.parse_args()
    run_comparison(args.date)
