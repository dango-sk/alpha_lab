"""
종목 기술적 분석 대화형 터미널
LLM 자체 금융 지식으로 기술적 분석 → hallucination 체크용

실행: python analysis/chat_stock.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

import anthropic
import openai

claude_client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
openai_client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

SYSTEM_PROMPT = """당신은 한국 주식 기술적 분석 전문가입니다.
사용자가 종목명이나 종목코드를 주면, 당신이 알고 있는 지식으로 기술적 분석을 수행하세요.

분석 항목:
1. 최근 주가 추세 (상승/하락/횡보)
2. 주요 지지선/저항선
3. 이동평균선 배열 (정배열/역배열)
4. RSI, MACD 등 기술적 지표 상태
5. 거래량 특이점
6. 종합 의견 (매수/매도/관망)

중요:
- 확실하지 않은 정보는 "확인 필요"라고 명시
- 구체적인 수치(가격, RSI 값 등)를 가능한 한 제시
- 데이터 기준일을 명시"""

history = []


def chat(user_input: str, model: str = "claude") -> str:
    history.append({"role": "user", "content": user_input})

    if model == "claude":
        resp = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=history,
        )
        reply = resp.content[0].text
    elif model == "gpt":
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2048,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        )
        reply = resp.choices[0].message.content
    else:
        reply = "지원하지 않는 모델입니다. claude 또는 gpt를 선택하세요."

    history.append({"role": "assistant", "content": reply})
    return reply


def main():
    model = "claude"
    print("=" * 60)
    print("  종목 기술적 분석 대화 터미널")
    print("  명령어: /gpt, /claude (모델 전환), /clear (대화 초기화), /quit")
    print("=" * 60)
    print(f"  현재 모델: {model}\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료.")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/clear":
            history.clear()
            print("  대화 초기화됨.\n")
            continue
        if user_input == "/gpt":
            model = "gpt"
            print(f"  모델 전환: {model}\n")
            continue
        if user_input == "/claude":
            model = "claude"
            print(f"  모델 전환: {model}\n")
            continue

        print(f"\n[{model}] 분석 중...\n")
        reply = chat(user_input, model=model)
        print(reply)
        print()


if __name__ == "__main__":
    main()
