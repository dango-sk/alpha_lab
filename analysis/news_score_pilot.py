"""
analysis/news_score_pilot.py
LLM 뉴스 3-score 파일럿 (2018~2023). date-bound 직전월 news_nate만 입력(lookahead-safe 검색).
점수: macro_tightening / macro_stress / recovery (0~100). 사후지식 금지 프롬프트 + rationale 감사.
이후 hsmm 4/5 · ai_v2 레짐과 divergence 달(2018·2022·2023) 비교 → hsmm 오판 설명력 확인.
사용: DATABASE_URL=<ip> GEMINI_API_KEY=... .venv/bin/python analysis/news_score_pilot.py
"""
import os, re, json, warnings
import numpy as np, pandas as pd, psycopg2
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from google import genai
from google.genai import types as genai_types
warnings.filterwarnings("ignore")
A = Path(__file__).parent
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
MODEL = "gemini-2.5-flash"

PROMPT = """당신은 {ym} 시점의 한국 시장 분석가입니다. 아래는 {prev}에 발행된 뉴스 헤드라인/요약입니다.
**오직 이 텍스트만 근거로** 판단하세요. {ym} 이후 실제로 무슨 일이 있었는지에 대한 어떤 사후지식도 절대 사용하지 마세요(매우 중요).
다음 3개를 0~100 정수로 매기세요:
- macro_tightening: 통화긴축(금리인상/유동성축소/연준 매파/물가급등) 신호 강도
- macro_stress: 거시 위험·불안(경기침체우려/지정학/신용경색/외국인이탈) 강도
- recovery: 경기·시장 회복/반등 기대 강도
JSON만 출력: {{"macro_tightening":N,"macro_stress":N,"recovery":N,"rationale":"근거 한줄(반드시 위 헤드라인 인용)"}}

[{prev} 헤드라인]
{news}"""


def get_news(cur, ym):
    prev = (date.fromisoformat(ym + "-01") - relativedelta(months=1)).strftime("%Y-%m")
    cur.execute("SELECT title, summary FROM alpha_lab.news_nate WHERE published_date LIKE %s ORDER BY published_date DESC LIMIT 40", (prev + "%",))
    rows = cur.fetchall()
    if not rows:
        cur.execute("SELECT title, summary FROM alpha_lab.news WHERE published_date LIKE %s ORDER BY published_date DESC LIMIT 40", (prev + "%",))
        rows = cur.fetchall()
    items = []
    for t, s in rows:
        line = (t or "").strip()
        if s: line += " — " + str(s)[:120]
        items.append("- " + line[:200])
    return prev, "\n".join(items), len(rows)


def score(ym, prev, news):
    last = "ERR"
    for _ in range(2):  # 1회 재시도
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=PROMPT.format(ym=ym, prev=prev, news=news),
                config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=600,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0)))
            txt = (resp.text or "").strip()
            m = re.search(r'\{.*\}', txt, re.S)
            j = json.loads(m.group(0))
            return int(j['macro_tightening']), int(j['macro_stress']), int(j['recovery']), j.get('rationale', '')[:120]
        except Exception as e:
            last = f"ERR {type(e).__name__}"
    return None, None, None, last


def main():
    conn = psycopg2.connect(os.environ['DATABASE_URL']); cur = conn.cursor()
    months = [f"{y}-{m:02d}" for y in range(2018, 2027) for m in range(1, 13) if f"{y}-{m:02d}" <= "2026-06"]
    rows = []
    for ym in months:
        prev, news, n = get_news(cur, ym)
        if n == 0:
            rows.append(dict(ym=ym, n=0, tight=None, stress=None, recov=None, rat="(뉴스없음)")); print(f"  {ym}: 뉴스0", flush=True); continue
        t, s, r, rat = score(ym, prev, news)
        rows.append(dict(ym=ym, n=n, tight=t, stress=s, recov=r, rat=rat))
        print(f"  {ym}: n={n} T={t} S={s} R={r} | {rat[:60]}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(A / "news_scores_2018_2026.csv", index=False)
    conn.close()
    print(f"\n저장: analysis/news_scores_2018_2026.csv ({len(df)}개월, 유효 {df.tight.notna().sum()})")


if __name__ == "__main__":
    main()
