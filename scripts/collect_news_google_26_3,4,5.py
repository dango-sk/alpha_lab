"""
Gemini Google Search 기반 종목 뉴스 → alpha_lab.news_google 저장

analysis/results/{date}_news_rank.json (2026-03/04/05) 의 _raw.user_prompt 에
들어있는 종목별 news_items(Gemini 수집 원본)를 DB 테이블에 적재한다.
news_item 1건 = 1행. (출력 토큰 한도로 잘린 블록도 완전한 item만 살려서 저장)

실행: python "scripts/collect_news_google_26_3,4,5.py"
"""

import os
import re
import json
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')
DATABASE_URL = os.environ['DATABASE_URL']

RESULTS_DIR = Path(__file__).parent.parent / 'analysis' / 'results'
CALC_DATES = ['2026-03-03', '2026-04-01', '2026-05-04']


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alpha_lab.news_google (
                id             SERIAL PRIMARY KEY,
                calc_date      DATE NOT NULL,
                stock_code     TEXT NOT NULL,
                stock_name     TEXT,
                factor_score   NUMERIC,
                news_date      TEXT,
                source_type    TEXT,
                source_name    TEXT,
                category       TEXT,
                fact           TEXT,
                figures        TEXT,
                price_reaction TEXT,
                created_at     TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (calc_date, stock_code, fact)
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_google_date "
            "ON alpha_lab.news_google (calc_date)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_google_code "
            "ON alpha_lab.news_google (stock_code)"
        )
        conn.commit()
    print("✅ alpha_lab.news_google 테이블 확인/생성 완료")


def _parse_header(header: str):
    """'한국타이어앤테크놀로지 (161390) | 팩터점수: 95.0' → (name, code, factor)"""
    name, code, factor = header, '', None
    m = re.match(r'(.+?)\s*\((\w+)\)', header)
    if m:
        name, code = m.group(1).strip(), m.group(2)
    mf = re.search(r'팩터점수:\s*([\d.]+)', header)
    if mf:
        factor = float(mf.group(1))
    return name, code, factor


def _extract_news_items(part: str):
    """블록 텍스트에서 완전한 news_item 객체만 개별 파싱 (잘림 견딤)"""
    items = []
    idx = part.find('news_items')
    if idx == -1:
        return items
    i = part.find('[', idx)
    if i == -1:
        return items
    n = len(part)
    while i < n:
        ch = part[i]
        if ch == '{':
            depth, j = 0, i
            while j < n:
                if part[j] == '{':
                    depth += 1
                elif part[j] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j < n and depth == 0:
                try:
                    items.append(json.loads(part[i:j + 1]))
                except json.JSONDecodeError:
                    pass
                i = j + 1
            else:
                break  # 미완성 객체 → 중단
        elif ch == ']':
            break
        else:
            i += 1
    return items


def insert_file(conn, calc_date: str):
    path = RESULTS_DIR / f"{calc_date}_news_rank.json"
    if not path.exists():
        print(f"  ⏭️ 파일 없음: {path.name}")
        return 0

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    user_prompt = data.get('_raw', {}).get('user_prompt', '')
    if not user_prompt:
        print(f"  ⚠️ {calc_date}: user_prompt 없음, 건너뜀")
        return 0

    parts = user_prompt.split('### ')[1:]
    inserted, stock_cnt = 0, 0

    with conn.cursor() as cur:
        for part in parts:
            header = part.split('\n')[0].strip()
            name, code, factor = _parse_header(header)
            if not code:
                continue
            items = _extract_news_items(part)
            if items:
                stock_cnt += 1
            for it in items:
                fact = it.get('fact', '')
                if not fact:
                    continue
                cur.execute("""
                    INSERT INTO alpha_lab.news_google (
                        calc_date, stock_code, stock_name, factor_score,
                        news_date, source_type, source_name, category,
                        fact, figures, price_reaction
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (calc_date, stock_code, fact) DO NOTHING
                """, (
                    calc_date, code, name, factor,
                    it.get('date'),
                    it.get('source_type'),
                    it.get('source_name'),
                    it.get('category'),
                    fact,
                    it.get('figures'),
                    it.get('price_reaction'),
                ))
                if cur.rowcount:
                    inserted += 1
        conn.commit()

    print(f"  ✅ {calc_date}: 종목 {stock_cnt}개 → news_item {inserted}건 적재")
    return inserted


def main():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        create_table(conn)
        total = 0
        for d in CALC_DATES:
            total += insert_file(conn, d)
        print(f"\n총 {total}건 적재 완료 → alpha_lab.news_google")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
