"""FCF 재무 수집 단독 실행"""
import sys, os, time, requests
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import psycopg2
from psycopg2.extras import execute_values

PG_URL = os.environ["DATABASE_URL"]
FNSPACE_API_KEY = os.environ.get("FNSPACE_API_KEY", "")
API_DELAY = 0.5

def get_pg():
    conn = psycopg2.connect(PG_URL, connect_timeout=30)
    conn.cursor().execute("SET search_path TO alpha_lab, public")
    conn.commit()
    return conn

FINANCE_ITEMS = {
    "M382500": "pbr", "M211500": "roe", "M211700": "roic",
    "M213000": "ev", "M124000": "ic", "M331010": "ev_ebit",
    "M123100": "ebit", "M113900": "net_debt", "M113800": "interest_debt",
    "M115000": "total_equity", "M312000": "eps", "M314000": "bps",
    "M382100": "per", "M383300": "psr", "M331030": "ev_ebitda",
    "M123200": "ebitda", "M121000": "revenue", "M121500": "operating_income",
    "M122700": "net_income", "M211000": "oi_margin",
    "M431800": "div_yield", "M385110": "pcf", "M124155": "fcf",
}

# FCF만 수집
FCF_ONLY = {"M124155": "fcf"}

COL_NAMES = ["fcf"]
MAX_CODES = 10

if not FNSPACE_API_KEY:
    print("[ERROR] FNSPACE_API_KEY 없음")
    sys.exit(1)

conn = get_pg()
cur = conn.cursor()

# 컬럼 추가
cur.execute("ALTER TABLE alpha_lab.fnspace_finance ADD COLUMN IF NOT EXISTS fcf FLOAT")
conn.commit()
print("[DB] fcf 컬럼 준비 완료")

# 전체 종목
cur.execute("SELECT DISTINCT stock_code FROM alpha_lab.fnspace_master")
all_codes = [r[0] for r in cur.fetchall()]
print(f"전체 종목: {len(all_codes)}개")

code_chunks = [all_codes[i:i+MAX_CODES] for i in range(0, len(all_codes), MAX_CODES)]
total_saved = 0

for ci, code_chunk in enumerate(code_chunks):
    codes_str = ",".join(code_chunk)
    row_dict = {}

    try:
        resp = requests.get("https://www.fnspace.com/Api/FinanceApi", params={
            "key": FNSPACE_API_KEY, "format": "json",
            "code": codes_str, "item": "M124155",
            "consolgb": "M", "annualgb": "A",
            "fraccyear": "2017", "toaccyear": str(datetime.now().year),
        }, timeout=30)
        data = resp.json()
    except Exception as e:
        print(f"  [{ci+1}] API 오류: {e}")
        time.sleep(API_DELAY)
        continue

    if data.get("success") != "true" or not data.get("dataset"):
        time.sleep(API_DELAY)
        continue

    for sd in data["dataset"]:
        code = sd.get("CODE", "")
        for row in sd.get("DATA", []):
            fy = row.get("FS_YEAR")
            fq = row.get("FS_QTR", "Annual")
            if not fy:
                continue
            val = row.get("M124155")
            if val is not None:
                row_dict[(code, fy, fq)] = val

    if row_dict:
        rows = [(k[0], k[1], k[2], v / 1000.0) for k, v in row_dict.items()]
        from psycopg2.extras import execute_values
        execute_values(cur, """
            INSERT INTO alpha_lab.fnspace_finance (stock_code, fiscal_year, fiscal_quarter, fcf)
            VALUES %s
            ON CONFLICT (stock_code, fiscal_year, fiscal_quarter)
            DO UPDATE SET fcf = EXCLUDED.fcf, updated_at = NOW()
        """, rows)
        total_saved += len(rows)
        conn.commit()

    if (ci + 1) % 50 == 0:
        print(f"  [{(ci+1)*MAX_CODES}/{len(all_codes)}종목] 저장: {total_saved:,}건")

    time.sleep(API_DELAY)

conn.commit()
cur.execute("SELECT COUNT(*) FROM alpha_lab.fnspace_finance WHERE fcf IS NOT NULL")
final_cnt = cur.fetchone()[0]
print(f"\n완료: fcf 업데이트 {total_saved:,}건, DB fcf 값 있는 행 {final_cnt:,}건")
conn.close()
