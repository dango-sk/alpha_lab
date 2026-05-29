"""JSON 캐시 → Railway DB, 배치+재연결 방식"""
import sys, json, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')
import psycopg2
from pathlib import Path

PG_URL = os.environ["DATABASE_URL"]
CACHE = Path("analysis/data/fcf_annual_cache.json")
BATCH = 500

def connect():
    conn = psycopg2.connect(PG_URL, connect_timeout=30)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET search_path TO alpha_lab, public")
    return conn, cur

conn, cur = connect()

# 컬럼 추가
cur.execute("ALTER TABLE alpha_lab.fnspace_finance ADD COLUMN IF NOT EXISTS fcf FLOAT")
print("[DB] fcf 컬럼 준비 완료")

# 캐시 로드
raw = json.loads(CACHE.read_text(encoding="utf-8"))
rows = []
for k, v in raw.items():
    code, yr = k.split(":")
    rows.append((v / 1000.0, f"A{code}", int(yr)))
print(f"캐시: {len(rows)}건")

# 배치 업데이트
total = 0
for i in range(0, len(rows), BATCH):
    batch = rows[i:i+BATCH]
    for attempt in range(3):
        try:
            cur.executemany(
                "UPDATE alpha_lab.fnspace_finance SET fcf=%s "
                "WHERE stock_code=%s AND fiscal_year=%s AND fiscal_quarter='Annual'",
                batch
            )
            total += len(batch)
            break
        except Exception as e:
            print(f"  [{i}] 재연결... ({e})")
            try: conn.close()
            except: pass
            conn, cur = connect()
    if (i // BATCH + 1) % 10 == 0:
        print(f"  {i+len(batch)}/{len(rows)}건 처리")

conn.autocommit = False
conn.commit()

cur.execute("SELECT COUNT(*) FROM alpha_lab.fnspace_finance WHERE fcf IS NOT NULL")
print(f"\n완료: fcf 있는 행 {cur.fetchone()[0]:,}건")
conn.close()
