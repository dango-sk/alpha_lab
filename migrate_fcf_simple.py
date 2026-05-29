import sys, json, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')
import psycopg2
from pathlib import Path

DATABASE_URL = os.environ['DATABASE_URL']
FCF_CACHE = Path('analysis/data/fcf_annual_cache.json')

conn = psycopg2.connect(DATABASE_URL, connect_timeout=30)
conn.autocommit = True
cur = conn.cursor()
cur.execute("SET search_path TO alpha_lab, public")

# 컬럼 추가
try:
    cur.execute("ALTER TABLE fnspace_finance ADD COLUMN fcf FLOAT")
    print("[DB] fcf 컬럼 추가 완료")
except psycopg2.errors.DuplicateColumn:
    print("[DB] fcf 컬럼 이미 존재 (skip)")

# autocommit 끄고 bulk update
conn.autocommit = False
raw = json.loads(FCF_CACHE.read_text(encoding='utf-8'))
print(f"캐시 로드: {len(raw)}건")

updated = 0
batch = []
for k, v in raw.items():
    code, yr = k.split(':')
    batch.append((v / 1000.0, f'A{code}', int(yr)))

cur.executemany(
    "UPDATE fnspace_finance SET fcf=%s WHERE stock_code=%s AND fiscal_year=%s AND fiscal_quarter='Annual'",
    batch
)
updated = cur.rowcount
conn.commit()
print(f"[DB] fcf 업데이트 완료: {updated}건")

cur.execute("SELECT COUNT(*) FROM fnspace_finance WHERE fcf IS NOT NULL")
print(f"[DB] fcf 값 있는 행: {cur.fetchone()[0]}건")
conn.close()
