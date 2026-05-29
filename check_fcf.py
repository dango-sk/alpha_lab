import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')
import psycopg2, os

conn = psycopg2.connect(os.environ['DATABASE_URL'], connect_timeout=10)
cur = conn.cursor()
cur.execute("SET search_path TO alpha_lab, public")
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='fnspace_finance' AND column_name='fcf'")
r = cur.fetchone()
print('fcf 컬럼:', 'EXISTS' if r else 'NOT EXISTS')
if r:
    cur.execute("SELECT COUNT(*) FROM fnspace_finance WHERE fcf IS NOT NULL")
    print('fcf 데이터 건수:', cur.fetchone()[0])
conn.close()
