"""
FCF 데이터 DB 가용 범위 확인
실행: python analysis/check_fcf_range.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.db import get_conn, read_sql

conn = get_conn()

# 1. FCF 컬럼 존재 여부
has_fcf = read_sql(
    "SELECT 1 FROM information_schema.columns "
    "WHERE table_name='fnspace_finance' AND column_name='fcf' LIMIT 1", conn
)
if has_fcf.empty:
    print("fcf 컬럼 없음")
    conn.close()
    exit()

# 2. fiscal_quarter별 FCF 비null 첫해/끝해 + 종목 수
summary = read_sql("""
    SELECT fiscal_quarter,
           MIN(fiscal_year) AS first_year,
           MAX(fiscal_year) AS last_year,
           COUNT(DISTINCT stock_code) AS stocks
    FROM fnspace_finance
    WHERE fcf IS NOT NULL
    GROUP BY fiscal_quarter
    ORDER BY fiscal_quarter
""", conn)
print("=== FCF 비null 구간 (fiscal_quarter별) ===")
print(summary.to_string(index=False))

# 3. Annual FCF: 연도별 종목 수
annual = read_sql("""
    SELECT fiscal_year, COUNT(DISTINCT stock_code) AS stocks
    FROM fnspace_finance
    WHERE fiscal_quarter = 'Annual' AND fcf IS NOT NULL
    GROUP BY fiscal_year
    ORDER BY fiscal_year
""", conn)
print("\n=== Annual FCF 연도별 종목 수 ===")
print(annual.to_string(index=False))

# 4. fnspace_finance 전체 데이터 범위 (fcf 무관)
total = read_sql("""
    SELECT fiscal_quarter,
           MIN(fiscal_year) AS first_year,
           MAX(fiscal_year) AS last_year,
           COUNT(DISTINCT stock_code) AS stocks
    FROM fnspace_finance
    GROUP BY fiscal_quarter
    ORDER BY fiscal_quarter
""", conn)
print("\n=== fnspace_finance 전체 범위 (fcf 무관) ===")
print(total.to_string(index=False))

conn.close()
