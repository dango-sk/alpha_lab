"""
FNspace FCF + DB 시가총액 조회 테스트
M124155 = Free Cash Flow

실행: python analysis/fetch_fcf_test.py
"""
import os
import sys
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.db import get_conn, read_sql

FNSPACE_API_KEY = os.environ.get("FNSPACE_API_KEY", "")
if not FNSPACE_API_KEY:
    print("[ERROR] FNSPACE_API_KEY 없음")
    sys.exit(1)

TEST_CODES = ["A000270", "A004170", "A069960", "A001800", "A002350"]

# ── FNspace FCF 조회 ──────────────────────────────────────────
resp = requests.get("https://www.fnspace.com/Api/FinanceApi", params={
    "key":       FNSPACE_API_KEY,
    "format":    "json",
    "code":      ",".join(TEST_CODES),
    "item":      "M124155",
    "consolgb":  "M",
    "annualgb":  "A",
    "fraccyear": "2017",
    "toaccyear": "2025",
}, timeout=30)

data = resp.json()
if data.get("success") != "true":
    print(f"[ERROR] API 실패: {data}")
    sys.exit(1)

# code -> {fiscal_year -> fcf(원)}
fcf_map: dict[str, dict[int, float]] = {}
for sd in data.get("dataset", []):
    code = sd.get("CODE", "")
    fcf_map[code] = {}
    for row in sd.get("DATA", []):
        fy  = row.get("FS_YEAR")
        fq  = row.get("FS_QTR", "")
        val = row.get("M124155")
        if fy and fq == "Annual" and val is not None:
            fcf_map[code][int(fy)] = float(val)

# ── DB에서 연말 기준 시가총액 조회 ───────────────────────────
# daily_price에서 각 연도 12월 마지막 거래일 시가총액 사용
conn = get_conn()
raw_codes = [c[1:] for c in TEST_CODES]  # 'A' 제거

mktcap_df = read_sql(f"""
    SELECT 'A' || dp.stock_code AS stock_code,
           EXTRACT(YEAR FROM dp.trade_date::date)::int AS year,
           dp.market_cap
    FROM daily_price dp
    INNER JOIN (
        SELECT stock_code, MAX(trade_date) AS last_date
        FROM daily_price
        WHERE stock_code IN ({','.join(f"'{c}'" for c in raw_codes)})
          AND EXTRACT(MONTH FROM trade_date::date) = 12
          AND EXTRACT(YEAR FROM trade_date::date) BETWEEN 2017 AND 2025
        GROUP BY stock_code, EXTRACT(YEAR FROM trade_date::date)
    ) t ON dp.stock_code = t.stock_code AND dp.trade_date = t.last_date
    ORDER BY stock_code, year
""", conn)
conn.close()

# code -> {year -> market_cap}
mktcap_map: dict[str, dict[int, float]] = {}
for _, row in mktcap_df.iterrows():
    code = row["stock_code"]
    year = int(row["year"])
    mc   = float(row["market_cap"])
    mktcap_map.setdefault(code, {})[year] = mc

# ── 출력 ──────────────────────────────────────────────────────
print(f"\n{'종목코드':<10} {'연도':<6} {'FCF (억원)':>12} {'시가총액 (억원)':>16} {'FCF_YIELD':>10}")
print("-" * 60)

for code in TEST_CODES:
    fcf_years = fcf_map.get(code, {})
    mc_years  = mktcap_map.get(code, {})
    years = sorted(set(fcf_years) | set(mc_years))
    for yr in years:
        fcf = fcf_years.get(yr)
        mc  = mc_years.get(yr)
        fcf_eok = fcf / 1e8 if fcf is not None else None
        mc_eok  = mc  / 1e8 if mc  is not None else None
        yield_val = (fcf / mc * 100) if (fcf is not None and mc and mc > 0) else None

        fcf_str   = f"{fcf_eok:>12,.1f}"  if fcf_eok   is not None else f"{'N/A':>12}"
        mc_str    = f"{mc_eok:>16,.1f}"   if mc_eok    is not None else f"{'N/A':>16}"
        yield_str = f"{yield_val:>9.2f}%" if yield_val is not None else f"{'N/A':>10}"
        print(f"{code:<10} {yr:<6} {fcf_str} {mc_str} {yield_str}")
    print()
