"""
Step 6: 유니버스 테이블 생성 (Railway PostgreSQL)
실행: python scripts/step6_build_universe.py

Railway PG의 fnspace_master, fnspace_finance, daily_price를 직접 참조하여
universe 테이블을 PG에 생성.

필터:
  1. fnspace_master 스냅샷 존재 (생존자편향 방지)
  2. market_cap > 0
  3. 금융업 제외 (금융업, 은행업, 보험업, 증권업, 여신전문금융업, 기타금융업)
  4. 스팩/ETF/REIT/ETN 제외 (종목명 키워드)
  5. 부채비율 200% 이하 (interest_debt / total_equity * 100)
  6. 연간 영업이익 > 0
  7. 거래대금 20일 평균 5억 이상

리밸런싱 주기:
  - monthly: 매월 첫 거래일
  - biweekly: 월초 + 월 중간 (15일 이후 첫 거래일)

재무 데이터 look-ahead bias 방지:
  - 연간 재무제표는 익년 4월부터 적용 (공시 래그)

백테스트 기간: 2018-01-01 ~
"""
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

PG_URL = "postgresql://postgres:NgHDMsgiGwbvMpWHLUgTQguaedbGoxvv@metro.proxy.rlwy.net:50087/railway"

FINANCE_TYPES = ["금융업", "은행업", "보험업", "증권업", "여신전문금융업", "기타금융업"]
EXCLUDE_KEYWORDS = ["스팩", "SPAC", "ETF", "ETN", "리츠", "REIT"]
MAX_DEBT_RATIO = 200  # %
MIN_MARKET_CAP = 500_000_000_000  # 5000억
MIN_TRADE_AMOUNT_20D = 500_000_000  # 5억
START_DATE = "2018-01-01"


def get_applicable_fiscal_year(rebal_date):
    """리밸런싱 날짜에 사용 가능한 재무제표 연도 (공시 래그 적용)"""
    year = int(rebal_date[:4])
    month = int(rebal_date[5:7])
    if month >= 4:
        return year - 1
    else:
        return year - 2


def build_universe():
    import psycopg2

    conn = psycopg2.connect(PG_URL)
    cur = conn.cursor()

    # 테이블 생성
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alpha_lab.universe (
            rebal_date  TEXT,
            rebal_type  TEXT,
            stock_code  TEXT,
            market_cap  REAL,
            PRIMARY KEY (rebal_date, rebal_type, stock_code)
        )
    """)
    cur.execute("DELETE FROM alpha_lab.universe")
    conn.commit()

    # 1) 리밸런싱 날짜 생성 (monthly: 매월 첫 거래일)
    cur.execute("""
        SELECT SUBSTRING(trade_date FROM 1 FOR 7) AS ym, MIN(trade_date) AS first_day
        FROM alpha_lab.daily_price
        WHERE trade_date >= %s
        GROUP BY SUBSTRING(trade_date FROM 1 FOR 7)
        ORDER BY ym
    """, (START_DATE,))
    monthly_dates = cur.fetchall()

    # biweekly: 15일 이후 첫 거래일
    cur.execute("""
        SELECT SUBSTRING(trade_date FROM 1 FOR 7) AS ym, MIN(trade_date) AS mid_day
        FROM alpha_lab.daily_price
        WHERE trade_date >= %s AND CAST(SUBSTRING(trade_date FROM 9 FOR 2) AS INT) >= 15
        GROUP BY SUBSTRING(trade_date FROM 1 FOR 7)
        ORDER BY ym
    """, (START_DATE,))
    mid_dates = {r[0]: r[1] for r in cur.fetchall()}

    rebal_dates = []
    for ym, first_day in monthly_dates:
        rebal_dates.append((first_day, "monthly"))
        rebal_dates.append((first_day, "biweekly"))
        if ym in mid_dates:
            rebal_dates.append((mid_dates[ym], "biweekly"))

    print(f"리밸런싱 날짜: {len(rebal_dates)}건")

    # 2) fnspace_master 스냅샷 로드 (stock_code: A prefix)
    cur.execute("""
        SELECT snapshot_date, stock_code, stock_name, finacc_typ
        FROM alpha_lab.fnspace_master
    """)
    master_data = cur.fetchall()

    # {YYYY-MM: {stock_code_6: (name, finacc_typ)}}
    master_by_month = defaultdict(dict)
    for snap, code, name, finacc in master_data:
        code6 = code[1:] if code.startswith("A") else code
        master_by_month[snap][code6] = (name, finacc)

    # 3) 재무 데이터 로드 (stock_code: A prefix → 6자리로 변환)
    cur.execute("""
        SELECT SUBSTRING(stock_code FROM 2), fiscal_year,
               interest_debt, total_equity, operating_income
        FROM alpha_lab.fnspace_finance
        WHERE fiscal_quarter = 'Annual' OR fiscal_quarter IS NULL
    """)
    finance_data = cur.fetchall()

    finance_by = {}
    for code, fy, int_debt, equity, oi in finance_data:
        debt_ratio = None
        if equity and equity > 0 and int_debt is not None:
            debt_ratio = (int_debt / equity) * 100
        finance_by[(code, fy)] = (debt_ratio, oi)

    # 4) 각 리밸런싱 날짜별 유니버스 생성
    total_inserted = 0

    for idx, (rebal_date, rebal_type) in enumerate(rebal_dates):
        rebal_ym = rebal_date[:7]
        applicable_fy = get_applicable_fiscal_year(rebal_date)

        # fnspace_master 스냅샷
        snapshot = master_by_month.get(rebal_ym, {})
        if not snapshot:
            continue

        # 거래대금 20일 평균 (직전 40일 범위)
        cur.execute("""
            SELECT stock_code, AVG(trade_amount)
            FROM (
                SELECT stock_code, trade_amount
                FROM alpha_lab.daily_price
                WHERE trade_date <= %s AND trade_date >= TO_CHAR((%s::DATE - INTERVAL '40 days'), 'YYYY-MM-DD')
                  AND trade_amount > 0
            ) sub
            GROUP BY stock_code
        """, (rebal_date, rebal_date))
        avg_trade = {r[0]: r[1] for r in cur.fetchall()}

        # 해당 날짜 market_cap
        cur.execute("""
            SELECT stock_code, market_cap
            FROM alpha_lab.daily_price
            WHERE trade_date = %s AND market_cap > 0
        """, (rebal_date,))
        mcap_by = {r[0]: r[1] for r in cur.fetchall()}

        inserts = []
        for code6, (name, finacc) in snapshot.items():
            # 금융업 제외
            if finacc in FINANCE_TYPES:
                continue

            # 스팩/ETF/REIT/ETN 제외
            if any(kw in name.upper() for kw in [k.upper() for k in EXCLUDE_KEYWORDS]):
                continue

            # market_cap 하한
            mcap = mcap_by.get(code6)
            if not mcap or mcap < MIN_MARKET_CAP:
                continue

            # 거래대금 20일 평균 5억 이상
            ta = avg_trade.get(code6, 0)
            if ta < MIN_TRADE_AMOUNT_20D:
                continue

            # 재무 필터 (데이터 없으면 통과)
            fin = finance_by.get((code6, applicable_fy))
            if fin:
                debt_ratio, oi = fin
                if debt_ratio is not None and debt_ratio > MAX_DEBT_RATIO:
                    continue
                if oi is not None and oi <= 0:
                    continue

            inserts.append((rebal_date, rebal_type, code6, mcap))

        if inserts:
            from psycopg2.extras import execute_values
            execute_values(cur, """
                INSERT INTO alpha_lab.universe (rebal_date, rebal_type, stock_code, market_cap)
                VALUES %s
                ON CONFLICT (rebal_date, rebal_type, stock_code) DO NOTHING
            """, inserts)
            total_inserted += len(inserts)

        if (idx + 1) % 20 == 0 or idx == len(rebal_dates) - 1:
            conn.commit()
            print(f"  [{idx+1}/{len(rebal_dates)}] {rebal_date} ({rebal_type}): "
                  f"{len(inserts)}종목, 누적: {total_inserted:,}건")

    conn.commit()

    # 요약
    cur.execute("""
        SELECT rebal_type, COUNT(DISTINCT rebal_date), COUNT(*),
               ROUND(AVG(cnt)::NUMERIC, 1)
        FROM (
            SELECT rebal_type, rebal_date, COUNT(*) as cnt
            FROM alpha_lab.universe GROUP BY rebal_type, rebal_date
        ) sub
        GROUP BY rebal_type
    """)
    summary = cur.fetchall()

    print(f"\n=== 완료: {total_inserted:,}건 ===")
    for rtype, dates, total, avg in summary:
        print(f"  {rtype}: {dates}회 리밸런싱, 평균 {avg}종목/회")

    conn.close()


def main():
    print(f"=== Step 6: 유니버스 생성 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    build_universe()


if __name__ == "__main__":
    main()
