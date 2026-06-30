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

재무 데이터 look-ahead bias 방지:
  - 연간 재무제표는 익년 4월부터 적용 (공시 래그)

백테스트 기간: 2018-01-01 ~
"""
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

# .env 로드 (DATABASE_URL)
for env_path in [
    Path(__file__).parent.parent / ".env",
]:
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"'))

PG_URL = os.environ["DATABASE_URL"]

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


def build_universe(rebuild_all: bool = False):
    """
    유니버스 테이블 채우기.

    - 기본(incremental): 이미 있는 (rebal_date, rebal_type)은 스킵, 새 리밸 날짜만 추가.
      → 다른 사람과 공유 DB에서 안전. 중간에 끊겨도 데이터 보존.
    - rebuild_all=True: 기존 전체 DELETE 후 처음부터 재구축. 과거 데이터까지
      뒤집어 다시 만들어야 할 때만 사용 (fnspace_master 스냅샷 정정 등).
    """
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
    if rebuild_all:
        print("  [rebuild_all=True] 기존 universe 전체 삭제 후 재구축")
        cur.execute("DELETE FROM alpha_lab.universe")
        conn.commit()
        existing = set()
    else:
        cur.execute("SELECT DISTINCT rebal_date, rebal_type FROM alpha_lab.universe")
        existing = {(r[0], r[1]) for r in cur.fetchall()}
        print(f"  [incremental] 기존 {len(existing)}건 → 없는 리밸 날짜만 추가")

    # 1) 리밸런싱 날짜 생성 (monthly: 매월 첫 거래일)
    cur.execute("""
        SELECT SUBSTRING(trade_date FROM 1 FOR 7) AS ym, MIN(trade_date) AS first_day
        FROM alpha_lab.daily_price
        WHERE trade_date >= %s
        GROUP BY SUBSTRING(trade_date FROM 1 FOR 7)
        ORDER BY ym
    """, (START_DATE,))
    monthly_dates = cur.fetchall()

    rebal_dates = []
    for ym, first_day in monthly_dates:
        rebal_dates.append((first_day, "monthly"))

    # ── 예정(forward) 리밸: 최신 데이터 다음 달 1일 ──
    # 그 달 거래일이 아직 없어도(예: 6/30 시점의 7/1) 직전 최신 거래일 데이터로
    # 편입 종목을 미리 산출한다. (실거래: 6/30 종가로 결정 → 7/1 매수)
    if monthly_dates:
        latest_first = monthly_dates[-1][1]          # 최신 월 첫 거래일 (예: 2026-06-01)
        ly, lm = int(latest_first[:4]), int(latest_first[5:7])
        fy, fm = (ly + 1, 1) if lm == 12 else (ly, lm + 1)
        forward_rebal = f"{fy:04d}-{fm:02d}-01"      # 예: 2026-07-01
        if (forward_rebal, "monthly") not in rebal_dates:
            rebal_dates.append((forward_rebal, "monthly"))

        # 최신 월 + 예정 리밸은 데이터가 아직 갱신될 수 있으므로 매 실행 시 재구축한다.
        # (지난달에 6/30 데이터로 만든 7/1 forward → 7/1 실데이터 도착 시 자동 갱신)
        if not rebuild_all:
            cur.execute(
                "DELETE FROM alpha_lab.universe WHERE rebal_type='monthly' AND rebal_date >= %s",
                (latest_first,),
            )
            conn.commit()
            existing = {e for e in existing if not (e[1] == "monthly" and e[0] >= latest_first)}

    todo_rebal_dates = [rd for rd in rebal_dates if rd not in existing]
    print(f"리밸런싱 날짜: 전체 {len(rebal_dates)}건 / 추가 대상 {len(todo_rebal_dates)}건")
    if not todo_rebal_dates:
        print("  추가할 리밸런싱 날짜 없음 (이미 최신)")
        conn.close()
        return

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

    for idx, (rebal_date, rebal_type) in enumerate(todo_rebal_dates):
        applicable_fy = get_applicable_fiscal_year(rebal_date)

        # 데이터 as-of 날짜: 그 리밸 날짜에 거래 데이터가 없으면(예정 리밸)
        # 직전 최신 거래일을 사용한다. (정상 과거 리밸은 rebal_date == 거래일이라 동일)
        cur.execute(
            "SELECT MAX(trade_date) FROM alpha_lab.daily_price WHERE trade_date <= %s",
            (rebal_date,),
        )
        as_of_date = cur.fetchone()[0] or rebal_date
        rebal_ym = rebal_date[:7]

        # fnspace_master 스냅샷: 해당 월 없으면 as-of 월로 fallback (예정 리밸용)
        snapshot = master_by_month.get(rebal_ym) or master_by_month.get(as_of_date[:7], {})
        if not snapshot:
            continue

        # 거래대금 20일 평균 (as-of 직전 40일 범위)
        cur.execute("""
            SELECT stock_code, AVG(trade_amount)
            FROM (
                SELECT stock_code, trade_amount
                FROM alpha_lab.daily_price
                WHERE trade_date <= %s AND trade_date >= TO_CHAR((%s::DATE - INTERVAL '40 days'), 'YYYY-MM-DD')
                  AND trade_amount > 0
            ) sub
            GROUP BY stock_code
        """, (as_of_date, as_of_date))
        avg_trade = {r[0]: r[1] for r in cur.fetchall()}

        # as-of 날짜 market_cap
        cur.execute("""
            SELECT stock_code, market_cap
            FROM alpha_lab.daily_price
            WHERE trade_date = %s AND market_cap > 0
        """, (as_of_date,))
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

        if (idx + 1) % 20 == 0 or idx == len(todo_rebal_dates) - 1:
            conn.commit()
            print(f"  [{idx+1}/{len(todo_rebal_dates)}] {rebal_date} ({rebal_type}): "
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
    import argparse
    parser = argparse.ArgumentParser(description="유니버스 테이블 채우기 (기본: incremental)")
    parser.add_argument("--rebuild-all", action="store_true",
                        help="기존 universe 전체 삭제 후 재구축 (주의: 도중 끊기면 데이터 손실)")
    args = parser.parse_args()

    print(f"=== Step 6: 유니버스 생성 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    build_universe(rebuild_all=args.rebuild_all)


if __name__ == "__main__":
    main()
