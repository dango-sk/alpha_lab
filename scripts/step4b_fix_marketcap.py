"""
Step 4b: 시총 보정 (DART 이상치 → EV-ND 대체)
실행: python scripts/step4b_fix_marketcap.py

Step 4 이후 실행. market_cap이 채워진 값 중 이상치를 보정하고, 아직 0인 것을 보충.

로직:
  ① DART로 채워진 시총 vs FnSpace EV-ND 비교 (연말 기준)
     → 비율 0.5~2.0 범위 밖이면 해당 종목-연도의 DART 값을 EV-ND로 대체
  ② market_cap = 0이고 DART 없지만 EV-ND 있으면 → EV-ND로 채움
  ③ 둘 다 없으면 → 0 유지
"""
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

LAB_DB = Path(__file__).parent.parent / "data" / "alpha_lab.db"
RATIO_LOW = 0.5
RATIO_HIGH = 2.0


def fix_market_cap():
    conn = sqlite3.connect(str(LAB_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    # 1) FnSpace EV-ND 로드 (천원 단위 → 원)
    evnd_data = conn.execute("""
        SELECT SUBSTR(stock_code, 2), fiscal_year, (ev - net_debt) * 1000
        FROM fnspace_finance
        WHERE ev IS NOT NULL AND net_debt IS NOT NULL AND (ev - net_debt) > 0
    """).fetchall()

    evnd_by_stock_year = {}
    for code, fy, mcap in evnd_data:
        evnd_by_stock_year[(code, str(fy))] = mcap

    print(f"FnSpace EV-ND: {len(evnd_by_stock_year)}건 (종목-연도)")

    # 2) DART 이상치 보정: 연말 기준 비율 체크
    #    DART로 채워진 종목 (shares_outstanding에 있는 종목)
    dart_stocks = set(r[0] for r in conn.execute(
        "SELECT DISTINCT stock_code FROM shares_outstanding"
    ).fetchall())
    print(f"DART 종목: {len(dart_stocks)}개")

    replaced = 0
    checked_years = 0

    for stock_code in dart_stocks:
        # 연도별 마지막 거래일의 market_cap 가져오기
        year_last = conn.execute("""
            SELECT SUBSTR(trade_date, 1, 4) as year, MAX(trade_date) as last_date
            FROM daily_price
            WHERE stock_code = ? AND market_cap > 0
            GROUP BY SUBSTR(trade_date, 1, 4)
        """, (stock_code,)).fetchall()

        for year, last_date in year_last:
            evnd = evnd_by_stock_year.get((stock_code, year))
            if not evnd or evnd <= 0:
                continue

            dart_mcap = conn.execute(
                "SELECT market_cap FROM daily_price WHERE stock_code = ? AND trade_date = ?",
                (stock_code, last_date)
            ).fetchone()[0]

            checked_years += 1
            ratio = dart_mcap / evnd

            if ratio < RATIO_LOW or ratio > RATIO_HIGH:
                # 해당 종목-연도 전체를 EV-ND로 대체
                cnt = conn.execute("""
                    UPDATE daily_price SET market_cap = ?
                    WHERE stock_code = ? AND SUBSTR(trade_date, 1, 4) = ? AND market_cap > 0
                """, (evnd, stock_code, year)).rowcount
                replaced += cnt

        if checked_years % 500 == 0 and checked_years > 0:
            conn.commit()
            print(f"  체크: {checked_years}건, 대체: {replaced:,}건")

    conn.commit()
    print(f"\n이상치 보정: {checked_years}건 체크, {replaced:,}건 EV-ND로 대체")

    # 3) market_cap = 0이고 EV-ND 있는 종목 보충
    zero_stocks = conn.execute("""
        SELECT DISTINCT stock_code FROM daily_price
        WHERE market_cap = 0 OR market_cap IS NULL
    """).fetchall()
    zero_codes = set(r[0] for r in zero_stocks)
    print(f"\nmarket_cap = 0 종목: {len(zero_codes)}개")

    evnd_filled = 0
    for stock_code in zero_codes:
        rows = conn.execute("""
            SELECT rowid, trade_date FROM daily_price
            WHERE stock_code = ? AND (market_cap = 0 OR market_cap IS NULL)
        """, (stock_code,)).fetchall()

        updates = []
        for rowid, trade_date in rows:
            year = trade_date[:4]
            mcap = evnd_by_stock_year.get((stock_code, year))
            if mcap and mcap > 0:
                updates.append((mcap, rowid))

        if updates:
            conn.executemany("UPDATE daily_price SET market_cap = ? WHERE rowid = ?", updates)
            evnd_filled += len(updates)

    conn.commit()
    conn.close()

    print(f"EV-ND 보충: {evnd_filled:,}건")
    print(f"\n=== 완료: 대체 {replaced:,}건 + 보충 {evnd_filled:,}건 ===")


def main():
    print(f"=== Step 4b: 시총 보정 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    fix_market_cap()


if __name__ == "__main__":
    main()
