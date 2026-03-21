"""
Step 4: 시총 계산
실행: python scripts/step4_calc_marketcap.py

우선순위:
  ① 기존 daily_price.market_cap > 0 → 유지
  ② DART 발행주식수 × 종가 (이상치 shares_common > 10B 제외)
  ③ 나머지는 market_cap = 0 유지 (Step 4b에서 EV-ND로 보충)

look-ahead bias 방지: disclosure_date 이후 거래일에만 해당 주식수 적용
"""
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

LAB_DB = Path(__file__).parent.parent / "data" / "alpha_lab.db"
MAX_SHARES = 10_000_000_000  # 이상치 기준


def calc_market_cap():
    conn = sqlite3.connect(str(LAB_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    # 1) 기존 market_cap > 0 건수 확인
    existing = conn.execute("SELECT COUNT(*) FROM daily_price WHERE market_cap > 0").fetchone()[0]
    need_fill = conn.execute("SELECT COUNT(*) FROM daily_price WHERE market_cap = 0 OR market_cap IS NULL").fetchone()[0]
    print(f"기존 market_cap > 0: {existing:,}건")
    print(f"채워야 할 건: {need_fill:,}건")

    # 2) DART 발행주식수 로드 (이상치 제외, disclosure_date 기준 정렬)
    shares_data = conn.execute("""
        SELECT stock_code, disclosure_date, shares_common
        FROM shares_outstanding
        WHERE shares_common > 0 AND shares_common <= ?
        ORDER BY stock_code, disclosure_date
    """, (MAX_SHARES,)).fetchall()

    # 종목별 (disclosure_date, shares_common) 리스트
    from collections import defaultdict
    shares_by_stock = defaultdict(list)
    for code, disc_date, shares in shares_data:
        # disclosure_date: YYYYMMDD -> YYYY-MM-DD
        d = f"{disc_date[:4]}-{disc_date[4:6]}-{disc_date[6:8]}"
        shares_by_stock[code].append((d, shares))

    print(f"DART 발행주식수: {len(shares_by_stock)}개 종목")

    # 3) market_cap = 0인 종목별 처리
    updated = 0
    processed_stocks = 0

    for stock_code, share_list in shares_by_stock.items():
        # market_cap = 0인 해당 종목의 거래일 가져오기
        rows = conn.execute("""
            SELECT rowid, trade_date, close FROM daily_price
            WHERE stock_code = ? AND (market_cap = 0 OR market_cap IS NULL) AND close > 0
            ORDER BY trade_date
        """, (stock_code,)).fetchall()

        if not rows:
            continue

        # share_list는 (disclosure_date, shares) 정렬됨
        # 각 거래일에 대해 disclosure_date <= trade_date인 가장 최근 shares 적용
        updates = []
        for rowid, trade_date, close in rows:
            applicable_shares = 0
            for disc_date, shares in share_list:
                if disc_date < trade_date:
                    applicable_shares = shares
                else:
                    break

            if applicable_shares > 0:
                mcap = applicable_shares * close
                updates.append((mcap, rowid))

        if updates:
            conn.executemany("UPDATE daily_price SET market_cap = ? WHERE rowid = ?", updates)
            updated += len(updates)

        processed_stocks += 1
        if processed_stocks % 100 == 0:
            conn.commit()
            print(f"  [{processed_stocks}/{len(shares_by_stock)}] 업데이트: {updated:,}건")

    conn.commit()
    conn.close()

    print(f"\n=== 완료: {updated:,}건 시총 계산 완료 ===")


def main():
    print(f"=== Step 4: 시총 계산 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    calc_market_cap()


if __name__ == "__main__":
    main()
