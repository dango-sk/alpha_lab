"""
Step 3: 주가 보충 (2017~2019 + 누락 종목)
실행: python scripts/step3_fill_prices.py

- fnspace_master 기준 종목만 대상
- pykrx 종목별 get_market_ohlcv로 OHLCV 수집
- 이미 있는 데이터는 스킵 (market_cap 포함 기존 데이터 유지)
- 새로 추가하는 건 market_cap = 0 (Step 4에서 계산)

소요 시간: 종목별 0.5초 딜레이 × ~4,000종목 = 약 30분~1시간
"""
import sqlite3
import sys
import time
from pathlib import Path
from datetime import datetime

from pykrx import stock as pykrx

sys.path.append(str(Path(__file__).parent.parent))

LAB_DB = Path(__file__).parent.parent / "data" / "alpha_lab.db"

COLLECT_START = "20170101"
COLLECT_END = datetime.now().strftime("%Y%m%d")
API_DELAY = 0.5


def collect_prices():
    conn = sqlite3.connect(str(LAB_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    # fnspace_master 종목 (6자리)
    target_stocks = conn.execute(
        "SELECT DISTINCT SUBSTR(stock_code, 2) FROM fnspace_master"
    ).fetchall()
    target_list = sorted(set(r[0] for r in target_stocks))
    print(f"대상 종목: {len(target_list)}개 (fnspace_master 기준)")

    # 종목별 이미 있는 날짜 범위 확인
    existing_counts = {}
    for row in conn.execute(
        "SELECT stock_code, COUNT(*) FROM daily_price GROUP BY stock_code"
    ).fetchall():
        existing_counts[row[0]] = row[1]

    # 이미 있는 (stock_code, trade_date) 쌍을 빠르게 체크하기 위해
    existing_pairs = set()
    for row in conn.execute("SELECT stock_code, trade_date FROM daily_price").fetchall():
        existing_pairs.add((row[0], row[1]))

    print(f"기존 daily_price: {len(existing_pairs)}건")

    total_new = 0
    total_skip = 0
    errors = 0

    for i, ticker in enumerate(target_list):
        try:
            df = pykrx.get_market_ohlcv(COLLECT_START, COLLECT_END, ticker)
            if df.empty:
                time.sleep(API_DELAY)
                continue
        except Exception as e:
            errors += 1
            if errors % 10 == 0:
                print(f"  [에러] {ticker}: {e}")
            time.sleep(API_DELAY)
            continue

        new_count = 0
        for idx, row in df.iterrows():
            trade_date = idx.strftime("%Y-%m-%d")

            if (ticker, trade_date) in existing_pairs:
                total_skip += 1
                continue

            close = row.get("종가", 0)
            if close == 0:
                continue

            conn.execute("""
                INSERT OR IGNORE INTO daily_price
                (stock_code, trade_date, open, high, low, close, volume, trade_amount, market_cap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                ticker, trade_date,
                row.get("시가", 0), row.get("고가", 0), row.get("저가", 0),
                close, row.get("거래량", 0), row.get("거래대금", 0),
            ))
            new_count += 1

        total_new += new_count
        time.sleep(API_DELAY)

        if (i + 1) % 50 == 0 or i == len(target_list) - 1:
            conn.commit()
            print(f"  [{i+1}/{len(target_list)}] {ticker} | 신규: {total_new}, 스킵: {total_skip}, 에러: {errors}")

    conn.commit()

    # 거래대금 보충: trade_amount가 0인 행을 volume × close로 채움
    filled = conn.execute("""
        UPDATE daily_price
        SET trade_amount = volume * close
        WHERE (trade_amount = 0 OR trade_amount IS NULL)
          AND volume > 0 AND close > 0
    """).rowcount
    conn.commit()
    print(f"거래대금 보충: {filled:,}건 (volume × close)")

    conn.close()
    print(f"\n=== 완료: 신규 {total_new}건, 스킵 {total_skip}건, 에러 {errors}건 ===")


def main():
    print(f"=== Step 3: 주가 보충 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    collect_prices()


if __name__ == "__main__":
    main()
