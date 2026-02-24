"""
Step 1: 주가 업데이트 (alpha_radar DB에 최신 주가 반영)
실행: python scripts/step1_update_prices.py

alpha_radar의 step1 경량 버전:
  - 종목 마스터는 이미 있다고 가정 (alpha_radar에서 초기 수집 완료)
  - 최근 10일 주가만 보완 (일일 업데이트용)
  - 벤치마크 ETF도 함께 업데이트

소요 시간: 약 10~20분
"""
import sqlite3
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

from pykrx import stock as pykrx
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DB_PATH

# alpha_radar 유니버스 기준 (settings.py와 동일)
MIN_MARKET_CAP = 50_000_000_000  # 시총 500억 이상

BENCHMARK_ETFS = {
    "292150": "KODEX KRX300",
    "069500": "KODEX 200",
    "229200": "KODEX 코스닥150",
}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def get_recent_trade_date():
    """최근 거래일 찾기"""
    today = datetime.now()
    for i in range(10):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y%m%d")
        try:
            df = pykrx.get_market_cap(date_str, market="KOSPI")
            if len(df) > 0 and df["시가총액"].sum() > 0:
                return date_str
        except Exception:
            continue
    return (today - timedelta(days=3)).strftime("%Y%m%d")


def update_prices():
    """최근 10일 주가 업데이트"""
    print("\n" + "=" * 60)
    print("Step 1: 주가 업데이트")
    print(f"   DB: {DB_PATH}")
    print("=" * 60)

    conn = get_db()
    trade_date = get_recent_trade_date()
    print(f"  최근 거래일: {trade_date}")

    # DB에서 마지막 수집일 확인
    last_date = conn.execute("""
        SELECT MAX(trade_date) FROM daily_price
        WHERE stock_code IN (SELECT stock_code FROM stock_master WHERE market_cap >= ?)
    """, (MIN_MARKET_CAP,)).fetchone()[0]
    print(f"  DB 마지막 주가: {last_date}")

    # 이미 최신이면 스킵
    trade_date_fmt = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    existing = conn.execute("""
        SELECT COUNT(DISTINCT stock_code) FROM daily_price
        WHERE trade_date = ?
    """, (trade_date_fmt,)).fetchone()[0]

    if existing > 500:
        print(f"  이미 {existing}종목 수집됨 ({trade_date_fmt}) -> 스킵")
        conn.close()
        return

    # 대상 종목
    stocks = conn.execute("""
        SELECT stock_code, stock_name FROM stock_master
        WHERE market_cap >= ?
    """, (MIN_MARKET_CAP,)).fetchall()
    print(f"  대상: {len(stocks)}종목")

    # 최근 10일 수집 (빠진 날 보완)
    today = datetime.now()
    start_date = (today - timedelta(days=15)).strftime("%Y%m%d")
    end_date = trade_date

    success, fail = 0, 0
    for code, name in tqdm(stocks, desc="  주가 수집"):
        try:
            df = pykrx.get_market_ohlcv(start_date, end_date, code)
            if df.empty:
                fail += 1
                continue

            cap_df = pykrx.get_market_cap(start_date, end_date, code)

            records = []
            for date_idx in df.index:
                date_str = date_idx.strftime("%Y-%m-%d")
                row = df.loc[date_idx]

                trade_amount = 0
                mkt_cap = 0
                if date_idx in cap_df.index:
                    trade_amount = int(cap_df.loc[date_idx, "거래대금"])
                    mkt_cap = int(cap_df.loc[date_idx, "시가총액"])

                records.append((
                    code, date_str,
                    int(row["시가"]), int(row["고가"]),
                    int(row["저가"]), int(row["종가"]),
                    int(row["거래량"]),
                    trade_amount, mkt_cap,
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO daily_price
                (stock_code, trade_date, open, high, low, close, volume, trade_amount, market_cap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            success += 1
        except Exception:
            fail += 1
            continue

        time.sleep(0.2)

    conn.commit()
    print(f"\n  주가 업데이트 완료 (성공: {success}, 실패: {fail})")

    # 벤치마크 ETF 업데이트
    update_benchmark_etfs(conn, start_date, end_date)

    # stock_master 시총 갱신
    update_master_market_cap(conn, trade_date)

    conn.close()


def update_benchmark_etfs(conn, start_date, end_date):
    """벤치마크 ETF 최근 주가 업데이트"""
    print("\n  벤치마크 ETF 업데이트...")

    for etf_code, etf_name in BENCHMARK_ETFS.items():
        try:
            df = pykrx.get_market_ohlcv(start_date, end_date, etf_code)
            if df.empty:
                continue

            records = []
            for date_idx in df.index:
                row = df.loc[date_idx]
                records.append((
                    etf_code, date_idx.strftime("%Y-%m-%d"),
                    int(row["시가"]), int(row["고가"]),
                    int(row["저가"]), int(row["종가"]),
                    int(row["거래량"]), 0, 0,
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO daily_price
                (stock_code, trade_date, open, high, low, close, volume, trade_amount, market_cap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
            print(f"    {etf_name}: {len(df)}일 업데이트")
        except Exception as e:
            print(f"    {etf_name}: 실패 ({e})")

        time.sleep(0.5)


def update_master_market_cap(conn, trade_date):
    """stock_master 시총을 최신 거래일 기준으로 갱신"""
    trade_date_fmt = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    updated = conn.execute("""
        UPDATE stock_master SET market_cap = (
            SELECT dp.market_cap FROM daily_price dp
            WHERE dp.stock_code = stock_master.stock_code
              AND dp.trade_date = ?
              AND dp.market_cap > 0
        ), updated_at = datetime('now','localtime')
        WHERE EXISTS (
            SELECT 1 FROM daily_price dp
            WHERE dp.stock_code = stock_master.stock_code
              AND dp.trade_date = ?
              AND dp.market_cap > 0
        )
    """, (trade_date_fmt, trade_date_fmt)).rowcount
    conn.commit()
    print(f"\n  stock_master 시총 갱신: {updated}종목")


def show_summary():
    """업데이트 결과 요약"""
    conn = get_db()
    last_date = conn.execute("SELECT MAX(trade_date) FROM daily_price").fetchone()[0]
    stock_count = conn.execute("""
        SELECT COUNT(DISTINCT stock_code) FROM daily_price WHERE trade_date = ?
    """, (last_date,)).fetchone()[0]
    conn.close()

    print(f"\n  === 결과 ===")
    print(f"  최신 주가일: {last_date} ({stock_count}종목)")
    print(f"  DB 크기: {DB_PATH.stat().st_size / 1024 / 1024:.1f}MB")


if __name__ == "__main__":
    t0 = time.time()
    update_prices()
    show_summary()
    print(f"\n  소요 시간: {time.time() - t0:.0f}초")
