"""
scripts/build_stock_indicators.py

종목별 일별 기술적 지표(MA, MFI 등)를 미리 계산해서 alpha_lab.stock_indicators
테이블에 적재. backend가 매 calc_date 호출마다 rolling 계산을 반복하는 대신
DB에서 SELECT 만으로 가져와 사용 → 백테스트 시간 큰 폭 단축.

적재 항목:
- ma_120: 120일 이동평균
- deviation_120: (close - ma_120) / ma_120 * 100
- mfi_val: 14일 Money Flow Index
- pos_sum_14, neg_sum_14: MFI 계산용 중간값 (재계산 가능하지만 저장하면 빠름)

OBV는 cumsum 시작점에 민감하므로 일단 캐시 제외 (기존 함수 사용).

실행:
  python scripts/build_stock_indicators.py             # 전체 재계산 (~5~10분)
  python scripts/build_stock_indicators.py --since 2025-01-01  # 그 이후만
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
import pandas as pd

from lib.db import get_conn, read_sql


SCHEMA = """
CREATE TABLE IF NOT EXISTS alpha_lab.stock_indicators (
    stock_code      TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    ma_120          REAL,
    deviation_120   REAL,
    mfi_val         REAL,
    pos_sum_14      REAL,
    neg_sum_14      REAL,
    updated_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (stock_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_stock_indicators_date
    ON alpha_lab.stock_indicators(trade_date);
"""


def ensure_schema(conn):
    raw = conn._conn.cursor() if hasattr(conn, "_conn") else conn.cursor()
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip():
            raw.execute(stmt)
    conn.commit()
    print("✓ schema ready")


def fetch_price(conn, since: str | None = None) -> pd.DataFrame:
    """daily_price 전체(또는 since 이후) 데이터를 읽음.
    Returns: stock_code(with 'A' prefix), trade_date, close, high, low, volume
    """
    raw = conn._conn.cursor()
    if since:
        raw.execute("""
            SELECT 'A' || stock_code AS stock_code, trade_date,
                   close, high, low, volume
            FROM alpha_lab.daily_price
            WHERE trade_date >= %s
            ORDER BY stock_code, trade_date
        """, (since,))
    else:
        raw.execute("""
            SELECT 'A' || stock_code AS stock_code, trade_date,
                   close, high, low, volume
            FROM alpha_lab.daily_price
            ORDER BY stock_code, trade_date
        """)
    rows = raw.fetchall()
    cols = [d[0] for d in raw.description]
    return pd.DataFrame(rows, columns=cols)


def compute_indicators(price: pd.DataFrame, ma_window: int = 120, mfi_period: int = 14) -> pd.DataFrame:
    """daily_price → ma_120, deviation_120, mfi_val, pos_sum_14, neg_sum_14"""
    df = price.sort_values(["stock_code", "trade_date"]).copy()

    # MA
    df["ma_120"] = df.groupby("stock_code")["close"].transform(
        lambda x: x.rolling(ma_window, min_periods=ma_window).mean()
    )
    df["deviation_120"] = (df["close"] - df["ma_120"]) / df["ma_120"] * 100

    # MFI 중간값
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["mf"] = df["tp"] * df["volume"]
    tp_diff = df.groupby("stock_code")["tp"].transform(lambda x: x.diff())
    df["pos_mf"] = np.where(tp_diff > 0, df["mf"], 0.0)
    df["neg_mf"] = np.where(tp_diff < 0, df["mf"], 0.0)
    df["pos_sum_14"] = df.groupby("stock_code")["pos_mf"].transform(
        lambda x: x.rolling(mfi_period, min_periods=mfi_period).sum()
    )
    df["neg_sum_14"] = df.groupby("stock_code")["neg_mf"].transform(
        lambda x: x.rolling(mfi_period, min_periods=mfi_period).sum()
    )
    mfr = df["pos_sum_14"] / df["neg_sum_14"].replace(0, np.nan)
    df["mfi_val"] = 100 - (100 / (1 + mfr))

    return df[["stock_code", "trade_date", "ma_120", "deviation_120",
               "mfi_val", "pos_sum_14", "neg_sum_14"]]


def upsert(conn, df: pd.DataFrame, batch_size: int = 50000):
    """UPSERT (INSERT ... ON CONFLICT DO UPDATE) using execute_values for speed.

    psycopg2.extras.execute_values 는 한 쿼리에 batch_size 행을 다중 VALUES 로
    묶어서 보냄. executemany 보다 10~50배 빠름. 외부 proxy 통해도 합리적 속도.
    """
    from psycopg2.extras import execute_values
    raw = conn._conn.cursor()
    cols = ["stock_code", "trade_date", "ma_120", "deviation_120",
            "mfi_val", "pos_sum_14", "neg_sum_14"]
    update_cols = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c not in ("stock_code", "trade_date"))
    sql = f"""
        INSERT INTO alpha_lab.stock_indicators ({", ".join(cols)})
        VALUES %s
        ON CONFLICT (stock_code, trade_date)
        DO UPDATE SET {update_cols}, updated_at = NOW()
    """
    total = len(df)
    t_start = time.time()
    for i in range(0, total, batch_size):
        chunk = df.iloc[i:i+batch_size]
        rows = [
            tuple(None if pd.isna(v) else v for v in row)
            for row in chunk[cols].itertuples(index=False, name=None)
        ]
        execute_values(raw, sql, rows, page_size=batch_size)
        conn.commit()
        elapsed = time.time() - t_start
        done = min(i + batch_size, total)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  ... upserted {done:,}/{total:,}  ({elapsed:.0f}s, {rate:.0f} rows/s, ETA {eta:.0f}s)", flush=True)
    print(f"✓ upserted {total:,} rows in {time.time()-t_start:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=str, default=None,
                        help="이 날짜 이후 데이터만 재계산 (incremental). 미지정시 전체.")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print(f"  Stock Indicators 적재 — since={args.since or '전체'}")
    print("=" * 60)

    conn = get_conn()
    ensure_schema(conn)

    print("\n[1/3] daily_price 읽는 중...")
    price = fetch_price(conn, since=args.since)
    print(f"  → {len(price):,} rows, {price['stock_code'].nunique()} 종목")

    print("\n[2/3] indicator 계산 중...")
    t1 = time.time()
    ind = compute_indicators(price)
    print(f"  → {len(ind):,} rows ({time.time()-t1:.1f}s)")

    print("\n[3/3] DB UPSERT 중...")
    upsert(conn, ind)

    conn.close()
    print(f"\n완료 ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
