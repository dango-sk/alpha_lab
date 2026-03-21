"""
daily_price SQLite → PostgreSQL 마이그레이션
실행: python scripts/migrate_daily_price_to_pg.py
"""
import sqlite3
import psycopg2
from psycopg2.extras import execute_values

PG_URL = "postgresql://postgres:NgHDMsgiGwbvMpWHLUgTQguaedbGoxvv@metro.proxy.rlwy.net:50087/railway"
SQLITE_DB = "data/alpha_lab.db"
BATCH_SIZE = 10000

sq = sqlite3.connect(SQLITE_DB)
pg = psycopg2.connect(PG_URL)
cur = pg.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS alpha_lab.daily_price (
        stock_code TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        trade_amount REAL,
        market_cap REAL,
        PRIMARY KEY (stock_code, trade_date)
    )
""")
pg.commit()

total = sq.execute("SELECT COUNT(*) FROM daily_price").fetchone()[0]
print(f"총 {total:,}건 마이그레이션 시작")

offset = 0
inserted = 0
while offset < total:
    rows = sq.execute(f"""
        SELECT stock_code, trade_date, open, high, low, close, volume, trade_amount, market_cap
        FROM daily_price
        LIMIT {BATCH_SIZE} OFFSET {offset}
    """).fetchall()

    if not rows:
        break

    execute_values(cur, """
        INSERT INTO alpha_lab.daily_price (stock_code, trade_date, open, high, low, close, volume, trade_amount, market_cap)
        VALUES %s
        ON CONFLICT (stock_code, trade_date) DO NOTHING
    """, rows)
    pg.commit()

    inserted += len(rows)
    offset += BATCH_SIZE
    if inserted % 100000 == 0 or offset >= total:
        print(f"  [{inserted:,}/{total:,}] ({inserted/total*100:.1f}%)")

pg.close()
sq.close()
print(f"\n=== 완료: {inserted:,}건 ===")
