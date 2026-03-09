"""
SQLite → PostgreSQL 마이그레이션 스크립트.

사용법:
    python scripts/migrate_to_pg.py                    # 전체 마이그레이션
    python scripts/migrate_to_pg.py --tables daily_price fnspace_forward  # 특정 테이블만
    python scripts/migrate_to_pg.py --dry-run          # 실제 전송 없이 테이블 확인만
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DB_PATH

# Railway PostgreSQL URL
PG_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:NgHDMsgiGwbvMpWHLUgTQguaedbGoxvv@metro.proxy.rlwy.net:50087/railway",
)

# 마이그레이션 대상 테이블 (뉴스 제외)
TARGET_TABLES = [
    "stock_master",
    "daily_price",
    "fnspace_master",
    "fnspace_finance",
    "fnspace_forward",
    "fnspace_consensus_daily",
    "valuation_factors",
    "signals",
]

# SQLite → PostgreSQL 타입 매핑
TYPE_MAP = {
    "INTEGER": "BIGINT",
    "REAL": "DOUBLE PRECISION",
    "TEXT": "TEXT",
    "BLOB": "BYTEA",
    "NUMERIC": "NUMERIC",
    "": "TEXT",
}


def get_sqlite_conn():
    return sqlite3.connect(str(DB_PATH))


def get_pg_conn():
    import psycopg2
    return psycopg2.connect(PG_URL)


def get_sqlite_tables(conn):
    """SQLite에서 테이블 목록과 행 수."""
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    result = {}
    for (name,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
        result[name] = count
    return result


def get_create_table_sql(sqlite_conn, table_name):
    """SQLite 스키마 → PostgreSQL CREATE TABLE."""
    rows = sqlite_conn.execute(f"PRAGMA table_info([{table_name}])").fetchall()
    cols = []
    for row in rows:
        col_name = row[1]
        col_type = row[2].upper() if row[2] else "TEXT"

        pg_type = TYPE_MAP.get(col_type, "TEXT")
        # FLOAT, DOUBLE 등 처리
        if "FLOAT" in col_type or "DOUBLE" in col_type:
            pg_type = "DOUBLE PRECISION"
        elif "INT" in col_type:
            pg_type = "BIGINT"
        elif "CHAR" in col_type or "VAR" in col_type:
            pg_type = "TEXT"

        cols.append(f'  "{col_name}" {pg_type}')

    return f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n' + ",\n".join(cols) + "\n);"


def migrate_table(sqlite_conn, pg_conn, table_name, batch_size=5000):
    """단일 테이블 마이그레이션."""
    cur = pg_conn.cursor()

    # 기존 테이블 삭제 후 재생성
    cur.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
    create_sql = get_create_table_sql(sqlite_conn, table_name)
    cur.execute(create_sql)
    pg_conn.commit()

    # 데이터 전송
    total = sqlite_conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
    if total == 0:
        print(f"  ⚠️  {table_name}: 빈 테이블 (스키마만 생성)")
        return

    # 컬럼명 가져오기
    cols_info = sqlite_conn.execute(f"PRAGMA table_info([{table_name}])").fetchall()
    col_names = [c[1] for c in cols_info]
    col_str = ", ".join(f'"{c}"' for c in col_names)

    # COPY 방식으로 빠르게 전송
    import io
    import csv

    offset = 0
    with tqdm(total=total, desc=f"  {table_name}", unit="rows") as pbar:
        while offset < total:
            rows = sqlite_conn.execute(
                f"SELECT * FROM [{table_name}] LIMIT {batch_size} OFFSET {offset}"
            ).fetchall()
            if not rows:
                break

            # CSV 버퍼로 변환 후 COPY로 전송
            buf = io.StringIO()
            writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
            for row in rows:
                writer.writerow(["\\N" if v is None else v for v in row])
            buf.seek(0)
            cur.copy_from(buf, table_name, sep="\t", null="\\N", columns=col_names)
            pg_conn.commit()
            pbar.update(len(rows))
            offset += batch_size

    # 인덱스 생성
    _create_indexes(cur, table_name)
    pg_conn.commit()


def _create_indexes(cur, table_name):
    """테이블별 인덱스 생성."""
    indexes = {
        "daily_price": [
            ("idx_dp_stock_date", "stock_code, trade_date"),
            ("idx_dp_date", "trade_date"),
        ],
        "fnspace_finance": [
            ("idx_ff_stock", "stock_code"),
            ("idx_ff_year_q", "fiscal_year, fiscal_quarter"),
        ],
        "fnspace_forward": [
            ("idx_fwd_date", "trade_date"),
            ("idx_fwd_stock_date", "stock_code, trade_date"),
        ],
        "fnspace_consensus_daily": [
            ("idx_cons_date", "trade_date"),
            ("idx_cons_stock_date", "stock_code, trade_date"),
        ],
        "valuation_factors": [
            ("idx_vf_stock_date", "stock_code, calc_date"),
        ],
        "signals": [
            ("idx_sig_stock_date", "stock_code, calc_date"),
        ],
        "stock_master": [
            ("idx_sm_code", "stock_code"),
        ],
        "fnspace_master": [
            ("idx_fm_code", "stock_code"),
        ],
    }
    for idx_name, idx_cols in indexes.get(table_name, []):
        cur.execute(
            f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ({idx_cols})'
        )


def main():
    parser = argparse.ArgumentParser(description="SQLite → PostgreSQL 마이그레이션")
    parser.add_argument("--tables", nargs="+", help="특정 테이블만 마이그레이션")
    parser.add_argument("--dry-run", action="store_true", help="실제 전송 없이 확인만")
    args = parser.parse_args()

    print(f"📦 SQLite DB: {DB_PATH}")
    print(f"🐘 PostgreSQL: {PG_URL[:50]}...")
    print()

    sqlite_conn = get_sqlite_conn()
    all_tables = get_sqlite_tables(sqlite_conn)

    tables = args.tables or [t for t in TARGET_TABLES if t in all_tables]

    print("📋 마이그레이션 대상:")
    total_rows = 0
    for t in tables:
        count = all_tables.get(t, 0)
        total_rows += count
        print(f"  {t}: {count:,} rows")
    print(f"  총 {total_rows:,} rows")
    print()

    if args.dry_run:
        print("🔍 dry-run 모드 — 실제 전송 없음")
        sqlite_conn.close()
        return

    pg_conn = get_pg_conn()

    for table in tables:
        if table not in all_tables:
            print(f"  ⚠️  {table}: SQLite에 없음, 건너뜀")
            continue
        migrate_table(sqlite_conn, pg_conn, table)

    sqlite_conn.close()
    pg_conn.close()
    print()
    print("✅ 마이그레이션 완료!")


if __name__ == "__main__":
    main()
