"""
Database abstraction layer: SQLite (local) / PostgreSQL (Railway) 자동 감지.

사용법:
    from lib.db import get_conn, read_sql

    conn = get_conn()  # 환경에 맞는 커넥션
    df = read_sql("SELECT * FROM t WHERE col = ?", conn, params=(val,))
    row = conn.execute("SELECT ...", (val,)).fetchone()
"""
import os
import re
import sqlite3

from config.settings import DB_PATH

DATABASE_URL = os.environ.get("DATABASE_URL", "")

_is_pg = DATABASE_URL.startswith("postgresql")


def _translate_sql(sql: str) -> str:
    """SQLite → PostgreSQL SQL 변환.

    1. date(?, '-N days') → (CAST(%s AS date) + INTERVAL '-N days')
    2. ? placeholder → %s
    3. INSERT OR REPLACE → INSERT ... ON CONFLICT DO UPDATE (미지원, 별도 처리 필요)
    """
    if not _is_pg:
        return sql

    # date(?, '-N days') → (CAST(%s AS date) + INTERVAL '-N days')
    sql = re.sub(
        r"date\(\s*\?\s*,\s*'(-\d+)\s+days?'\s*\)",
        lambda m: f"CAST((CAST(%s AS date) + INTERVAL '{m.group(1)} days') AS text)",
        sql,
    )
    # ? → %s (남은 것들)
    sql = sql.replace("?", "%s")
    return sql


class _PgCursorWrapper:
    """psycopg2 cursor를 sqlite3 cursor처럼 감싸는 래퍼."""

    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, sql, params=None):
        sql = _translate_sql(sql)
        self._cursor.execute(sql, params or ())
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def fetchmany(self, size=None):
        return self._cursor.fetchmany(size)

    @property
    def description(self):
        return self._cursor.description

    def close(self):
        self._cursor.close()

    def __iter__(self):
        return iter(self._cursor)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _PgConnWrapper:
    """psycopg2 connection을 sqlite3 인터페이스처럼 감싸는 래퍼.

    - execute(), cursor() 에서 SQL 자동 변환
    - pd.read_sql_query 호환 (cursor()를 통해 변환된 커서 반환)
    """

    def __init__(self, pg_conn):
        self._conn = pg_conn

    def execute(self, sql, params=None):
        sql = _translate_sql(sql)
        cur = self._conn.cursor()
        cur.execute(sql, params or ())
        return cur

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def cursor(self):
        return _PgCursorWrapper(self._conn.cursor())

    # pd.read_sql_query 호환을 위한 속성 위임
    def __getattr__(self, name):
        return getattr(self._conn, name)


def get_conn():
    """환경에 맞는 DB 커넥션 반환."""
    if _is_pg:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=30)
        cur = conn.cursor()
        cur.execute("SET search_path TO alpha_lab, public")
        conn.commit()
        cur.close()
        return _PgConnWrapper(conn)
    else:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def read_sql(sql: str, conn, params=None):
    """pd.read_sql_query의 래퍼. PostgreSQL일 때 SQL/placeholder 자동 변환."""
    import warnings
    import pandas as pd
    if _is_pg:
        sql = _translate_sql(sql)
        raw_conn = conn._conn if isinstance(conn, _PgConnWrapper) else conn
        cur = raw_conn.cursor()
        cur.execute(sql, params or ())
        cols = [desc[0] for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    else:
        return pd.read_sql_query(sql, conn, params=params)
