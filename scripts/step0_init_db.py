"""
Step 0: DB 초기화 — 스키마 추가 + corpCode.xml + alpha_radar 마이그레이션
실행: python scripts/step0_init_db.py
"""
import sqlite3
import sys
import os
import zipfile
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

import requests

sys.path.append(str(Path(__file__).parent.parent))

# alpha_lab 전용 DB (settings.py의 DB_PATH는 아직 alpha_radar를 가리키므로 직접 지정)
LAB_DB = Path(__file__).parent.parent / "data" / "alpha_lab.db"
RADAR_DB = Path.home() / "Downloads" / "alpha_radar" / "db" / "alpha_radar.db"
DART_API_KEY = "645227137bcce54961d01d3dccc61908fb607953"


def add_tables(conn: sqlite3.Connection):
    """corp_code_map, shares_outstanding 테이블 추가"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS corp_code_map (
            stock_code TEXT PRIMARY KEY,   -- 6자리 (005930)
            corp_code  TEXT NOT NULL,       -- DART 고유번호 (00126380)
            corp_name  TEXT,
            updated_at TEXT DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS shares_outstanding (
            stock_code     TEXT NOT NULL,   -- 6자리
            disclosure_date TEXT NOT NULL,  -- 공시일 (YYYYMMDD)
            fiscal_date    TEXT NOT NULL,   -- 결산기준일
            shares_common  BIGINT,         -- 보통주 발행주식수
            shares_pref    BIGINT,         -- 우선주 발행주식수
            report_type    TEXT,           -- 11011=사업보고서, 11012=반기, 11013=1분기, 11014=3분기
            updated_at     TEXT DEFAULT (datetime('now','localtime')),
            PRIMARY KEY (stock_code, disclosure_date, report_type)
        );
        CREATE INDEX IF NOT EXISTS idx_shares_code ON shares_outstanding(stock_code);
        CREATE INDEX IF NOT EXISTS idx_shares_disclosure ON shares_outstanding(disclosure_date);
    """)
    print("[OK] corp_code_map, shares_outstanding 테이블 생성 완료")


def download_corp_code(conn: sqlite3.Connection):
    """DART corpCode.xml 다운로드 → corp_code_map 저장"""
    existing = conn.execute("SELECT COUNT(*) FROM corp_code_map").fetchone()[0]
    if existing > 0:
        print(f"[SKIP] corp_code_map 이미 {existing}건 존재")
        return

    print("[...] DART corpCode.xml 다운로드 중...")
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    resp = requests.get(url, params={"crtfc_key": DART_API_KEY}, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_data = zf.read(zf.namelist()[0])

    root = ET.fromstring(xml_data)
    rows = []
    for item in root.findall("list"):
        stock_code = item.findtext("stock_code", "").strip()
        corp_code = item.findtext("corp_code", "").strip()
        corp_name = item.findtext("corp_name", "").strip()
        if stock_code and len(stock_code) == 6:
            rows.append((stock_code, corp_code, corp_name))

    conn.executemany(
        "INSERT OR IGNORE INTO corp_code_map (stock_code, corp_code, corp_name) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    print(f"[OK] corp_code_map: {len(rows)}건 저장")


def migrate_from_radar(conn: sqlite3.Connection):
    """alpha_radar DB에서 재무 데이터 마이그레이션"""
    if not RADAR_DB.exists():
        print(f"[SKIP] alpha_radar DB 없음: {RADAR_DB}")
        return

    radar = sqlite3.connect(str(RADAR_DB))

    # 1) daily_price (market_cap 포함)
    cnt = conn.execute("SELECT COUNT(*) FROM daily_price").fetchone()[0]
    if cnt == 0:
        print("[...] daily_price 마이그레이션 중...")
        df_count = radar.execute("SELECT COUNT(*) FROM daily_price").fetchone()[0]
        batch = 100_000
        offset = 0
        total = 0
        while offset < df_count:
            rows = radar.execute(
                f"SELECT stock_code, trade_date, open, high, low, close, volume, trade_amount, market_cap "
                f"FROM daily_price LIMIT {batch} OFFSET {offset}"
            ).fetchall()
            if not rows:
                break
            conn.executemany(
                "INSERT OR IGNORE INTO daily_price VALUES (?,?,?,?,?,?,?,?,?)", rows
            )
            total += len(rows)
            offset += batch
        conn.commit()
        print(f"[OK] daily_price: {total}건 마이그레이션")
    else:
        print(f"[SKIP] daily_price 이미 {cnt}건 존재")

    # 2) fnspace_finance
    cnt = conn.execute("SELECT COUNT(*) FROM fnspace_finance").fetchone()[0]
    if cnt == 0:
        print("[...] fnspace_finance 마이그레이션 중...")
        # alpha_lab schema 확인
        lab_cols = [r[1] for r in conn.execute("PRAGMA table_info(fnspace_finance)").fetchall()]
        radar_cols = [r[1] for r in radar.execute("PRAGMA table_info(fnspace_finance)").fetchall()]
        common = [c for c in radar_cols if c in lab_cols]
        cols_str = ", ".join(common)
        placeholders = ", ".join(["?"] * len(common))

        rows = radar.execute(f"SELECT {cols_str} FROM fnspace_finance").fetchall()
        conn.executemany(
            f"INSERT OR IGNORE INTO fnspace_finance ({cols_str}) VALUES ({placeholders})", rows
        )
        conn.commit()
        print(f"[OK] fnspace_finance: {len(rows)}건 마이그레이션")
    else:
        print(f"[SKIP] fnspace_finance 이미 {cnt}건 존재")

    # 3) fnspace_forward
    cnt = conn.execute("SELECT COUNT(*) FROM fnspace_forward").fetchone()[0]
    if cnt == 0:
        print("[...] fnspace_forward 마이그레이션 중...")
        radar_cols = [r[1] for r in radar.execute("PRAGMA table_info(fnspace_forward)").fetchall()]
        lab_cols = [r[1] for r in conn.execute("PRAGMA table_info(fnspace_forward)").fetchall()]
        common = [c for c in radar_cols if c in lab_cols]
        cols_str = ", ".join(common)
        placeholders = ", ".join(["?"] * len(common))

        rows = radar.execute(f"SELECT {cols_str} FROM fnspace_forward").fetchall()
        conn.executemany(
            f"INSERT OR IGNORE INTO fnspace_forward ({cols_str}) VALUES ({placeholders})", rows
        )
        conn.commit()
        print(f"[OK] fnspace_forward: {len(rows)}건 마이그레이션")
    else:
        print(f"[SKIP] fnspace_forward 이미 {cnt}건 존재")

    # 4) fnspace_consensus_daily
    cnt = conn.execute("SELECT COUNT(*) FROM fnspace_consensus_daily").fetchone()[0]
    if cnt == 0:
        print("[...] fnspace_consensus_daily 마이그레이션 중...")
        radar_cols = [r[1] for r in radar.execute("PRAGMA table_info(fnspace_consensus_daily)").fetchall()]
        lab_cols = [r[1] for r in conn.execute("PRAGMA table_info(fnspace_consensus_daily)").fetchall()]
        common = [c for c in radar_cols if c in lab_cols]
        cols_str = ", ".join(common)
        placeholders = ", ".join(["?"] * len(common))

        rows = radar.execute(f"SELECT {cols_str} FROM fnspace_consensus_daily").fetchall()
        conn.executemany(
            f"INSERT OR IGNORE INTO fnspace_consensus_daily ({cols_str}) VALUES ({placeholders})", rows
        )
        conn.commit()
        print(f"[OK] fnspace_consensus_daily: {len(rows)}건 마이그레이션")
    else:
        print(f"[SKIP] fnspace_consensus_daily 이미 {cnt}건 존재")

    radar.close()


def main():
    print(f"=== Step 0: DB 초기화 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    print(f"DB: {LAB_DB}")

    conn = sqlite3.connect(str(LAB_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    add_tables(conn)
    download_corp_code(conn)
    migrate_from_radar(conn)

    conn.close()
    print("\n=== Step 0 완료 ===")


if __name__ == "__main__":
    main()
