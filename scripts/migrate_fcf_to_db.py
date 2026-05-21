"""
fnspace_finance 테이블에 fcf 컬럼 추가 및 캐시 데이터 업로드

실행: python scripts/migrate_fcf_to_db.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.db import get_conn

FCF_CACHE_PATH = Path(__file__).parent.parent / "analysis" / "data" / "fcf_annual_cache.json"


def main():
    if not FCF_CACHE_PATH.exists():
        print(f"캐시 파일 없음: {FCF_CACHE_PATH}")
        return

    with open(FCF_CACHE_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    # raw: {"000020:2021": 값(원 단위), ...}
    # DB는 천원 단위 → /1000
    fcf_map = {}
    for k, v in raw.items():
        code, yr = k.split(":")
        fcf_map[(f"A{code}", int(yr))] = v / 1000.0

    conn = get_conn()
    cur = conn._conn.cursor()

    # 컬럼 추가 (이미 있으면 무시)
    try:
        cur.execute("ALTER TABLE fnspace_finance ADD COLUMN fcf FLOAT")
        conn._conn.commit()
        print("[DB] fcf 컬럼 추가 완료")
    except Exception:
        conn._conn.rollback()
        print("[DB] fcf 컬럼 이미 존재 (skip)")

    # 데이터 업로드
    updated = 0
    for (code, yr), val in fcf_map.items():
        cur.execute("""
            UPDATE fnspace_finance
            SET fcf = %s
            WHERE stock_code = %s AND fiscal_year = %s AND fiscal_quarter = 'Annual'
        """, (val, code, yr))
        updated += cur.rowcount

    conn._conn.commit()
    print(f"[DB] fcf 업데이트 완료: {updated}건")

    # 검증
    cur.execute("SELECT COUNT(*) FROM fnspace_finance WHERE fcf IS NOT NULL")
    cnt = cur.fetchone()[0]
    print(f"[DB] fcf 값 있는 행: {cnt}건")
    conn.close()


if __name__ == "__main__":
    main()
