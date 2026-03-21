"""
Step 5: fnspace_finance 보충 수집
실행: python scripts/step5_fill_finance.py

- fnspace_master에 있지만 fnspace_finance에 없는 종목만 대상
- FnSpace FinanceApi로 연간(Annual) + 분기(QQ) 재무 수집 (2017~현재)
- 기존 데이터는 건드리지 않음

소요 시간: 약 10~15분
"""
import sqlite3
import sys
import os
import time
import requests
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

LAB_DB = Path(__file__).parent.parent / "data" / "alpha_lab.db"

# .env 로드
for env_path in [
    Path(__file__).parent.parent / ".env",
    Path.home() / "Downloads" / "alpha_radar" / ".env",
]:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"'))

FNSPACE_API_KEY = os.environ.get("FNSPACE_API_KEY", "")
BASE_URL = "https://www.fnspace.com/Api"
API_DELAY = 0.5
MAX_CODES_PER_CALL = 10
MAX_ITEMS_PER_CALL = 10

# FinanceApi item 코드
FINANCE_ITEMS = {
    "M382500": "pbr",
    "M211500": "roe",
    "M211700": "roic",
    "M213000": "ev",
    "M124000": "ic",
    "M331010": "ev_ebit",
    "M123100": "ebit",
    "M113900": "net_debt",
    "M113800": "interest_debt",
    "M115000": "total_equity",
    "M312000": "eps",
    "M314000": "bps",
    "M382100": "per",
    "M383300": "psr",
    "M331030": "ev_ebitda",
    "M123200": "ebitda",
    "M121000": "revenue",
    "M121500": "operating_income",
    "M122700": "net_income",
    "M211000": "oi_margin",
    "M431800": "div_yield",
    "M385110": "pcf",
}


def api_call(endpoint, params):
    url = f"{BASE_URL}/{endpoint}"
    params["key"] = FNSPACE_API_KEY
    params["format"] = "json"
    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        if data.get("success") == "true":
            return data
        return None
    except Exception as e:
        return None


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def collect_finance():
    conn = sqlite3.connect(str(LAB_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    # fnspace_master 전체 종목 대상 (기존 종목도 빠진 연도 보충)
    all_stocks = conn.execute("""
        SELECT DISTINCT m.stock_code
        FROM fnspace_master m
        ORDER BY m.stock_code
    """).fetchall()
    all_codes = [r[0] for r in all_stocks]

    # 이미 완전히 수집된 종목 (2017~현재 전부 있는 경우) 스킵
    existing = conn.execute("""
        SELECT stock_code, COUNT(DISTINCT fiscal_year) as year_cnt
        FROM fnspace_finance
        GROUP BY stock_code
    """).fetchall()
    existing_map = {r[0]: r[1] for r in existing}

    target_years = int(datetime.now().strftime("%Y")) - 2017 + 1
    target_codes = [c for c in all_codes if existing_map.get(c, 0) < target_years]
    print(f"전체 종목: {len(all_codes)}개, 보충 대상: {len(target_codes)}개 (연도 누락 포함)")

    if not target_codes:
        print("보충할 종목 없음")
        conn.close()
        return

    from_year = "2017"
    to_year = str(datetime.now().year)

    # item 코드를 10개씩 분할
    item_keys = list(FINANCE_ITEMS.keys())
    item_groups = list(chunk_list(item_keys, MAX_ITEMS_PER_CALL))

    total_saved = 0
    api_calls = 0
    errors = 0

    code_chunks = list(chunk_list(target_codes, MAX_CODES_PER_CALL))

    for chunk_idx, code_chunk in enumerate(code_chunks):
        codes_str = ",".join(code_chunk)

        for item_group in item_groups:
            items_str = ",".join(item_group)

            data = api_call("FinanceApi", {
                "code": codes_str,
                "item": items_str,
                "consolgb": "M",
                "annualgb": "A",
                "fraccyear": from_year,
                "toaccyear": to_year,
            })
            api_calls += 1

            if data and data.get("dataset"):
                for stock_data in data["dataset"]:
                    code = stock_data.get("CODE", "")
                    for row in stock_data.get("DATA", []):
                        fy = row.get("FS_YEAR")
                        fq = row.get("FS_QTR", "Annual")
                        if not fy:
                            continue

                        values = {}
                        for item_cd in item_group:
                            col_name = FINANCE_ITEMS[item_cd]
                            val = row.get(item_cd)
                            if val is not None:
                                values[col_name] = val

                        if values:
                            cols = list(values.keys())
                            set_clause = ", ".join(f"{c} = ?" for c in cols)
                            conn.execute(f"""
                                INSERT INTO fnspace_finance (stock_code, fiscal_year, fiscal_quarter, {', '.join(cols)}, updated_at)
                                VALUES (?, ?, ?, {', '.join('?' for _ in cols)}, datetime('now','localtime'))
                                ON CONFLICT(stock_code, fiscal_year, fiscal_quarter) DO UPDATE SET
                                {set_clause}, updated_at = datetime('now','localtime')
                            """, (code, fy, fq, *values.values(), *values.values()))
                            total_saved += 1
            else:
                errors += 1

            time.sleep(API_DELAY)

        if (chunk_idx + 1) % 10 == 0 or chunk_idx == len(code_chunks) - 1:
            conn.commit()
            print(f"  [{(chunk_idx+1)*MAX_CODES_PER_CALL}/{len(target_codes)}] 저장: {total_saved}건, API: {api_calls}건, 에러: {errors}건")

    conn.commit()
    conn.close()
    print(f"\n=== 완료: {total_saved}건 저장, API {api_calls}건 호출, 에러 {errors}건 ===")


def collect_finance_quarterly():
    """분기(QQ) 재무 데이터 수집 — TTM 계산용."""
    conn = sqlite3.connect(str(LAB_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    all_stocks = conn.execute("""
        SELECT DISTINCT stock_code FROM fnspace_master ORDER BY stock_code
    """).fetchall()
    all_codes = [r[0] for r in all_stocks]

    # 이미 충분한 분기 데이터가 있는 종목 스킵
    # 정상: 2017~직전년도 × 4분기 (2025년이면 9년 × 4 = 36개)
    existing = conn.execute("""
        SELECT stock_code, COUNT(*) as cnt
        FROM fnspace_finance
        WHERE fiscal_quarter IN ('1Q', '2Q', '3Q', '4Q')
        GROUP BY stock_code
    """).fetchall()
    existing_map = {r[0]: r[1] for r in existing}

    complete_years = int(datetime.now().strftime("%Y")) - 2017
    expected_qtrs = complete_years * 4  # 올해 제외 (아직 미확정)
    target_codes = [c for c in all_codes if existing_map.get(c, 0) < expected_qtrs]
    print(f"분기 수집 대상: {len(target_codes)}개 (전체 {len(all_codes)}개)")

    if not target_codes:
        print("분기 보충할 종목 없음")
        conn.close()
        return

    from_year = "2017"
    to_year = str(datetime.now().year)

    item_keys = list(FINANCE_ITEMS.keys())
    item_groups = list(chunk_list(item_keys, MAX_ITEMS_PER_CALL))

    total_saved = 0
    api_calls = 0
    errors = 0

    code_chunks = list(chunk_list(target_codes, MAX_CODES_PER_CALL))

    for chunk_idx, code_chunk in enumerate(code_chunks):
        codes_str = ",".join(code_chunk)

        for item_group in item_groups:
            items_str = ",".join(item_group)

            data = api_call("FinanceApi", {
                "code": codes_str,
                "item": items_str,
                "consolgb": "M",
                "annualgb": "QQ",
                "accdategb": "C",
                "fraccyear": from_year,
                "toaccyear": to_year,
            })
            api_calls += 1

            if data and data.get("dataset"):
                for stock_data in data["dataset"]:
                    code = stock_data.get("CODE", "")
                    for row in stock_data.get("DATA", []):
                        fy = row.get("FS_YEAR")
                        fq = row.get("FS_QTR", "")
                        if not fy or fq not in ("1Q", "2Q", "3Q", "4Q"):
                            continue

                        values = {}
                        for item_cd in item_group:
                            col_name = FINANCE_ITEMS[item_cd]
                            val = row.get(item_cd)
                            if val is not None:
                                values[col_name] = val

                        if values:
                            cols = list(values.keys())
                            set_clause = ", ".join(f"{c} = ?" for c in cols)
                            conn.execute(f"""
                                INSERT INTO fnspace_finance (stock_code, fiscal_year, fiscal_quarter, {', '.join(cols)}, updated_at)
                                VALUES (?, ?, ?, {', '.join('?' for _ in cols)}, datetime('now','localtime'))
                                ON CONFLICT(stock_code, fiscal_year, fiscal_quarter) DO UPDATE SET
                                {set_clause}, updated_at = datetime('now','localtime')
                            """, (code, fy, fq, *values.values(), *values.values()))
                            total_saved += 1
            else:
                errors += 1

            time.sleep(API_DELAY)

        if (chunk_idx + 1) % 10 == 0 or chunk_idx == len(code_chunks) - 1:
            conn.commit()
            print(f"  [분기 {(chunk_idx+1)*MAX_CODES_PER_CALL}/{len(target_codes)}] 저장: {total_saved}건, API: {api_calls}건, 에러: {errors}건")

    conn.commit()
    conn.close()
    print(f"\n=== 분기 완료: {total_saved}건 저장, API {api_calls}건 호출, 에러 {errors}건 ===")


def upload_to_pg():
    """fnspace_finance를 Railway PostgreSQL에 업로드"""
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        print("[SKIP] psycopg2 없음, PostgreSQL 업로드 스킵")
        return

    PG_URL = "postgresql://postgres:NgHDMsgiGwbvMpWHLUgTQguaedbGoxvv@metro.proxy.rlwy.net:50087/railway"

    conn = sqlite3.connect(str(LAB_DB))
    pg = psycopg2.connect(PG_URL)
    cur = pg.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS alpha_lab.fnspace_finance (
            stock_code TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            fiscal_quarter TEXT NOT NULL,
            pbr REAL, roe REAL, roic REAL, ev REAL, ic REAL,
            ev_ebit REAL, ebit REAL, net_debt REAL, interest_debt REAL,
            total_equity REAL, eps REAL, bps REAL, per REAL, psr REAL,
            ev_ebitda REAL, ebitda REAL, revenue REAL, operating_income REAL,
            net_income REAL, oi_margin REAL, div_yield REAL, pcf REAL,
            updated_at TEXT,
            PRIMARY KEY (stock_code, fiscal_year, fiscal_quarter)
        )
    """)
    pg.commit()

    rows = conn.execute("""
        SELECT stock_code, fiscal_year, fiscal_quarter,
               pbr, roe, roic, ev, ic, ev_ebit, ebit, net_debt,
               interest_debt, total_equity, eps, bps, per, psr,
               ev_ebitda, ebitda, revenue, operating_income, net_income,
               oi_margin, div_yield, pcf, updated_at
        FROM fnspace_finance
    """).fetchall()

    if rows:
        execute_values(cur, """
            INSERT INTO alpha_lab.fnspace_finance (
                stock_code, fiscal_year, fiscal_quarter,
                pbr, roe, roic, ev, ic, ev_ebit, ebit, net_debt,
                interest_debt, total_equity, eps, bps, per, psr,
                ev_ebitda, ebitda, revenue, operating_income, net_income,
                oi_margin, div_yield, pcf, updated_at
            ) VALUES %s
            ON CONFLICT (stock_code, fiscal_year, fiscal_quarter) DO UPDATE SET
                pbr=EXCLUDED.pbr, roe=EXCLUDED.roe, roic=EXCLUDED.roic,
                ev=EXCLUDED.ev, ic=EXCLUDED.ic, ev_ebit=EXCLUDED.ev_ebit,
                ebit=EXCLUDED.ebit, net_debt=EXCLUDED.net_debt,
                interest_debt=EXCLUDED.interest_debt, total_equity=EXCLUDED.total_equity,
                eps=EXCLUDED.eps, bps=EXCLUDED.bps, per=EXCLUDED.per, psr=EXCLUDED.psr,
                ev_ebitda=EXCLUDED.ev_ebitda, ebitda=EXCLUDED.ebitda,
                revenue=EXCLUDED.revenue, operating_income=EXCLUDED.operating_income,
                net_income=EXCLUDED.net_income, oi_margin=EXCLUDED.oi_margin,
                div_yield=EXCLUDED.div_yield, pcf=EXCLUDED.pcf,
                updated_at=EXCLUDED.updated_at
        """, rows)
        pg.commit()

    print(f"PostgreSQL 업로드: {len(rows):,}건")
    pg.close()
    conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarterly-only", action="store_true", help="분기 데이터만 수집 (Annual 스킵)")
    args = parser.parse_args()

    if not FNSPACE_API_KEY:
        print("[ERROR] FNSPACE_API_KEY가 설정되지 않았습니다")
        return

    if not args.quarterly_only:
        print(f"=== Step 5: fnspace_finance 연간 보충 수집 ({datetime.now():%Y-%m-%d %H:%M}) ===")
        collect_finance()
        print()

    print(f"=== Step 5: 분기(QQ) 재무 수집 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    collect_finance_quarterly()
    upload_to_pg()


if __name__ == "__main__":
    main()
