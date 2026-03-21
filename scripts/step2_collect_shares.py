"""
Step 2: DART 발행주식수 수집 (2017~2025)
실행: python scripts/step2_collect_shares.py

- corp_code_map에 있는 종목 대상
- 사업보고서(11011) + 반기(11012) + 1분기(11013) + 3분기(11014)
- 공시일(disclosure_date) 저장 → look-ahead bias 방지
- DART API: 분당 1,000건 제한 → 0.1초 딜레이

소요 시간: 종목 3,949 × 연도 9 × 보고서 4 = ~142,000건 → 약 4시간
"""
import sqlite3
import sys
import time
import requests
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

LAB_DB = Path(__file__).parent.parent / "data" / "alpha_lab.db"
DART_API_KEY = "645227137bcce54961d01d3dccc61908fb607953"
DART_URL = "https://opendart.fss.or.kr/api/stockTotqySttus.json"

# 보고서 유형
REPORT_CODES = {
    "11013": "1분기",
    "11012": "반기",
    "11014": "3분기",
    "11011": "사업보고서",
}

START_YEAR = 2017
END_YEAR = 2026
API_DELAY = 0.1  # 초 (분당 1,000건 제한 대응)


def parse_number(s: str) -> int:
    """쉼표 포함 숫자 문자열 → int, 실패 시 0"""
    if not s or s == "-":
        return 0
    try:
        return int(s.replace(",", ""))
    except ValueError:
        return 0


def collect_shares():
    conn = sqlite3.connect(str(LAB_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    # 이미 수집된 (stock_code, bsns_year, report_type) 조합 확인
    existing = set()
    for row in conn.execute(
        "SELECT stock_code, substr(fiscal_date,1,4), report_type FROM shares_outstanding"
    ).fetchall():
        existing.add((row[0], row[1], row[2]))

    # fnspace_master에 있는 종목만 (유니버스에 한 번이라도 포함된 종목)
    # stock_code: fnspace_master는 A005930 형식, corp_code_map은 005930 형식
    stocks = conn.execute("""
        SELECT DISTINCT c.stock_code, c.corp_code, c.corp_name
        FROM corp_code_map c
        INNER JOIN (
            SELECT DISTINCT SUBSTR(stock_code, 2) AS code6
            FROM fnspace_master
        ) m ON c.stock_code = m.code6
    """).fetchall()

    print(f"대상 종목: {len(stocks)}개, 기간: {START_YEAR}~{END_YEAR}")
    print(f"이미 수집: {len(existing)}건")

    total_new = 0
    total_skip = 0
    total_empty = 0
    errors = 0

    for i, (stock_code, corp_code, corp_name) in enumerate(stocks):
        for year in range(START_YEAR, END_YEAR + 1):
            for reprt_code, reprt_name in REPORT_CODES.items():
                # 이미 수집된 건 스킵
                if (stock_code, str(year), reprt_code) in existing:
                    total_skip += 1
                    continue

                try:
                    resp = requests.get(DART_URL, params={
                        "crtfc_key": DART_API_KEY,
                        "corp_code": corp_code,
                        "bsns_year": str(year),
                        "reprt_code": reprt_code,
                    }, timeout=15)
                    data = resp.json()
                except Exception as e:
                    errors += 1
                    continue

                if data.get("status") != "000":
                    total_empty += 1
                    time.sleep(0.05)
                    continue

                items = data.get("list", [])
                rcept_no = ""
                shares_common = 0
                shares_pref = 0
                fiscal_date = ""

                for item in items:
                    se = item.get("se", "")
                    rcept_no = item.get("rcept_no", rcept_no)
                    fiscal_date = item.get("stlm_dt", fiscal_date)

                    if "보통주" in se:
                        shares_common = parse_number(item.get("istc_totqy", "0"))
                    elif "우선주" in se:
                        shares_pref = parse_number(item.get("istc_totqy", "0"))

                disclosure_date = rcept_no[:8] if len(rcept_no) >= 8 else ""

                if shares_common > 0 and disclosure_date:
                    conn.execute("""
                        INSERT OR REPLACE INTO shares_outstanding
                        (stock_code, disclosure_date, fiscal_date, shares_common, shares_pref, report_type)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (stock_code, disclosure_date, fiscal_date, shares_common, shares_pref, reprt_code))
                    total_new += 1

                time.sleep(API_DELAY)

        # 종목 단위 커밋 + 진행률
        if (i + 1) % 50 == 0 or i == len(stocks) - 1:
            conn.commit()
            print(f"  [{i+1}/{len(stocks)}] {corp_name} | 신규: {total_new}, 스킵: {total_skip}, 빈응답: {total_empty}, 에러: {errors}")

    conn.commit()
    conn.close()
    print(f"\n=== 완료: 신규 {total_new}건, 스킵 {total_skip}건, 빈응답 {total_empty}건, 에러 {errors}건 ===")


def main():
    print(f"=== Step 2: DART 발행주식수 수집 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    collect_shares()


if __name__ == "__main__":
    main()
