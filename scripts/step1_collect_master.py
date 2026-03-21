"""
Step 1: FnSpace CompanyListApi로 종목 마스터 월별 스냅샷 수집
실행: python scripts/step1_collect_master.py

- 2017-01 ~ 현재까지 매월 1일 기준 종목 리스트 수집
- KOSPI + KOSDAQ (스팩/ETF/ETN/리츠 제외)
- 생존자 편향 제거를 위해 과거 시점 종목 리스트 보존
- 이미 수집된 월은 스킵
"""
import sqlite3
import sys
import os
import time
import requests
from pathlib import Path
from datetime import datetime, date

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

EXCLUDE_KEYWORDS = ["스팩", "SPAC", "ETF", "ETN", "리츠", "REIT"]


def api_call(endpoint, params):
    url = f"{BASE_URL}/{endpoint}"
    params["key"] = FNSPACE_API_KEY
    params["format"] = "json"
    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        if data.get("success") == "true":
            return data
        else:
            print(f"  ⚠️ {endpoint}: {data.get('errmsg', 'unknown')}")
            return None
    except Exception as e:
        print(f"  ⚠️ {endpoint} 요청 실패: {e}")
        return None


def generate_monthly_dates(start_year=2017, start_month=1):
    """수집 대상 월 리스트 생성 (매월 1일)"""
    dates = []
    today = date.today()
    y, m = start_year, start_month
    while date(y, m, 1) <= today:
        dates.append(f"{y:04d}{m:02d}01")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


def collect_snapshot(conn, snapshot_yyyymmdd):
    """특정 월의 KOSPI+KOSDAQ 종목 스냅샷 수집"""
    snapshot_label = f"{snapshot_yyyymmdd[:4]}-{snapshot_yyyymmdd[4:6]}"

    # 이미 수집된 월인지 확인
    existing = conn.execute(
        "SELECT COUNT(*) FROM fnspace_master WHERE snapshot_date = ?",
        (snapshot_label,),
    ).fetchone()[0]
    if existing > 0:
        return existing, True  # skipped

    total = 0
    for mkttype, mkt_name in [(1, "KOSPI"), (2, "KOSDAQ")]:
        data = api_call("CompanyListApi", {
            "mkttype": str(mkttype),
            "date": snapshot_yyyymmdd,
        })
        if not data:
            continue

        stocks = data.get("dataset", [])
        for s in stocks:
            code = s.get("ITEM_CD", "")
            name = s.get("ITEM_NM", "")
            if not code or not name:
                continue
            if any(kw in name.upper() for kw in EXCLUDE_KEYWORDS):
                continue

            conn.execute("""
                INSERT OR REPLACE INTO fnspace_master
                (stock_code, stock_name, market, sec_cd, sec_cd_nm,
                 sec_cd_det, sec_cd_det_nm, finacc_typ, snapshot_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                code, name, mkt_name,
                s.get("SEC_CD"), s.get("SEC_CD_NM"),
                s.get("SEC_CD_DET"), s.get("SEC_CD_DET_NM"),
                s.get("FINACC_TYP"),
                snapshot_label,
            ))
            total += 1

        time.sleep(API_DELAY)

    conn.commit()
    return total, False


def main():
    print(f"=== Step 1: 종목 마스터 월별 스냅샷 수집 ({datetime.now():%Y-%m-%d %H:%M}) ===")

    if not FNSPACE_API_KEY:
        print("[ERROR] FNSPACE_API_KEY가 설정되지 않았습니다")
        return

    conn = sqlite3.connect(str(LAB_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    dates = generate_monthly_dates(2017, 1)
    print(f"수집 대상: {len(dates)}개월 (2017-01 ~ {dates[-1][:4]}-{dates[-1][4:6]})")

    collected = 0
    skipped = 0

    for i, d in enumerate(dates):
        label = f"{d[:4]}-{d[4:6]}"
        count, was_skipped = collect_snapshot(conn, d)
        if was_skipped:
            skipped += 1
        else:
            collected += 1
            print(f"  [{i+1}/{len(dates)}] {label}: {count}종목")

    conn.close()
    print(f"\n=== 완료: {collected}개월 신규 수집, {skipped}개월 스킵 ===")


if __name__ == "__main__":
    main()
