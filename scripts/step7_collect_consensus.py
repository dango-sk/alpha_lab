"""
Step 7: Forward + Consensus Daily 수집 (Railway PostgreSQL)
실행: python scripts/step7_collect_consensus.py

universe 테이블에 있는 종목만 대상으로 수집.

수집 데이터:
  1. Forward 지표 (Consensus4Api) — Fwd EPS, PER, EBIT, EBITDA, ROE 등
  2. 추정실적 Daily (Consensus3Api) — 일별 컨센서스 히스토리

옵션:
  --forward         Forward만 수집
  --consensus       Consensus Daily만 수집
  --from-date DATE  시작일 (YYYYMMDD)
  --to-date DATE    종료일 (YYYYMMDD)
  --backfill        백테스트 기간 전체 수집 (2018-01-01~)
"""
import sys
import os
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

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

FNSPACE_API_KEY = os.environ.get("FNSPACE_API_KEY", "6D1C172DBC447A2A47AF")
BASE_URL = "https://www.fnspace.com/Api"
PG_URL = "postgresql://postgres:NgHDMsgiGwbvMpWHLUgTQguaedbGoxvv@metro.proxy.rlwy.net:50087/railway"

API_DELAY = 0.5
MAX_CODES_PER_CALL = 10
PROGRESS_FILE = Path(__file__).parent.parent / "data" / ".step7_progress.json"


def load_progress():
    import json
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except:
            pass
    return {"forward_done": [], "consensus_done": {}}


def save_progress(progress):
    import json
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False))

# Forward 지표 (Consensus4Api)
FORWARD_ITEMS = {
    "E312060": "fwd_eps",        # EPS(Fwd.12M, 지배)
    "E382160": "fwd_per",        # P/E(Fwd.12M)
    "E124004": "fwd_ebit",       # EBIT(Fwd.12M)
    "E123060": "fwd_ebitda",     # EBITDA(Fwd.12M)
    "E331060": "fwd_ev_ebitda",  # EV/EBITDA(Fwd.12M)
    "E121060": "fwd_revenue",    # 매출액(Fwd.12M)
    "E121560": "fwd_oi",         # 영업이익(Fwd.12M)
    "E122770": "fwd_ni",         # 당기순이익(Fwd.12M, 지배)
    "E211560": "fwd_roe",        # ROE(Fwd.12M, 지배)
    "E314060": "fwd_bps",        # BPS(Fwd.12M, 지배)
}

# 추정실적 Daily (Consensus3Api)
CONSENSUS_DAILY_ITEMS = {
    "E121500": "est_oi",         # 추정 영업이익
    "E122710": "est_ni",         # 추정 당기순이익(지배)
    "E124000": "est_ebit",       # 추정 EBIT
    "E113900": "est_net_debt",   # 추정 순부채
    "E312000": "est_eps",        # 추정 EPS(지배)
    "E211500": "est_roe",        # 추정 ROE(지배)
}

FORWARD_COLS = list(FORWARD_ITEMS.values())
CONSENSUS_COLS = list(CONSENSUS_DAILY_ITEMS.values())


def api_call(endpoint, params, silent=False):
    url = f"{BASE_URL}/{endpoint}"
    params["key"] = FNSPACE_API_KEY
    params["format"] = "json"
    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        if data.get("success") == "true":
            return data
        if not silent:
            errmsg = data.get("errmsg", "unknown")
            if not hasattr(api_call, '_err_count'):
                api_call._err_count = {}
            key = f"{endpoint}:{errmsg}"
            api_call._err_count[key] = api_call._err_count.get(key, 0) + 1
            if api_call._err_count[key] <= 3:
                print(f"    [WARN] {endpoint}: {errmsg}")
        return None
    except Exception as e:
        if not silent:
            print(f"    [ERR] {endpoint}: {e}")
        return None


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_pg_conn():
    import psycopg2
    return psycopg2.connect(PG_URL)


def create_tables(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alpha_lab.fnspace_forward (
            stock_code  TEXT,
            trade_date  TEXT,
            fwd_eps REAL, fwd_per REAL,
            fwd_ebit REAL, fwd_ebitda REAL, fwd_ev_ebitda REAL,
            fwd_revenue REAL, fwd_oi REAL, fwd_ni REAL,
            fwd_roe REAL, fwd_bps REAL,
            updated_at  TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (stock_code, trade_date)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alpha_lab.fnspace_consensus_daily (
            stock_code  TEXT,
            trade_date  TEXT,
            fiscal_year INTEGER,
            est_oi REAL, est_ni REAL, est_ebit REAL,
            est_net_debt REAL, est_eps REAL, est_roe REAL,
            updated_at  TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (stock_code, trade_date, fiscal_year)
        )
    """)


def get_universe_stocks(cur):
    """universe 테이블의 고유 종목 코드 목록 (A 접두사 붙여서 반환)"""
    cur.execute("SELECT DISTINCT stock_code FROM alpha_lab.universe")
    codes = [row[0] for row in cur.fetchall()]
    # FnSpace API는 A 접두사 필요
    return [f"A{c}" if not c.startswith("A") else c for c in codes]


def collect_forward(conn, cur, stocks, from_date, to_date):
    """Consensus4Api로 Forward 12M 지표 수집
    - 종목별 마지막 수집일 확인 → 그 이후부터만 API 호출
    - 데이터 없는 종목은 from_date부터 전체 수집
    """
    print(f"\n{'='*60}")
    print(f"  Forward 지표 수집 (Consensus4Api)")
    print(f"  대상: {len(stocks)}종목, 기간: {from_date} ~ {to_date}")
    print(f"{'='*60}")

    if from_date > to_date:
        print("  이미 최신 상태입니다.")
        return

    to_fmt = f"{to_date[:4]}-{to_date[4:6]}-{to_date[6:8]}"
    fr_fmt = f"{from_date[:4]}-{from_date[4:6]}-{from_date[6:8]}"

    # 종목별 daily_price 거래일 수 vs fnspace_forward 데이터 수 비교
    # 일치하면 완료 → 스킵, 불일치면 빈 날짜만 수집
    cur.execute("""
        WITH dp AS (
            SELECT stock_code, COUNT(*) as dp_cnt
            FROM alpha_lab.daily_price
            WHERE trade_date BETWEEN %s AND %s
            GROUP BY stock_code
        ), fwd AS (
            SELECT stock_code, COUNT(*) as fwd_cnt, MIN(trade_date) as min_dt, MAX(trade_date) as max_dt
            FROM alpha_lab.fnspace_forward
            WHERE trade_date BETWEEN %s AND %s
            GROUP BY stock_code
        )
        SELECT dp.stock_code, dp.dp_cnt, COALESCE(fwd.fwd_cnt, 0), fwd.min_dt, fwd.max_dt
        FROM dp LEFT JOIN fwd ON dp.stock_code = fwd.stock_code
    """, (fr_fmt, to_fmt, fr_fmt, to_fmt))
    stock_stats = {f"A{r[0]}": (int(r[1]), int(r[2]), r[3], r[4]) for r in cur.fetchall()}

    groups = defaultdict(list)
    fully_done = 0
    for s in stocks:
        stat = stock_stats.get(s)
        code6 = s[1:] if s.startswith("A") else s

        if stat:
            dp_cnt, fwd_cnt, min_dt, max_dt = stat
            # 거래일 수 대비 90% 이상 있으면 완료로 간주
            if fwd_cnt >= dp_cnt * 0.9 and min_dt and max_dt:
                fully_done += 1
                continue

        # fnspace_forward에서 직접 min/max 확인
        if stat and stat[1] > 0 and stat[2] and stat[3]:
            min_date, max_date = stat[2], stat[3]
            # 앞쪽 빈 구간
            if fr_fmt < min_date:
                prev_day = (datetime.strptime(min_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y%m%d")
                groups[(from_date, prev_day)].append(s)
            # 뒤쪽 빈 구간
            if max_date < to_fmt:
                next_day = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d")
                groups[(next_day, to_date)].append(s)
        else:
            groups[(from_date, to_date)].append(s)

    total_stocks = len(set(s for v in groups.values() for s in v))
    print(f"  스킵: {fully_done}종목 (완료), 수집: {total_stocks}종목 ({len(groups)}개 구간)")

    if not groups:
        print("  모든 종목 이미 수집됨.")
        return

    MAX_ITEMS_PER_FORWARD_CALL = 3
    item_chunks = []
    keys = list(FORWARD_ITEMS.keys())
    for i in range(0, len(keys), MAX_ITEMS_PER_FORWARD_CALL):
        item_chunks.append(keys[i:i + MAX_ITEMS_PER_FORWARD_CALL])

    from psycopg2.extras import execute_values
    total_saved = 0
    api_ok, api_fail = 0, 0
    batch_count = 0

    progress = load_progress()
    fwd_done_set = set(progress.get("forward_done", []))

    for (grp_from, grp_to), grp_stocks in sorted(groups.items()):
        # 기간을 연도별로 쪼개기 (API 응답 크기 제한)
        year_ranges = []
        fr_y = int(grp_from[:4])
        to_y = int(grp_to[:4])
        for y in range(fr_y, to_y + 1):
            yr_start = grp_from if y == fr_y else f"{y}0101"
            yr_end = grp_to if y == to_y else f"{y}1231"
            year_ranges.append((yr_start, yr_end))

        # 이미 완료된 종목 제외
        remaining = [s for s in grp_stocks if f"{s}|{grp_from}|{grp_to}" not in fwd_done_set]
        if not remaining:
            continue

        print(f"  구간 {grp_from}~{grp_to}: {len(remaining)}종목 ({len(year_ranges)}년분할, 스킵 {len(grp_stocks)-len(remaining)})")
        code_chunks = list(chunk_list(remaining, MAX_CODES_PER_CALL))

        for yr_from, yr_to in year_ranges:
            yr_fr_fmt = f"{yr_from[:4]}-{yr_from[4:6]}-{yr_from[6:8]}"
            yr_to_fmt = f"{yr_to[:4]}-{yr_to[4:6]}-{yr_to[6:8]}"

            for code_chunk in code_chunks:
                # DB에 이미 해당 연도 데이터가 충분한 종목 제외
                # 구간 일수의 50% 이상 데이터 있으면 스킵
                days_in_range = (datetime.strptime(yr_to_fmt, "%Y-%m-%d") - datetime.strptime(yr_fr_fmt, "%Y-%m-%d")).days
                min_records = max(5, int(days_in_range * 0.63))  # 거래일~70% × 90%
                codes_6 = [c[1:] if c.startswith("A") else c for c in code_chunk]
                placeholders_check = ",".join(["%s"] * len(codes_6))
                cur.execute(f"""
                    SELECT stock_code FROM alpha_lab.fnspace_forward
                    WHERE stock_code IN ({placeholders_check})
                      AND trade_date BETWEEN %s AND %s
                    GROUP BY stock_code
                    HAVING COUNT(*) >= {min_records}
                """, codes_6 + [yr_fr_fmt, yr_to_fmt])
                already_done = {f"A{r[0]}" for r in cur.fetchall()}
                filtered_chunk = [c for c in code_chunk if c not in already_done]

                if not filtered_chunk:
                    batch_count += 1
                    continue

                codes_str = ",".join(filtered_chunk)
                merged = {}

                for item_keys in item_chunks:
                    items_str = ",".join(item_keys)
                    data = api_call("Consensus4Api", {
                        "code": codes_str,
                        "item": items_str,
                        "consolgb": "M",
                        "frdate": yr_from,
                        "todate": yr_to,
                    })

                    if data and data.get("dataset"):
                        api_ok += 1
                        for sd in data["dataset"]:
                            code = sd.get("CODE", "")
                            code6 = code[1:] if code.startswith("A") else code
                            if code6 not in merged:
                                merged[code6] = {}
                            for row in sd.get("DATA", []):
                                dt = row.get("DT", "").strip()
                                if not dt:
                                    continue
                                if dt not in merged[code6]:
                                    merged[code6][dt] = {}
                                for ik in item_keys:
                                    val = row.get(ik)
                                    if val is not None:
                                        merged[code6][dt][FORWARD_ITEMS[ik]] = val
                    else:
                        api_fail += 1

                    time.sleep(API_DELAY)

                # DB 저장
                rows = []
                for code6, dt_map in merged.items():
                    for dt, values in dt_map.items():
                        if not values:
                            continue
                        trade_date = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}" if len(dt) == 8 else dt[:10]
                        row = [code6, trade_date] + [values.get(c) for c in FORWARD_COLS]
                        rows.append(tuple(row))

                if rows:
                    cols_sql = ", ".join(FORWARD_COLS)
                    placeholders = ", ".join(["%s"] * (2 + len(FORWARD_COLS)))
                    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in FORWARD_COLS)
                    execute_values(cur, f"""
                        INSERT INTO alpha_lab.fnspace_forward (stock_code, trade_date, {cols_sql})
                        VALUES %s
                        ON CONFLICT (stock_code, trade_date) DO UPDATE SET
                        {update_set}, updated_at = NOW()
                    """, rows, template=f"({placeholders})")
                    total_saved += len(rows)

                batch_count += 1
                conn.commit()

                # 마지막 연도 처리 완료 시 진행상황 저장
                if yr_from == year_ranges[-1][0]:
                    for c in code_chunk:
                        fwd_done_set.add(f"{c}|{grp_from}|{grp_to}")
                    progress["forward_done"] = list(fwd_done_set)
                    save_progress(progress)

                if batch_count % 5 == 0:
                    print(f"  [{batch_count}] {yr_from[:4]}년 저장: {total_saved:,}건")

    conn.commit()
    print(f"\n  Forward 완료: {total_saved:,}건 (API 성공 {api_ok}, 실패 {api_fail})")


def collect_consensus_daily(conn, cur, stocks, from_date, to_date, target_years=None):
    """Consensus3Api로 일별 컨센서스 히스토리 수집
    - 연도별 + 종목별 마지막 수집일 확인 → 그 이후부터만 API 호출
    - 데이터 없는 종목은 from_date부터 전체 수집
    """
    if target_years is None:
        current_year = datetime.now().year
        target_years = [str(current_year), str(current_year + 1)]

    print(f"\n{'='*60}")
    print(f"  Consensus Daily 수집 (Consensus3Api)")
    print(f"  대상: {len(stocks)}종목, 기간: {from_date} ~ {to_date}")
    print(f"  추정 대상 연도: {target_years}")
    print(f"{'='*60}")

    if from_date > to_date:
        print("  이미 최신 상태입니다.")
        return

    to_fmt = f"{to_date[:4]}-{to_date[4:6]}-{to_date[6:8]}"
    fr_fmt = f"{from_date[:4]}-{from_date[4:6]}-{from_date[6:8]}"

    # 연도별 종목별 수집 범위 조회 (min, max)
    cur.execute("""
        SELECT stock_code, fiscal_year::TEXT, MIN(trade_date), MAX(trade_date)
        FROM alpha_lab.fnspace_consensus_daily
        GROUP BY stock_code, fiscal_year
    """)
    ranges_by_year = defaultdict(dict)  # {year: {Acode: (min, max)}}
    for code, fy, mn, mx in cur.fetchall():
        ranges_by_year[fy][f"A{code}"] = (mn, mx)

    items_str = ",".join(CONSENSUS_DAILY_ITEMS.keys())
    total_saved = 0
    api_ok, api_fail = 0, 0

    from psycopg2.extras import execute_values

    for target_year in target_years:
        year_ranges = ranges_by_year.get(target_year, {})

        # fiscal_year별 합리적 수집 범위 설정
        # 컨센서스 추정치는 fiscal_year 전년 초 ~ fiscal_year+1년 말 정도까지 존재
        fy = int(target_year)
        fy_from = max(from_date, f"{fy-1}0101")
        fy_to = min(to_date, f"{fy+1}1231")
        fy_fr_fmt = f"{fy_from[:4]}-{fy_from[4:6]}-{fy_from[6:8]}"
        fy_to_fmt = f"{fy_to[:4]}-{fy_to[4:6]}-{fy_to[6:8]}"

        # 종목별 빈 구간 파악
        groups = defaultdict(list)  # {(grp_from, grp_to): [stock_codes]}
        fully_done = 0
        for s in stocks:
            r = year_ranges.get(s)
            if not r:
                groups[(fy_from, fy_to)].append(s)
                continue

            min_date, max_date = r
            has_gap = False

            # 앞쪽 gap (7일 이상만)
            if fy_fr_fmt < min_date:
                gap_days = (datetime.strptime(min_date, "%Y-%m-%d") - datetime.strptime(fy_fr_fmt, "%Y-%m-%d")).days
                if gap_days >= 7:
                    prev_day = (datetime.strptime(min_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y%m%d")
                    groups[(fy_from, prev_day)].append(s)
                    has_gap = True

            # 뒤쪽 gap (7일 이상만)
            if max_date < fy_to_fmt:
                gap_days = (datetime.strptime(fy_to_fmt, "%Y-%m-%d") - datetime.strptime(max_date, "%Y-%m-%d")).days
                if gap_days >= 7:
                    next_day = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d")
                    groups[(next_day, fy_to)].append(s)
                    has_gap = True

            if not has_gap:
                fully_done += 1

        total_stocks = len(set(s for v in groups.values() for s in v))
        print(f"\n  {target_year}년: 스킵 {fully_done}, 수집 {total_stocks}종목 ({len(groups)}개 구간)")

        if not groups:
            print(f"  {target_year}년 모든 종목 이미 수집됨.")
            continue

        batch_count = 0
        for (grp_from, grp_to), grp_stocks in sorted(groups.items()):
            print(f"    구간 {grp_from}~{grp_to}: {len(grp_stocks)}종목")
            code_chunks = list(chunk_list(grp_stocks, MAX_CODES_PER_CALL))

            grp_fr_fmt = f"{grp_from[:4]}-{grp_from[4:6]}-{grp_from[6:8]}"
            grp_to_fmt = f"{grp_to[:4]}-{grp_to[4:6]}-{grp_to[6:8]}"

            for code_chunk in code_chunks:
                # DB에 이미 해당 구간+연도 데이터 충분한 종목 제외
                days_in_range = (datetime.strptime(grp_to_fmt, "%Y-%m-%d") - datetime.strptime(grp_fr_fmt, "%Y-%m-%d")).days
                min_records = max(5, int(days_in_range * 0.63))  # 거래일~70% × 90%
                codes_6 = [c[1:] if c.startswith("A") else c for c in code_chunk]
                ph = ",".join(["%s"] * len(codes_6))
                cur.execute(f"""
                    SELECT stock_code FROM alpha_lab.fnspace_consensus_daily
                    WHERE stock_code IN ({ph})
                      AND fiscal_year = %s
                      AND trade_date BETWEEN %s AND %s
                    GROUP BY stock_code
                    HAVING COUNT(*) >= {min_records}
                """, codes_6 + [int(target_year), grp_fr_fmt, grp_to_fmt])
                already_done = {f"A{r[0]}" for r in cur.fetchall()}
                filtered_chunk = [c for c in code_chunk if c not in already_done]

                if not filtered_chunk:
                    batch_count += 1
                    continue

                codes_str = ",".join(filtered_chunk)

                data = api_call("Consensus3Api", {
                    "code": codes_str,
                    "item": items_str,
                    "consolgb": "M",
                    "annualgb": "A",
                    "accdategb": "C",
                    "fraccyear": target_year,
                    "toaccyear": target_year,
                    "frdate": grp_from,
                    "todate": grp_to,
                })

                rows = []
                if data and data.get("dataset"):
                    api_ok += 1
                    for sd in data["dataset"]:
                        code = sd.get("CODE", "")
                        code6 = code[1:] if code.startswith("A") else code
                        for row_data in sd.get("DATA", []):
                            dt = row_data.get("DT", "").strip()
                            fy = row_data.get("FS_YEAR")
                            if not dt or not fy:
                                continue
                            trade_date = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}" if len(dt) == 8 else dt[:10]
                            values = {}
                            for item_cd, col_name in CONSENSUS_DAILY_ITEMS.items():
                                val = row_data.get(item_cd)
                                if val is not None:
                                    values[col_name] = val
                            if values:
                                row = [code6, trade_date, fy] + [values.get(c) for c in CONSENSUS_COLS]
                                rows.append(tuple(row))
                else:
                    api_fail += 1

                if rows:
                    cols_sql = ", ".join(CONSENSUS_COLS)
                    placeholders = ", ".join(["%s"] * (3 + len(CONSENSUS_COLS)))
                    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in CONSENSUS_COLS)
                    execute_values(cur, f"""
                        INSERT INTO alpha_lab.fnspace_consensus_daily
                        (stock_code, trade_date, fiscal_year, {cols_sql})
                        VALUES %s
                        ON CONFLICT (stock_code, trade_date, fiscal_year) DO UPDATE SET
                        {update_set}, updated_at = NOW()
                    """, rows, template=f"({placeholders})")
                    total_saved += len(rows)

                time.sleep(API_DELAY)

                batch_count += 1
                conn.commit()
                if batch_count % 5 == 0:
                    print(f"  [{batch_count}] {target_year}년 저장: {total_saved:,}건")

        conn.commit()

    print(f"\n  Consensus Daily 완료: {total_saved:,}건 (API 성공 {api_ok}, 실패 {api_fail})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 7: Forward + Consensus Daily 수집")
    parser.add_argument("--forward", action="store_true", help="Forward만 수집")
    parser.add_argument("--consensus", action="store_true", help="Consensus Daily만 수집")
    parser.add_argument("--from-date", type=str, default=None, help="시작일 (YYYYMMDD)")
    parser.add_argument("--to-date", type=str, default=None, help="종료일 (YYYYMMDD)")
    parser.add_argument("--backfill", action="store_true", help="백테스트 기간 전체 수집 (2018-01-01~)")
    args = parser.parse_args()

    if not FNSPACE_API_KEY:
        print("FNSPACE_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    print(f"=== Step 7: Forward + Consensus Daily 수집 ({datetime.now():%Y-%m-%d %H:%M}) ===")

    conn = get_pg_conn()
    cur = conn.cursor()
    create_tables(cur)
    conn.commit()

    stocks = get_universe_stocks(cur)
    print(f"  universe 종목: {len(stocks)}개")

    to_date = args.to_date or datetime.now().strftime("%Y%m%d")

    # from_date 결정
    if args.backfill:
        from_date = "20180101"
    elif args.from_date:
        from_date = args.from_date
    else:
        # 증분: DB 마지막 날짜 + 1일
        cur.execute("SELECT MAX(trade_date) FROM alpha_lab.fnspace_forward")
        last_fwd = cur.fetchone()[0]
        cur.execute("SELECT MAX(trade_date) FROM alpha_lab.fnspace_consensus_daily")
        last_con = cur.fetchone()[0]

        if last_fwd or last_con:
            last = max(filter(None, [last_fwd, last_con]))
            last_dt = datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
            from_date = last_dt.strftime("%Y%m%d")
        else:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")

    no_option = not args.forward and not args.consensus
    do_forward = args.forward or no_option
    do_consensus = args.consensus or no_option

    if do_forward:
        collect_forward(conn, cur, stocks, from_date, to_date)

    if do_consensus:
        if args.backfill:
            years = [str(y) for y in range(2018, datetime.now().year + 2)]
            collect_consensus_daily(conn, cur, stocks, from_date, to_date, target_years=years)
        else:
            collect_consensus_daily(conn, cur, stocks, from_date, to_date)

    # 요약
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT stock_code), COUNT(DISTINCT trade_date) FROM alpha_lab.fnspace_forward")
    fwd = cur.fetchone()
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT stock_code), COUNT(DISTINCT trade_date) FROM alpha_lab.fnspace_consensus_daily")
    con = cur.fetchone()

    print(f"\n{'='*60}")
    print(f"  Forward:         {fwd[0]:,}건 ({fwd[1]}종목, {fwd[2]}일)")
    print(f"  Consensus Daily: {con[0]:,}건 ({con[1]}종목, {con[2]}일)")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
