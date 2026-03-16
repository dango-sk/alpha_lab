"""
Alpha Lab 일일 파이프라인 (Railway PostgreSQL)

실행: python scripts/run_pipeline.py
  --daily     매일 실행 (기본)
  --monthly   월초 실행 (마스터 + 재무 + 유니버스 포함)

매일 (~5분):
  1. Forward/Consensus     FnSpace → PG (증분)
  2. 백테스트 캐시 갱신    factor_engine → JSON
  3. 강건성 검증           IS/OOS + bootstrap + 롤링윈도우 → JSON
  4. Railway 배포          git commit + railway up

※ 주가(daily_price) + 시총은 LG 그램에서 Railway PG로 별도 업로드

월초 추가 (1~3일 or --monthly):
  + 마스터 스냅샷          FnSpace CompanyListApi → PG fnspace_master
  + 재무 보충              FnSpace FinanceApi → PG fnspace_finance

월초 + 15일:
  + 유니버스 재구축        PG → PG universe (biweekly 리밸런싱 대응)
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

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

PG_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:NgHDMsgiGwbvMpWHLUgTQguaedbGoxvv@metro.proxy.rlwy.net:50087/railway",
)
FNSPACE_API_KEY = os.environ.get("FNSPACE_API_KEY", "")
API_DELAY = 0.5

BENCHMARK_ETFS = ["069500", "292150", "229200"]  # KODEX200, KRX300, 코스닥150


def get_pg():
    import psycopg2
    conn = psycopg2.connect(PG_URL)
    conn.cursor().execute("SET search_path TO alpha_lab, public")
    conn.commit()
    return conn


def timeit(label):
    """간단 타이머 데코레이터"""
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n{'─'*50}")
            print(f"  ▶ {label}")
            print(f"{'─'*50}")
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.t0
            s = f"{elapsed:.0f}초" if elapsed < 60 else f"{elapsed/60:.1f}분"
            print(f"  ✓ {label} 완료 ({s})")
    return Timer()


# ═══════════════════════════════════════════════════════════
# 1. 주가 업데이트 (pykrx → PG)
# ═══════════════════════════════════════════════════════════

def step_update_prices():
    """최근 15일 주가를 pykrx에서 가져와 PG daily_price에 upsert"""
    from pykrx import stock as pykrx
    from psycopg2.extras import execute_values

    conn = get_pg()
    cur = conn.cursor()

    today = datetime.now()
    start = (today - timedelta(days=20)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    # 대상 종목: PG fnspace_master (최신 스냅샷)
    cur.execute("""
        SELECT DISTINCT SUBSTRING(stock_code FROM 2)
        FROM alpha_lab.fnspace_master
        WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM alpha_lab.fnspace_master)
    """)
    stocks = [r[0] for r in cur.fetchall()]

    # 벤치마크 ETF 추가
    for etf in BENCHMARK_ETFS:
        if etf not in stocks:
            stocks.append(etf)

    print(f"  대상: {len(stocks)}종목, 기간: {start}~{end}")

    # 이미 있는 최신 날짜 확인
    cur.execute("SELECT MAX(trade_date) FROM alpha_lab.daily_price")
    last_date = cur.fetchone()[0]
    print(f"  PG 마지막 주가: {last_date}")

    total_new = 0
    errors = 0

    for i, ticker in enumerate(stocks):
        try:
            df = pykrx.get_market_ohlcv(start, end, ticker)
            if df.empty:
                time.sleep(0.15)
                continue
        except Exception:
            errors += 1
            time.sleep(0.15)
            continue

        rows = []
        for idx, row in df.iterrows():
            trade_date = idx.strftime("%Y-%m-%d")
            close = int(row.get("종가", 0))
            if close == 0:
                continue
            volume = int(row.get("거래량", 0))
            trade_amount = int(row.get("거래대금", 0))
            if trade_amount == 0:
                trade_amount = close * volume

            rows.append((
                ticker, trade_date,
                int(row.get("시가", 0)), int(row.get("고가", 0)),
                int(row.get("저가", 0)), close,
                volume, trade_amount,
            ))

        if rows:
            execute_values(cur, """
                INSERT INTO alpha_lab.daily_price
                    (stock_code, trade_date, open, high, low, close, volume, trade_amount)
                VALUES %s
                ON CONFLICT (stock_code, trade_date) DO UPDATE SET
                    open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                    close=EXCLUDED.close, volume=EXCLUDED.volume,
                    trade_amount=EXCLUDED.trade_amount
            """, rows)
            total_new += len(rows)

        time.sleep(0.15)

        if (i + 1) % 100 == 0:
            conn.commit()
            print(f"    [{i+1}/{len(stocks)}] 누적 {total_new:,}건, 에러 {errors}")

    conn.commit()

    # 최신 거래일 확인
    cur.execute("SELECT MAX(trade_date) FROM alpha_lab.daily_price")
    new_last = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(DISTINCT stock_code) FROM alpha_lab.daily_price
        WHERE trade_date = %s
    """, (new_last,))
    cnt = cur.fetchone()[0]
    print(f"  upsert: {total_new:,}건, 에러: {errors}")
    print(f"  최신 주가: {new_last} ({cnt}종목)")

    conn.close()


# ═══════════════════════════════════════════════════════════
# 2. 시총 갱신 (전일 시총 기반 추정)
# ═══════════════════════════════════════════════════════════

def step_update_marketcap():
    """market_cap이 0이거나 NULL인 최근 행을 전일 시총×(종가비율)로 채움"""
    conn = get_pg()
    cur = conn.cursor()

    # 최신 거래일
    cur.execute("SELECT MAX(trade_date) FROM alpha_lab.daily_price")
    latest = cur.fetchone()[0]
    if not latest:
        print("  daily_price 비어있음")
        conn.close()
        return

    cutoff = (datetime.strptime(latest, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

    # market_cap이 비어있는 최근 30일 행을 직전 시총 데이터로 채움
    cur.execute("""
        WITH prev AS (
            SELECT DISTINCT ON (d.stock_code)
                d.stock_code,
                d.close AS prev_close,
                d.market_cap AS prev_mcap
            FROM alpha_lab.daily_price d
            WHERE d.market_cap > 0 AND d.close > 0
              AND d.trade_date >= %s
            ORDER BY d.stock_code, d.trade_date DESC
        )
        UPDATE alpha_lab.daily_price dp
        SET market_cap = ROUND(p.prev_mcap * (dp.close::FLOAT / p.prev_close))
        FROM prev p
        WHERE dp.stock_code = p.stock_code
          AND dp.trade_date >= %s
          AND (dp.market_cap IS NULL OR dp.market_cap = 0)
          AND dp.close > 0
    """, (cutoff, cutoff))

    updated = cur.rowcount
    conn.commit()
    print(f"  시총 갱신: {updated:,}건")
    conn.close()


# ═══════════════════════════════════════════════════════════
# 3. Forward / Consensus 수집 (증분)
# ═══════════════════════════════════════════════════════════

def step_collect_consensus():
    """step7_collect_consensus의 증분 모드 호출"""
    from step7_collect_consensus import (
        get_pg_conn, create_tables, get_universe_stocks,
        collect_forward, collect_consensus_daily,
    )

    conn = get_pg_conn()
    cur = conn.cursor()
    create_tables(cur)
    conn.commit()

    stocks = get_universe_stocks(cur)
    print(f"  universe 종목: {len(stocks)}개")

    to_date = datetime.now().strftime("%Y%m%d")

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

    if from_date > to_date:
        print("  이미 최신 상태")
        conn.close()
        return

    print(f"  수집 기간: {from_date} ~ {to_date}")
    collect_forward(conn, cur, stocks, from_date, to_date)
    collect_consensus_daily(conn, cur, stocks, from_date, to_date)
    conn.close()


# ═══════════════════════════════════════════════════════════
# 4. 백테스트 캐시 갱신
# ═══════════════════════════════════════════════════════════

def step_sync_to_sqlite():
    """PG → 로컬 SQLite 증분 동기화 (백테스트/대시보드용)"""
    import sqlite3
    import psycopg2

    sqlite_path = Path(__file__).parent.parent / "data" / "alpha_lab.db"
    sq = sqlite3.connect(str(sqlite_path))
    sq.execute("PRAGMA journal_mode=WAL")
    sq.execute("PRAGMA synchronous=OFF")

    pg = psycopg2.connect(PG_URL)
    pg_cur = pg.cursor()
    pg_cur.execute("SET search_path TO alpha_lab, public")

    # 테이블별 동기화 설정: (테이블명, 증분키, full_replace 여부)
    sync_tables = [
        ("daily_price", "trade_date", False),
        ("fnspace_forward", "trade_date", False),
        ("fnspace_finance", None, True),      # 전체 교체 (작은 테이블)
        ("fnspace_master", None, True),        # 전체 교체
        ("universe", None, True),              # 전체 교체
        ("har_universe", None, True),          # 전체 교체
    ]

    for tbl, incr_col, full_replace in sync_tables:
        # PG 테이블 존재 확인
        pg_cur.execute(f"SELECT * FROM {tbl} LIMIT 0")
        cols = [d[0] for d in pg_cur.description]
        placeholders = ", ".join(["?"] * len(cols))

        # SQLite 테이블 없으면 생성
        sq.execute(f"CREATE TABLE IF NOT EXISTS {tbl} ({', '.join(cols)})")

        if full_replace:
            # 작은 테이블: DROP + 재생성
            pg_cur.execute(f"SELECT count(*) FROM {tbl}")
            pg_cnt = pg_cur.fetchone()[0]
            sq.execute(f"DELETE FROM {tbl}")
            pg_cur.execute(f"SELECT * FROM {tbl}")
            rows = pg_cur.fetchall()
            if rows:
                sq.executemany(f"INSERT INTO {tbl} VALUES ({placeholders})", rows)
            sq.commit()
            print(f"    {tbl}: {len(rows):,}행 (전체 교체)")
        else:
            # 큰 테이블: 증분만
            local_max = sq.execute(f"SELECT MAX({incr_col}) FROM {tbl}").fetchone()[0]
            if local_max:
                pg_cur.execute(f"SELECT * FROM {tbl} WHERE {incr_col} > %s", (local_max,))
            else:
                pg_cur.execute(f"SELECT * FROM {tbl}")

            total = 0
            while True:
                rows = pg_cur.fetchmany(50000)
                if not rows:
                    break
                sq.executemany(f"INSERT OR REPLACE INTO {tbl} VALUES ({placeholders})", rows)
                total += len(rows)
            sq.commit()
            print(f"    {tbl}: +{total:,}행 (증분, since {local_max or 'N/A'})")

    # 인덱스
    sq.execute("CREATE INDEX IF NOT EXISTS idx_dp_code_date ON daily_price(stock_code, trade_date)")
    sq.execute("CREATE INDEX IF NOT EXISTS idx_dp_date ON daily_price(trade_date)")
    sq.execute("CREATE INDEX IF NOT EXISTS idx_ff_code_year ON fnspace_finance(stock_code, fiscal_year)")
    sq.execute("CREATE INDEX IF NOT EXISTS idx_fwd_date ON fnspace_forward(trade_date)")
    sq.execute("CREATE INDEX IF NOT EXISTS idx_univ_date ON universe(rebal_date, rebal_type)")
    sq.commit()

    sq.close()
    pg.close()
    print("  동기화 완료")


def step_backtest():
    """step7_backtest 실행 + 캐시 저장 (로컬 SQLite 사용)"""
    os.environ["DATABASE_URL"] = ""  # 로컬 SQLite 강제
    from step7_backtest import run_all_backtests, save_backtest_cache, show_comparison

    results = run_all_backtests()
    if results:
        show_comparison(results)
        save_backtest_cache(results)
    else:
        print("  백테스트 결과 없음")


# ═══════════════════════════════════════════════════════════
# 5. 강건성 검증
# ═══════════════════════════════════════════════════════════

def step_robustness():
    """step8_robustness 실행 + 캐시 저장"""
    from step8_robustness import (
        test_is_oos_split, test_statistical_significance,
        test_rolling_window, show_results, generate_chart,
        save_robustness_cache,
    )

    is_oos = test_is_oos_split()
    stat = test_statistical_significance()
    rolling = test_rolling_window(stat["full_results"])
    show_results(is_oos, stat, rolling)
    generate_chart(stat, rolling)
    save_robustness_cache(is_oos, stat, rolling)


# ═══════════════════════════════════════════════════════════
# 6. Railway 배포
# ═══════════════════════════════════════════════════════════

def step_deploy_railway():
    """git commit + railway up으로 대시보드 배포"""
    import subprocess

    project_dir = Path(__file__).parent.parent

    # 캐시 파일 커밋
    subprocess.run(["git", "add", "cache/"], cwd=str(project_dir), capture_output=True)
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(project_dir), capture_output=True,
    )
    if result.returncode != 0:
        subprocess.run(
            ["git", "commit", "-m", f"파이프라인: 캐시 갱신 ({datetime.now():%Y-%m-%d})"],
            cwd=str(project_dir), capture_output=True,
        )
        print("  캐시 커밋 완료")

    # railway up
    print("  railway up 실행 중...")
    result = subprocess.run(
        ["railway", "up"],
        cwd=str(project_dir),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode == 0:
        print("  Railway 배포 성공")
    else:
        print(f"  Railway 배포 실패: {result.stderr[:200]}")


# ═══════════════════════════════════════════════════════════
# 월초: 마스터 스냅샷 수집 (FnSpace → PG)
# ═══════════════════════════════════════════════════════════

def step_collect_master():
    """FnSpace CompanyListApi로 이번달 마스터 스냅샷을 PG에 저장"""
    import requests
    from psycopg2.extras import execute_values

    if not FNSPACE_API_KEY:
        print("  [SKIP] FNSPACE_API_KEY 없음")
        return

    conn = get_pg()
    cur = conn.cursor()

    snapshot_date = datetime.now().strftime("%Y-%m")
    exclude_kw = ["스팩", "SPAC", "ETF", "ETN", "리츠", "REIT"]

    # 이미 수집된 월인지 확인
    cur.execute(
        "SELECT COUNT(*) FROM alpha_lab.fnspace_master WHERE snapshot_date = %s",
        (snapshot_date,)
    )
    if cur.fetchone()[0] > 0:
        print(f"  {snapshot_date} 이미 수집됨 → 스킵")
        conn.close()
        return

    total = 0
    yyyymmdd = datetime.now().strftime("%Y%m01")

    for mkttype, mkt_name in [(1, "KOSPI"), (2, "KOSDAQ")]:
        resp = requests.get(f"https://www.fnspace.com/Api/CompanyListApi", params={
            "key": FNSPACE_API_KEY, "format": "json",
            "mkttype": str(mkttype), "date": yyyymmdd,
        }, timeout=30)
        data = resp.json()
        if data.get("success") != "true":
            print(f"  [WARN] {mkt_name}: {data.get('errmsg')}")
            continue

        rows = []
        for s in data.get("dataset", []):
            code = s.get("ITEM_CD", "")
            name = s.get("ITEM_NM", "")
            if not code or not name:
                continue
            if any(kw in name.upper() for kw in [k.upper() for k in exclude_kw]):
                continue
            rows.append((
                code, name, mkt_name,
                s.get("SEC_CD"), s.get("SEC_CD_NM"),
                s.get("SEC_CD_DET"), s.get("SEC_CD_DET_NM"),
                s.get("FINACC_TYP"), snapshot_date,
            ))
        if rows:
            execute_values(cur, """
                INSERT INTO alpha_lab.fnspace_master
                (stock_code, stock_name, market, sec_cd, sec_cd_nm,
                 sec_cd_det, sec_cd_det_nm, finacc_typ, snapshot_date)
                VALUES %s
                ON CONFLICT (stock_code, snapshot_date) DO NOTHING
            """, rows)
            total += len(rows)

        time.sleep(API_DELAY)

    conn.commit()
    print(f"  {snapshot_date} 마스터: {total}종목")
    conn.close()


# ═══════════════════════════════════════════════════════════
# 월초: 재무 보충 (FnSpace FinanceApi → PG)
# ═══════════════════════════════════════════════════════════

def step_collect_finance():
    """fnspace_finance 보충 수집 → PG 직접 저장"""
    import requests
    from psycopg2.extras import execute_values

    if not FNSPACE_API_KEY:
        print("  [SKIP] FNSPACE_API_KEY 없음")
        return

    FINANCE_ITEMS = {
        "M382500": "pbr", "M211500": "roe", "M211700": "roic",
        "M213000": "ev", "M124000": "ic", "M331010": "ev_ebit",
        "M123100": "ebit", "M113900": "net_debt", "M113800": "interest_debt",
        "M115000": "total_equity", "M312000": "eps", "M314000": "bps",
        "M382100": "per", "M383300": "psr", "M331030": "ev_ebitda",
        "M123200": "ebitda", "M121000": "revenue", "M121500": "operating_income",
        "M122700": "net_income", "M211000": "oi_margin",
        "M431800": "div_yield", "M385110": "pcf",
    }
    COL_NAMES = list(FINANCE_ITEMS.values())
    MAX_CODES = 10
    MAX_ITEMS = 10

    conn = get_pg()
    cur = conn.cursor()

    # 전체 종목
    cur.execute("SELECT DISTINCT stock_code FROM alpha_lab.fnspace_master")
    all_codes = [r[0] for r in cur.fetchall()]

    # 이미 수집된 종목별 연도 수
    cur.execute("""
        SELECT stock_code, COUNT(DISTINCT fiscal_year)
        FROM alpha_lab.fnspace_finance GROUP BY stock_code
    """)
    existing = {r[0]: r[1] for r in cur.fetchall()}

    target_years = datetime.now().year - 2017 + 1
    target_codes = [c for c in all_codes if existing.get(c, 0) < target_years]
    print(f"  전체: {len(all_codes)}종목, 보충 대상: {len(target_codes)}종목")

    if not target_codes:
        print("  보충할 종목 없음")
        conn.close()
        return

    item_keys = list(FINANCE_ITEMS.keys())
    item_groups = [item_keys[i:i+MAX_ITEMS] for i in range(0, len(item_keys), MAX_ITEMS)]
    code_chunks = [target_codes[i:i+MAX_CODES] for i in range(0, len(target_codes), MAX_CODES)]

    total_saved = 0
    api_calls = 0

    for ci, code_chunk in enumerate(code_chunks):
        codes_str = ",".join(code_chunk)

        for item_group in item_groups:
            items_str = ",".join(item_group)
            try:
                resp = requests.get("https://www.fnspace.com/Api/FinanceApi", params={
                    "key": FNSPACE_API_KEY, "format": "json",
                    "code": codes_str, "item": items_str,
                    "consolgb": "M", "annualgb": "A",
                    "fraccyear": "2017", "toaccyear": str(datetime.now().year),
                }, timeout=30)
                data = resp.json()
                api_calls += 1
            except Exception:
                api_calls += 1
                time.sleep(API_DELAY)
                continue

            if data.get("success") != "true" or not data.get("dataset"):
                time.sleep(API_DELAY)
                continue

            rows = []
            for sd in data["dataset"]:
                code = sd.get("CODE", "")
                for row in sd.get("DATA", []):
                    fy = row.get("FS_YEAR")
                    fq = row.get("FS_QTR", "Annual")
                    if not fy:
                        continue
                    values = {}
                    for ik in item_group:
                        val = row.get(ik)
                        if val is not None:
                            values[FINANCE_ITEMS[ik]] = val
                    if values:
                        r = [code, fy, fq] + [values.get(c) for c in COL_NAMES]
                        rows.append(tuple(r))

            if rows:
                cols_sql = ", ".join(COL_NAMES)
                ph = ", ".join(["%s"] * (3 + len(COL_NAMES)))
                update_set = ", ".join(f"{c}=EXCLUDED.{c}" for c in COL_NAMES)
                execute_values(cur, f"""
                    INSERT INTO alpha_lab.fnspace_finance
                    (stock_code, fiscal_year, fiscal_quarter, {cols_sql})
                    VALUES %s
                    ON CONFLICT (stock_code, fiscal_year, fiscal_quarter) DO UPDATE SET
                    {update_set}, updated_at=NOW()
                """, rows, template=f"({ph})")
                total_saved += len(rows)

            time.sleep(API_DELAY)

        if (ci + 1) % 20 == 0 or ci == len(code_chunks) - 1:
            conn.commit()
            print(f"    [{(ci+1)*MAX_CODES}/{len(target_codes)}] 저장: {total_saved:,}건")

    conn.commit()
    print(f"  재무 수집 완료: {total_saved:,}건 (API {api_calls}건)")
    conn.close()


# ═══════════════════════════════════════════════════════════
# 월초: 유니버스 재구축
# ═══════════════════════════════════════════════════════════

def step_build_universe():
    """step6_build_universe 호출"""
    from step6_build_universe import build_universe
    build_universe()


# ═══════════════════════════════════════════════════════════
# macOS 알림
# ═══════════════════════════════════════════════════════════

def notify(title, message):
    import subprocess
    try:
        subprocess.run([
            "osascript", "-e",
            f'display notification "{message}" with title "Alpha Lab" subtitle "{title}" sound name "Glass"'
        ], timeout=5, capture_output=True)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Alpha Lab 파이프라인 (PG)")
    parser.add_argument("--monthly", action="store_true", help="월초 전체 실행")
    parser.add_argument("--skip-backtest", action="store_true", help="백테스트 스킵")
    args = parser.parse_args()

    is_monthly = args.monthly or datetime.now().day <= 3
    mode = "월초 전체" if is_monthly else "일일"

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Alpha Lab {mode} 파이프라인")
    print(f"  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  DB: Railway PostgreSQL")
    print(f"{'='*60}")

    steps = []
    failed = []

    def run(name, func):
        steps.append(name)
        try:
            with timeit(name):
                func()
        except Exception as e:
            failed.append((name, str(e)))
            print(f"  ✗ {name} 실패: {e}")

    # ── 월초: 마스터 + 재무 ──
    if is_monthly:
        run("마스터 스냅샷 수집", step_collect_master)
        run("재무 보충 수집", step_collect_finance)

    # ── 월초 + 15일: 유니버스 재구축 (biweekly 대응) ──
    if is_monthly or datetime.now().day in range(15, 18):
        run("유니버스 재구축", step_build_universe)

    # ── 매일 (주가/시총은 LG 그램에서 PG로 별도 업로드) ──
    run("Forward/Consensus 수집", step_collect_consensus)

    # ── PG → 로컬 SQLite 동기화 ──
    run("PG→SQLite 동기화", step_sync_to_sqlite)

    if not args.skip_backtest:
        run("백테스트", step_backtest)
        run("강건성 검증", step_robustness)

    # ── 배포 ──
    run("Railway 배포", step_deploy_railway)

    # ── 요약 ──
    elapsed = time.time() - t0
    elapsed_str = f"{elapsed/60:.1f}분" if elapsed >= 60 else f"{elapsed:.0f}초"

    print(f"\n{'='*60}")
    print(f"  {mode} 파이프라인 완료  ({elapsed_str})")
    print(f"  실행: {len(steps)}단계, 실패: {len(failed)}건")
    if failed:
        for name, err in failed:
            print(f"    ✗ {name}: {err}")
    print(f"{'='*60}")

    notify("파이프라인 완료", f"{mode} {elapsed_str}, 실패 {len(failed)}건")


if __name__ == "__main__":
    main()
