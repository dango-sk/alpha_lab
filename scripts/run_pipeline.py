"""
Alpha Lab 일일 파이프라인 (Railway PostgreSQL)

실행: python scripts/run_pipeline.py
  --daily     매일 실행 (기본)
  --monthly   월초 실행 (마스터 + 재무 + 유니버스 포함)

매일:
  1. Forward/Consensus     FnSpace → PG (증분)
  2. 백테스트 캐시 갱신    4콤보 (KOSPI×월간, KOSPI×격주, KOSPI+KOSDAQ×월간, KOSPI+KOSDAQ×격주)
  3. 강건성 검증           IS/OOS + bootstrap + 롤링윈도우 → PG

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
from datetime import datetime
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
# 주가/시총은 LG 그램에서 PG에 별도 업로드 (step_update_prices, step_update_marketcap 제거)
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# Forward / Consensus 수집 (PG 증분)
# ═══════════════════════════════════════════════════════════

def _get_consensus_conn_and_range(from_date_override=None):
    """공통: PG 연결 + 수집 범위 반환.
    from_date_override: "YYYYMMDD" 형식으로 시작일 강제 지정 (일회성 보충용)
    """
    from step7_collect_consensus import get_pg_conn, create_tables, get_universe_stocks
    conn = get_pg_conn()
    cur = conn.cursor()
    create_tables(cur)
    conn.commit()
    stocks = get_universe_stocks(cur)
    to_date = datetime.now().strftime("%Y%m%d")
    if from_date_override:
        from_date = from_date_override
    else:
        cur.execute("""
            SELECT DISTINCT trade_date FROM alpha_lab.daily_price
            ORDER BY trade_date DESC LIMIT 3
        """)
        recent_dates = [r[0] for r in cur.fetchall()]
        from_date = min(recent_dates).replace("-", "") if recent_dates else to_date
    from_dash = f"{from_date[:4]}-{from_date[4:6]}-{from_date[6:]}"
    return conn, cur, stocks, from_date, to_date, from_dash


def step_collect_forward(from_date_override=None):
    """Forward 지표만 수집 (최근 2거래일 DELETE + 재수집)"""
    from step7_collect_consensus import collect_forward, save_progress
    conn, cur, stocks, from_date, to_date, from_dash = _get_consensus_conn_and_range(from_date_override)
    print(f"  universe 종목: {len(stocks)}개")

    cur.execute("DELETE FROM alpha_lab.fnspace_forward WHERE trade_date >= %s", (from_dash,))
    print(f"  forward 삭제: {from_dash} ~ ({cur.rowcount}건)")
    conn.commit()
    save_progress({})

    print(f"  수집 기간: {from_date} ~ {to_date}")
    collect_forward(conn, cur, stocks, from_date, to_date)
    conn.close()


def step_collect_consensus_daily(from_date_override=None):
    """Consensus Daily만 수집 (최근 2거래일 DELETE + 재수집)"""
    import time as _time
    from psycopg2.extras import execute_values
    from step7_collect_consensus import (
        api_call, chunk_list,
        CONSENSUS_DAILY_ITEMS, CONSENSUS_COLS, MAX_CODES_PER_CALL, API_DELAY,
    )
    conn, cur, stocks, from_date, to_date, from_dash = _get_consensus_conn_and_range(from_date_override)
    print(f"  universe 종목: {len(stocks)}개")

    cur.execute("DELETE FROM alpha_lab.fnspace_consensus_daily WHERE trade_date >= %s", (from_dash,))
    print(f"  consensus daily 삭제: {from_dash} ~ ({cur.rowcount}건)")
    conn.commit()

    current_year = datetime.now().year
    target_years = [str(current_year), str(current_year + 1)]
    items_str = ",".join(CONSENSUS_DAILY_ITEMS.keys())
    total_saved = 0

    print(f"  수집 기간: {from_date} ~ {to_date}, 연도 {target_years}")
    for target_year in target_years:
        for code_chunk in chunk_list(stocks, MAX_CODES_PER_CALL):
            codes_str = ",".join(code_chunk)
            data = api_call("Consensus3Api", {
                "code": codes_str,
                "item": items_str,
                "consolgb": "M",
                "annualgb": "A",
                "accdategb": "C",
                "fraccyear": target_year,
                "toaccyear": target_year,
                "frdate": from_date,
                "todate": to_date,
            })

            rows = []
            if data and data.get("dataset"):
                for sd in data["dataset"]:
                    code = sd.get("CODE", "")
                    code6 = code[1:] if code.startswith("A") else code
                    for row_data in sd.get("DATA", []):
                        dt = row_data.get("DT", "").strip()
                        fy = row_data.get("FS_YEAR")
                        if not dt or not fy:
                            continue
                        trade_date = dt[:10] if "-" in dt else f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}"
                        values = {}
                        for item_cd, col_name in CONSENSUS_DAILY_ITEMS.items():
                            val = row_data.get(item_cd)
                            if val is not None:
                                values[col_name] = val
                        if values:
                            row = [code6, trade_date, fy] + [values.get(c) for c in CONSENSUS_COLS]
                            rows.append(tuple(row))

            # 중복 제거 (stock_code, trade_date, fiscal_year)
            seen = {}
            for r in rows:
                seen[(r[0], r[1], r[2])] = r
            rows = list(seen.values())

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
                conn.commit()

            _time.sleep(API_DELAY)

        print(f"    {target_year}년 완료: 누적 {total_saved:,}건")

    print(f"  Consensus Daily 완료: {total_saved:,}건")
    conn.close()


def step_collect_consensus(from_date_override=None):
    """Forward + Consensus Daily 둘 다 수집"""
    step_collect_forward(from_date_override)
    step_collect_consensus_daily(from_date_override)


# ═══════════════════════════════════════════════════════════
# 4. 백테스트 캐시 갱신
# ═══════════════════════════════════════════════════════════


def step_backtest(skip_combos=None):
    """step7_backtest 실행 + 캐시 저장 (PG 직접, 4 콤보)"""
    os.environ["DATABASE_URL"] = PG_URL  # PG 직접 사용
    from step7_backtest import run_all_backtests, save_backtest_cache, \
        save_portfolio_cache, show_comparison
    from config.settings import BACKTEST_CONFIG as _BC
    from lib.factor_engine import clear_factor_cache

    skip_set = set(skip_combos or [])

    combos = [
        ("KOSPI", "monthly"),
        ("KOSPI", "biweekly"),
        ("KOSPI+KOSDAQ", "monthly"),
        ("KOSPI+KOSDAQ", "biweekly"),
    ]

    for universe, rebal_type in combos:
        combo_key = f"{universe}_{rebal_type}"
        if combo_key in skip_set:
            print(f"\n  ── {universe} / {rebal_type} ── 스킵")
            continue
        print(f"\n  ── {universe} / {rebal_type} ──")
        orig_u, orig_r = _BC.get("universe"), _BC.get("rebal_type")
        _BC["universe"] = universe
        _BC["rebal_type"] = rebal_type
        try:
            clear_factor_cache()
            results = run_all_backtests(rebal_type=rebal_type)
            if results:
                show_comparison(results)
                save_backtest_cache(results, universe=universe, rebal_type=rebal_type)
                save_portfolio_cache(results, universe=universe, rebal_type=rebal_type)
            else:
                print(f"  {universe}/{rebal_type} 백테스트 결과 없음")
        finally:
            _BC["universe"] = orig_u
            _BC["rebal_type"] = orig_r


def step_custom_strategies():
    """저장된 커스텀 전략들을 재계산하여 PG에 업데이트"""
    os.environ["DATABASE_URL"] = PG_URL
    import psycopg2
    from lib.data import run_strategy_backtest, save_strategy

    conn = psycopg2.connect(PG_URL)
    conn.cursor().execute("SET search_path TO alpha_lab, public")
    conn.commit()
    cur = conn.cursor()
    cur.execute("""
        SELECT name, strategy_code, universe, rebal_type
        FROM backtest_cache
        WHERE strategy_code IS NOT NULL
          AND name NOT IN ('A0', 'KOSPI', '__ROBUSTNESS__')
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("  커스텀 전략 없음")
        return

    # 레짐조합 전략은 코드가 빈 문자열이므로 제외
    rows = [(n, c, u, r) for n, c, u, r in rows if c and c.strip() and not n.startswith("레짐조합_")]
    print(f"  커스텀 전략 {len(rows)}개 재계산 시작")
    for name, code, universe, rebal_type in rows:
        try:
            print(f"    ▶ {name} ({universe}/{rebal_type})")
            result = run_strategy_backtest(
                strategy_code=code,
                universe=universe,
                rebal_type=rebal_type,
            )
            if result and "error" not in result:
                save_strategy(
                    name=name,
                    code=code,
                    results=result,
                    universe=universe,
                    rebal_type=rebal_type,
                )
                tr = result.get("CUSTOM", result).get("total_return", 0)
                print(f"      ✓ 완료 (총수익률: {tr:+.1%})")
            else:
                err = result.get("error", "no result") if result else "no result"
                print(f"      ✗ 실패: {err}")
        except Exception as e:
            print(f"      ✗ 에러: {e}")


def step_regime_combo_strategies():
    """저장된 레짐조합 전략들을 재계산하여 PG에 업데이트"""
    import re
    import psycopg2
    os.environ["DATABASE_URL"] = PG_URL
    from lib.data import run_regime_combo_backtest, save_strategy

    conn = psycopg2.connect(PG_URL)
    conn.cursor().execute("SET search_path TO alpha_lab, public")
    conn.commit()
    cur = conn.cursor()
    cur.execute("""
        SELECT name, results_json, universe, rebal_type
        FROM backtest_cache
        WHERE name LIKE '레짐조합_%%'
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("  레짐조합 전략 없음")
        return

    print(f"  레짐조합 전략 {len(rows)}개 재계산 시작")
    for name, results_json, universe, rebal_type in rows:
        try:
            # 이름에서 bull/bear 키 파싱: 레짐조합_{bull}↑_{bear}↓
            body = name.replace("레짐조합_", "")
            m = re.match(r"(.+?)↑_(.+?)↓$", body)
            if not m:
                print(f"    ▶ {name} — 이름 파싱 실패, 스킵")
                continue

            bull_key, bear_key = m.group(1), m.group(2)
            # 저장된 결과에서 regime_mode 추출 (없으면 cycle)
            regime_mode = "cycle"
            _rj = results_json
            if isinstance(_rj, str):
                try:
                    _rj = json.loads(_rj)
                except Exception:
                    _rj = {}
            if isinstance(_rj, dict):
                regime_mode = _rj.get("_debug_regime_mode", "cycle")

            print(f"    ▶ {name} (bull={bull_key}, bear={bear_key}, mode={regime_mode})")
            result = run_regime_combo_backtest(
                bull_key=bull_key,
                bear_key=bear_key,
                universe=universe,
                rebal_type=rebal_type,
                regime_mode=regime_mode,
            )
            if result and "error" not in result:
                combo = result.get("REGIME_COMBO", {})
                save_strategy(
                    name=name,
                    code="",
                    results=combo,
                    universe=universe,
                    rebal_type=rebal_type,
                )
                tr = combo.get("total_return", 0)
                print(f"      ✓ 완료 (총수익률: {tr:+.1%})")
            else:
                err = result.get("error", "no result") if result else "no result"
                print(f"      ✗ 실패: {err}")
        except Exception as e:
            print(f"      ✗ 에러: {e}")


# ═══════════════════════════════════════════════════════════
# 5. 강건성 검증
# ═══════════════════════════════════════════════════════════

def step_collect_news():
    """네이트 뉴스 매크로 수집 (DB 마지막 수집일 ~ 어제)"""
    from collect_news_nate import run_daily
    inserted = run_daily()
    print(f"  뉴스 {inserted}건 수집")


def step_robustness():
    """step8_robustness 실행 + 캐시 저장"""
    from step8_robustness import (
        _load_cached_results,
        test_is_oos_split, test_statistical_significance,
        test_rolling_window, show_results, generate_chart,
        save_robustness_cache,
    )

    full_results = _load_cached_results()
    is_oos = test_is_oos_split(full_results)
    stat = test_statistical_significance(full_results)
    rolling = test_rolling_window(full_results)
    show_results(is_oos, stat, rolling)
    generate_chart(stat, rolling)
    save_robustness_cache(is_oos, stat, rolling)



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
        row_dict = {}  # (code, fy, fq) -> {col: val}

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

            for sd in data["dataset"]:
                code = sd.get("CODE", "")
                for row in sd.get("DATA", []):
                    fy = row.get("FS_YEAR")
                    fq = row.get("FS_QTR", "Annual")
                    if not fy:
                        continue
                    key = (code, fy, fq)
                    if key not in row_dict:
                        row_dict[key] = {}
                    for ik in item_group:
                        val = row.get(ik)
                        if val is not None:
                            row_dict[key][FINANCE_ITEMS[ik]] = val

            time.sleep(API_DELAY)

        # item_group 루프 끝난 뒤 중복 없이 한번에 insert
        if row_dict:
            rows = []
            for (code, fy, fq), vals in row_dict.items():
                r = [code, fy, fq] + [vals.get(c) for c in COL_NAMES]
                rows.append(tuple(r))
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

        if (ci + 1) % 20 == 0 or ci == len(code_chunks) - 1:
            conn.commit()
            print(f"    [{(ci+1)*MAX_CODES}/{len(target_codes)}] 저장: {total_saved:,}건")

    conn.commit()
    print(f"  재무 수집 완료: {total_saved:,}건 (API {api_calls}건)")
    conn.close()


# ═══════════════════════════════════════════════════════════
# 월초: TTM 계산
# ═══════════════════════════════════════════════════════════

def step_calc_ttm():
    """분기 재무 → TTM 계산 → fnspace_finance에 저장"""
    from step5b_calc_ttm import calc_ttm
    calc_ttm()


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

def step_ai_filter(strategy_path: str = None):
    """AI 종목 필터: 팩터 상위 30종목 → AI 분석 → 최종 10종목."""
    from lib.ai_stock_filter import run_ai_filter
    from lib.factor_engine import score_stocks_from_strategy, load_strategy_module, code_to_module, DEFAULT_STRATEGY_CODE
    from lib.db import get_conn

    conn = get_conn()
    try:
        calc_date = datetime.now().strftime("%Y-%m-%d")
        if strategy_path:
            strategy = load_strategy_module(strategy_path)
            print(f"  전략: {strategy_path}")
        else:
            strategy = code_to_module(DEFAULT_STRATEGY_CODE)
            print(f"  전략: 기본 (A0)")
        stocks = score_stocks_from_strategy(conn, calc_date, strategy)
        stocks = stocks[:30]
        print(f"  팩터 상위 {len(stocks)}종목 → AI 필터 시작")
        result = run_ai_filter(stocks, calc_date, conn)
        portfolio = result.get("final_portfolio", [])
        print(f"  최종 {len(portfolio)}종목 선정 완료")
        for item in portfolio:
            print(f"    {item.get('stock_name', '?'):>10s} | "
                  f"비중: {item.get('weight_pct', 0):5.1f}% | "
                  f"{item.get('reason', '')}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Alpha Lab 파이프라인 (PG)")
    parser.add_argument("--monthly", action="store_true", help="월초 전체 실행")
    parser.add_argument("--skip-backtest", action="store_true", help="백테스트 스킵")
    parser.add_argument("--only-backtest", action="store_true", help="유니버스+백테스트+커스텀만 실행 (수집 스킵)")
    parser.add_argument("--skip-universe", action="store_true", help="유니버스 재구축 스킵")
    parser.add_argument("--skip-combos", type=str, default="", help="스킵할 콤보 (예: KOSPI_monthly,KOSPI_biweekly)")
    parser.add_argument("--only-custom", action="store_true", help="커스텀 전략 재계산+강건성만 실행")
    parser.add_argument("--consensus-from", type=str, default="", help="Forward/Consensus 수집 시작일 (YYYYMMDD, 일회성 보충용)")
    parser.add_argument("--ai-filter", action="store_true", help="AI 종목 필터 실행 (30→10종목)")
    parser.add_argument("--only-ai-filter", action="store_true", help="AI 종목 필터만 실행")
    parser.add_argument("--ai-strategy", type=str, default=None, help="AI 필터에 사용할 전략 파일 경로")
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

    skip_combos = [s.strip() for s in args.skip_combos.split(",") if s.strip()] if args.skip_combos else []
    consensus_from = args.consensus_from or None

    if args.only_ai_filter:
        run("AI 종목 필터", lambda: step_ai_filter(args.ai_strategy))
    elif args.only_custom:
        # 커스텀 전략 재계산 + 강건성만
        run("커스텀 전략 재계산", step_custom_strategies)
        run("레짐조합 전략 재계산", step_regime_combo_strategies)
        run("강건성 검증", step_robustness)
    elif args.only_backtest:
        # 유니버스 + 백테스트 + 커스텀만 (수집 스킵)
        if not args.skip_universe:
            run("유니버스 재구축", step_build_universe)
        run("백테스트", lambda: step_backtest(skip_combos=skip_combos))
        run("커스텀 전략 재계산", step_custom_strategies)
        run("레짐조합 전략 재계산", step_regime_combo_strategies)
        run("강건성 검증", step_robustness)
    else:
        # ── 월초: 마스터 + 재무 + TTM ──
        if is_monthly:
            run("마스터 스냅샷 수집", step_collect_master)
            run("재무 보충 수집", step_collect_finance)
            run("TTM 계산", step_calc_ttm)

        # ── 유니버스 재구축 (매 실행 시) ──
        run("유니버스 재구축", step_build_universe)

        # ── 매일 (주가/시총은 LG 그램에서 PG로 별도 업로드) ──
        run("Forward/Consensus 수집", lambda: step_collect_consensus(consensus_from))
        run("뉴스 수집", step_collect_news)

        if not args.skip_backtest:
            run("백테스트", step_backtest)
            run("커스텀 전략 재계산", step_custom_strategies)
            run("레짐조합 전략 재계산", step_regime_combo_strategies)
            run("강건성 검증", step_robustness)

        if args.ai_filter:
            run("AI 종목 필터", lambda: step_ai_filter(args.ai_strategy))

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
