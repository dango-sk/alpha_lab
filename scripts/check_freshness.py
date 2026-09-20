"""
regime_agent.py 돌리기 직전 데이터 신선도 점검.

run_monthly_with_regime.sh 의 0-1 ~ 0-3 (collect_macro / backfill_global_indices /
collect_technical) 을 돌린 뒤, 1단계(regime_agent) 앞에서 실행한다.
stale 데이터로 레짐이 판정되는 사고를 미리 잡는 것이 목적.

사용법:
    .venv/bin/python check_freshness.py            # 기준일 = 오늘
    .venv/bin/python check_freshness.py 2026-08-31 # 기준일 명시
"""
import os
import sys
from datetime import date, datetime, timedelta

import psycopg2
from dotenv import load_dotenv

load_dotenv('/Users/namsugyeong/Desktop/alpha_lab/.env')

# regime_agent 가 실제로 읽는 것들만 점검한다.
MACRO_DAILY = [
    'kospi', 'sp500', 'sox', 'vix', 'dxy', 'us10y', 'usd_krw',
    'bond_1y', 'bond_10y', 'wti_daily',
    'investor_foreign_kospi', 'investor_institution_kospi', 'investor_individual_kospi',
]
TECH_SYMBOLS = ['KOSPI', 'SP500', 'SOX']
KOSPI_ETF = '069500'  # collect_technical 의 KOSPI 소스


def last_business_day(ref: date) -> date:
    """ref 이하의 마지막 평일. (공휴일은 보지 않음 — 여유 허용치로 흡수)"""
    d = ref
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def main():
    ref = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    target = last_business_day(ref)
    # 해외지표는 시차, 국내지표는 공휴일 때문에 며칠 여유를 준다.
    tol_days = 4
    floor = target - timedelta(days=tol_days)

    print(f"기준일 {ref} → 최근 평일 {target} (허용 지연 {tol_days}일, {floor} 이후면 OK)\n")

    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()
    fails = []

    def check(label, last, note=''):
        if last is None:
            ok = False
        else:
            last = date.fromisoformat(last) if isinstance(last, str) else last
            ok = last >= floor
        mark = '✅' if ok else '❌'
        print(f"  {mark} {label:34s} {last}  {note}")
        if not ok:
            fails.append(label)

    print("[macro_indicators — 일별]")
    for ind in MACRO_DAILY:
        cur.execute(
            "SELECT MAX(period) FROM alpha_lab.macro_indicators "
            "WHERE indicator=%s AND freq='D'", (ind,))
        check(ind, cur.fetchone()[0])

    print("\n[technical_indicators]")
    for sym in TECH_SYMBOLS:
        cur.execute(
            "SELECT MAX(trade_date) FROM alpha_lab.technical_indicators "
            "WHERE symbol=%s", (sym,))
        check(f'technical/{sym}', cur.fetchone()[0])

    print("\n[daily_price (LG그램 업로드분)]")
    cur.execute("SELECT MAX(trade_date) FROM alpha_lab.daily_price")
    check('daily_price 전체', cur.fetchone()[0])
    cur.execute("SELECT MAX(trade_date) FROM alpha_lab.daily_price WHERE stock_code=%s",
                (KOSPI_ETF,))
    check(f'daily_price {KOSPI_ETF}', cur.fetchone()[0], '(KOSPI 기술지표 소스)')

    print("\n[news_nate — regime_agent 프롬프트 입력]")
    cur.execute("SELECT MAX(published_date), COUNT(*) FROM alpha_lab.news_nate "
                "WHERE published_date LIKE %s", (f'{target:%Y-%m}%',))
    last_news, n_news = cur.fetchone()
    print(f"  {'✅' if n_news else '⚠️ '} {target:%Y-%m} 뉴스 {n_news}건 (최신 {last_news})")

    print("\n[regime_agent 기존 결과 — 중복/덮어쓰기 확인]")
    nxt = f"{(target.replace(day=1) + timedelta(days=32)):%Y-%m}"
    try:
        cur.execute("SELECT as_of, regime, created_at FROM alpha_lab.regime_agent_results "
                    "ORDER BY as_of DESC LIMIT 3")
        for r in cur.fetchall():
            print(f"     {r[0]}  {r[1]}  (생성 {r[2]})")
        print(f"     → 실행 후 {nxt}-01 행이 새로 생겼는지 확인할 것")
    except Exception as e:
        conn.rollback()
        print(f"     (테이블 조회 실패: {str(e)[:70]} — JSON 파일 방식일 수 있음)")

    conn.close()

    print()
    if fails:
        print(f"❌ stale {len(fails)}건: {', '.join(fails)}")
        print("   → regime_agent.py 실행하지 말 것. 0-1~0-3 단계 재확인 필요.")
        sys.exit(1)
    print("✅ 전부 최신. regime_agent.py 진행해도 됩니다.")


if __name__ == '__main__':
    main()
