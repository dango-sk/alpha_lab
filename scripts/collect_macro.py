"""
매크로 지표 수집 스크립트
- FnSpace EconomyApi: 선행지수, 장단기금리차, WTI, CPI, PPI, BSI, CSI
- yfinance: VIX
DB 테이블: alpha_lab.macro_indicators
"""

import os
import sys
import requests
import psycopg2
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATABASE_URL = os.environ['DATABASE_URL']
FNSPACE_KEY = os.environ.get('FNSPACE_API_KEY', 'D0E7A9A250B8C43545C5')
FNSPACE_URL = 'https://www.fnspace.com/Api/EconomyApi'

# ── FnSpace 월별 수집 항목 ───────────────────────────────────
FNSPACE_ITEMS = {
    'leading_index':    ('agKOCB2010LD',  '선행종합지수',     'M'),
    'yield_spread':     ('agKOCB2010LD8', '장단기금리차',     'M'),
    'wti_monthly':      ('aGLSCPUSWTI',   'WTI유가(월)',      'M'),
    'cpi':              ('ahKOPR20C0',    'CPI총지수',        'M'),
    'ppi':              ('ahKOPR20PIBA',  'PPI총지수',        'M'),
    'bsi_all':          ('arKOBSNFKBCA',  '전산업업황BSI',    'M'),
    'csi_outlook':      ('arKOCSBK13BB',  '향후경기전망CSI',  'M'),
}

# ── FnSpace 일별 수집 항목 ───────────────────────────────────
FNSPACE_DAILY_ITEMS = {
    'bond_1y':   ('arKOIRKSDATB1',  '국고채1년',    'D'),
    'bond_10y':  ('arKOIRKSDATB10', '국고채10년',   'D'),
    'usd_krw':   ('arKOFXUSDD',     '원/달러환율',  'D'),
    'wti_daily': ('cCHPRWDQWTI',    'WTI유가(일)',  'D'),
}


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alpha_lab.macro_indicators (
                id          SERIAL PRIMARY KEY,
                indicator   VARCHAR(50) NOT NULL,   -- 'leading_index', 'vix', etc.
                period      VARCHAR(10) NOT NULL,   -- '202501' (월별) or '2025-01-03' (일별)
                freq        CHAR(1) NOT NULL,       -- 'M' or 'D'
                value       FLOAT,
                name_ko     VARCHAR(100),
                updated_at  TIMESTAMP DEFAULT NOW(),
                UNIQUE(indicator, period)
            )
        """)
        conn.commit()
    print("✅ 테이블 확인/생성 완료")


def upsert_rows(conn, rows):
    """rows: list of (indicator, period, freq, value, name_ko)"""
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO alpha_lab.macro_indicators (indicator, period, freq, value, name_ko)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (indicator, period) DO UPDATE
              SET value = EXCLUDED.value, updated_at = NOW()
        """, rows)
    conn.commit()


def get_latest_period(conn, indicator):
    """DB에서 해당 indicator의 최신 period 반환"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(period) FROM alpha_lab.macro_indicators WHERE indicator = %s",
            (indicator,)
        )
        result = cur.fetchone()[0]
    return result  # None이면 데이터 없음


def collect_fnspace(conn, frdate='20180101', todate=None):
    if todate is None:
        todate = datetime.today().strftime('%Y%m%d')

    for key, (code, name, freq) in FNSPACE_ITEMS.items():
        # DB 최신값 이후부터만 수집
        latest = get_latest_period(conn, key)
        if latest:
            # 월별: '202501' → 다음달부터
            from_ym = datetime.strptime(latest, '%Y%m') + timedelta(days=32)
            start = from_ym.replace(day=1).strftime('%Y%m%d')
        else:
            start = frdate

        if start > todate:
            print(f"  ⏭ [{name}] 이미 최신 ({latest})")
            continue

        r = requests.get(FNSPACE_URL, params={
            'key': FNSPACE_KEY, 'format': 'json',
            'item': code, 'frdate': start, 'todate': todate,
            'accdategb': 'M'
        }, timeout=15)
        d = r.json()

        if d.get('errcd'):
            print(f"  ❌ [{name}] API 오류: {d.get('errmsg', '')}")
            continue

        data = d.get('dataset', [{}])[0].get('DATA', []) if d.get('dataset') else []
        if not data:
            print(f"  ⏭ [{name}] 신규 데이터 없음 (최신: {latest})")
            continue
        rows = [(key, row['DT'], freq, row['AMOUNT'], name) for row in data]
        upsert_rows(conn, rows)
        print(f"  ✅ [{name}] {len(rows)}건 저장 (최신: {data[-1]['DT']} = {data[-1]['AMOUNT']})")


def collect_fnspace_daily(conn, frdate='20180101', todate=None):
    if todate is None:
        todate = datetime.today().strftime('%Y%m%d')

    for key, (code, name, freq) in FNSPACE_DAILY_ITEMS.items():
        # DB 최신값 이후부터만 수집
        latest = get_latest_period(conn, key)
        if latest:
            # 일별: '2025-01-03' → 다음날부터
            from_dt = datetime.strptime(latest, '%Y-%m-%d') + timedelta(days=1)
            start = from_dt.strftime('%Y%m%d')
        else:
            start = frdate

        if start > todate:
            print(f"  ⏭ [{name}] 이미 최신 ({latest})")
            continue

        r = requests.get(FNSPACE_URL, params={
            'key': FNSPACE_KEY, 'format': 'json',
            'item': code, 'frdate': start, 'todate': todate,
        }, timeout=15)
        d = r.json()

        if d.get('errcd'):
            print(f"  ❌ [{name}] API 오류: {d.get('errmsg', '')}")
            continue

        data = d.get('dataset', [{}])[0].get('DATA', []) if d.get('dataset') else []
        if not data:
            print(f"  ⏭ [{name}] 신규 데이터 없음 (최신: {latest})")
            continue
        rows = []
        for row in data:
            dt = row['DT']
            period = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}" if len(dt) == 8 else dt
            rows.append((key, period, freq, row['AMOUNT'], name))
        upsert_rows(conn, rows)
        print(f"  ✅ [{name}] {len(rows)}건 저장 (최신: {rows[-1][1]} = {rows[-1][3]})")


def collect_vix(conn, start='2018-01-01', end=None):
    if end is None:
        # yfinance는 end가 exclusive → 내일까지 지정해야 오늘 데이터 포함
        end = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

    # DB 최신값 이후부터만 수집
    latest = get_latest_period(conn, 'vix')
    if latest:
        from_dt = datetime.strptime(latest, '%Y-%m-%d') + timedelta(days=1)
        start = from_dt.strftime('%Y-%m-%d')

    if start > end:
        print(f"  ⏭ [VIX] 이미 최신 ({latest})")
        return

    df = yf.download('^VIX', start=start, end=end, progress=False)
    if df.empty:
        print("  ❌ [VIX] 데이터 없음")
        return

    # yfinance 멀티인덱스 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rows = [
        ('vix', row.name.strftime('%Y-%m-%d'), 'D', float(row['Close']), 'VIX변동성지수')
        for _, row in df.iterrows()
        if not pd.isna(row['Close'])
    ]
    upsert_rows(conn, rows)
    print(f"  ✅ [VIX] {len(rows)}건 저장 (최신: {rows[-1][1]} = {rows[-1][3]:.2f})")


def main():
    print("=" * 50)
    print("매크로 지표 수집 시작")
    print("=" * 50)

    conn = psycopg2.connect(DATABASE_URL)

    create_table(conn)

    print("\n[FnSpace 월별 지표]")
    collect_fnspace(conn)

    print("\n[FnSpace 일별 지표]")
    collect_fnspace_daily(conn)

    print("\n[yfinance VIX 일별]")
    collect_vix(conn)

    conn.close()
    print("\n완료!")


if __name__ == '__main__':
    main()
