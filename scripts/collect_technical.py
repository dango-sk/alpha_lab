"""
Technical 지표 수집/계산 스크립트
- KOSPI(069500 ETF 기준) MA50/MA200, MACD, RSI → DB 저장
- S&P500, SOX → macro_indicators 일별 종가 기반 동일 지표 계산 → DB 저장

DB 테이블: alpha_lab.technical_indicators (symbol 컬럼으로 구분)
"""

import os
import sys
import psycopg2
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
sys.path.append(str(Path(__file__).parent.parent))

DATABASE_URL = os.environ['DATABASE_URL']
KOSPI_CODE = '069500'  # KODEX 200 (KOSPI 추종)


def migrate_table(conn):
    """symbol 컬럼이 없으면 추가 (이미 있으면 스킵)"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'alpha_lab' AND table_name = 'technical_indicators'
              AND column_name = 'symbol'
        """)
        if cur.fetchone():
            print("✅ 테이블 확인 완료 (symbol 컬럼 존재)")
            return

        print("  symbol 컬럼 추가 중...")
        cur.execute("ALTER TABLE alpha_lab.technical_indicators ADD COLUMN symbol VARCHAR(20) NOT NULL DEFAULT 'KOSPI'")
        cur.execute("ALTER TABLE alpha_lab.technical_indicators DROP CONSTRAINT IF EXISTS technical_indicators_trade_date_indicator_key")
        cur.execute("ALTER TABLE alpha_lab.technical_indicators ADD CONSTRAINT technical_indicators_date_ind_sym_key UNIQUE(trade_date, indicator, symbol)")
        conn.commit()
        print("  ✅ symbol 컬럼 추가 완료")


def upsert_rows(conn, rows, batch_size=100):
    """rows: list of (trade_date, indicator, symbol, value)"""
    total = len(rows)
    with conn.cursor() as cur:
        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]
            cur.executemany("""
                INSERT INTO alpha_lab.technical_indicators (trade_date, indicator, symbol, value)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (trade_date, indicator, symbol) DO UPDATE
                  SET value = EXCLUDED.value, updated_at = NOW()
            """, batch)
            conn.commit()
            print(f"    {min(i + batch_size, total)}/{total}", end="\r")


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_and_store(conn, price, symbol):
    """price: Series(DatetimeIndex → close). symbol: 'KOSPI', 'SP500', 'SOX'"""
    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()
    ma50_signal = np.where(price > ma50, 1.0, -1.0)
    ma200_signal = np.where(price > ma200, 1.0, -1.0)
    macd_line, macd_signal_line, macd_hist = calc_macd(price)
    rsi = calc_rsi(price)

    rows = []
    for dt, row in pd.DataFrame({
        'ma50': ma50,
        'ma200': ma200,
        'ma50_signal': ma50_signal,
        'ma200_signal': ma200_signal,
        'macd': macd_line,
        'macd_signal': macd_signal_line,
        'macd_hist': macd_hist,
        'rsi14': rsi,
        'close': price,
    }).iterrows():
        for col, val in row.items():
            if pd.notna(val):
                rows.append((dt.date(), col, symbol, float(val)))

    upsert_rows(conn, rows)
    latest = price.index[-1].date()
    print(f"  ✅ {symbol} 지표 {len(rows)}건 저장 (최신: {latest})")
    print(f"     MA50={ma50.iloc[-1]:.1f}, MA200={ma200.iloc[-1]:.1f}, RSI={rsi.iloc[-1]:.1f}, MACD={macd_line.iloc[-1]:.2f}")


def collect_kospi(conn):
    """KOSPI ETF 가격 → 기술지표 계산/저장"""
    df = pd.read_sql("""
        SELECT trade_date, close
        FROM alpha_lab.daily_price
        WHERE stock_code = %s
        ORDER BY trade_date
    """, conn, params=(KOSPI_CODE,), parse_dates=['trade_date'])

    if df.empty:
        print(f"  ❌ {KOSPI_CODE} 가격 데이터 없음")
        return

    df = df.set_index('trade_date').sort_index()
    price = df['close'].astype(float)
    print(f"  KOSPI({KOSPI_CODE}) 가격 로드: {len(price)}건 ({price.index[0].date()} ~ {price.index[-1].date()})")
    calc_and_store(conn, price, symbol='KOSPI')


def collect_global(conn):
    """macro_indicators의 S&P500/SOX 일별 종가 → 기술지표 계산/저장"""
    for ind_name, symbol in [('sp500', 'SP500'), ('sox', 'SOX')]:
        df = pd.read_sql("""
            SELECT period AS trade_date, value
            FROM alpha_lab.macro_indicators
            WHERE indicator = %s AND freq = 'D'
            ORDER BY period
        """, conn, params=(ind_name,), parse_dates=['trade_date'])

        if df.empty:
            print(f"  ❌ {ind_name} 데이터 없음 (macro_indicators)")
            continue

        df = df.set_index('trade_date').sort_index()
        price = df['value'].astype(float)
        print(f"  {symbol} 가격 로드: {len(price)}건 ({price.index[0].date()} ~ {price.index[-1].date()})")
        calc_and_store(conn, price, symbol=symbol)


def main():
    print("=" * 50)
    print("Technical 지표 계산/저장 시작")
    print("=" * 50)

    conn = psycopg2.connect(DATABASE_URL)
    migrate_table(conn)

    print("\n[KOSPI MA/MACD/RSI]")
    collect_kospi(conn)

    print("\n[S&P500 / SOX MA/MACD/RSI]")
    collect_global(conn)

    conn.close()
    print("\n완료!")


if __name__ == '__main__':
    main()
