"""
SP500 (^GSPC) / SOX (^SOX) backfill into alpha_lab.macro_indicators.

용도:
  - regime_agent_multimodel에서 쓰는 SP500/SOX MA200이 2018-10-16부터만 계산되는
    문제(close lookback 부족) 해결: 2016~2017년 데이터 backfill.
  - 동일 경로로 2026-03-25 이후 stale 데이터도 갱신.

실행: .venv/bin/python scripts/backfill_global_indices.py
이후: .venv/bin/python scripts/legacy/collect_technical.py
"""
import os
import sys
from datetime import date, timedelta

import psycopg2
from psycopg2.extras import execute_values
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

TICKERS = [
    ("sp500", "^GSPC"),
    ("sox", "^SOX"),
    ("kospi", "^KS11"),     # KOSPI 종합지수 1996-12~ (레짐 장기학습)
    # ── 외생 리스크 feature (월별 레짐 예측률 개선용) ──
    # vix·usd_krw는 *기존 지표 연장*(이름 동일 → 중복 X, ON CONFLICT로 갱신)
    ("vix", "^VIX"),        # 공포지수: 2017~ → 1990~ 연장
    ("usd_krw", "USDKRW=X"),# 원/달러: 2017~ → 2003~ 연장
    # us10y·dxy는 신규 (한국 bond_10y와 다른 미국물/달러인덱스)
    ("us10y", "^TNX"),      # 미국 10년물 1990~
    ("dxy", "DX-Y.NYB"),    # 달러인덱스 1990~
]

# 주의: 기본 START를 1996-12로 내림 → KOSPI 장기 히스토리 확보.
# sp500/sox도 같이 1996부터 재적재되지만 upsert라 무해(MA200 lookback에도 이득).
# 특정 기간만 원하면 BACKFILL_START 환경변수로 덮어쓰기.
START = os.environ.get("BACKFILL_START", "1996-12-01")
END = os.environ.get("BACKFILL_END", (date.today() + timedelta(days=1)).isoformat())


def fetch(ticker: str):
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
    if df.empty:
        return []
    if hasattr(df.columns, "get_level_values"):
        close = df["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
    else:
        close = df["Close"]
    rows = []
    for ts, val in close.items():
        if val is None or (val != val):
            continue
        rows.append((ts.date().isoformat(), float(val)))
    return rows


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    total_inserted = 0
    for indicator, ticker in TICKERS:
        cur.execute(
            "SELECT COUNT(*), MIN(period), MAX(period) FROM alpha_lab.macro_indicators WHERE indicator=%s AND freq='D'",
            (indicator,),
        )
        before = cur.fetchone()
        print(f"[{indicator}] before: cnt={before[0]}, {before[1]} ~ {before[2]}")

        rows = fetch(ticker)
        print(f"[{indicator}] yfinance fetched: {len(rows)} rows ({rows[0][0] if rows else 'N/A'} ~ {rows[-1][0] if rows else 'N/A'})")
        if not rows:
            print(f"[{indicator}] ⚠️  no rows fetched, skipping")
            continue

        # 배치 upsert (행마다 왕복 X) — execute_values로 한 번에
        execute_values(
            cur,
            """
            INSERT INTO alpha_lab.macro_indicators (indicator, period, freq, value, updated_at)
            VALUES %s
            ON CONFLICT (indicator, period) DO UPDATE
              SET value = EXCLUDED.value,
                  updated_at = NOW()
              WHERE alpha_lab.macro_indicators.value IS DISTINCT FROM EXCLUDED.value
            """,
            [(indicator, period, value) for period, value in rows],
            template="(%s, %s, 'D', %s, NOW())",
            page_size=2000,
        )
        conn.commit()

        cur.execute(
            "SELECT COUNT(*), MIN(period), MAX(period) FROM alpha_lab.macro_indicators WHERE indicator=%s AND freq='D'",
            (indicator,),
        )
        after = cur.fetchone()
        net_new = after[0] - before[0]
        total_inserted += max(net_new, 0)
        print(f"[{indicator}] after:  cnt={after[0]}, {after[1]} ~ {after[2]}  (net_new={net_new})")

    print(f"\nDone. total net_new rows = {total_inserted}")
    conn.close()


if __name__ == "__main__":
    main()
