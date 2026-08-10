"""
scripts/collect_kospi_stocks_yf.py

KOSPI 개별종목 일별 종가 2000~2016 수집 (yfinance, "{종목코드}.KS").
목적: breadth·newlow를 2000년까지 확장 → HMM 장기 학습/검증.
(pykrx는 2015~만 가능해 폐기. yfinance는 2000~ 됨 — 검증완료.)

유니버스: alpha_lab.daily_price의 종목코드(2017~ 보유분)에 .KS 붙여 과거 수집.
  (생존편향: 현재상장 종목만 → 과거 상폐종목 누락. 1차 근사로 충분, 후에 보강 가능)
배치 다운로드 + 재개 가능(이미 받은 종목 skip).

출력: analysis/kospi_stocks_2000_2016.csv  (dt, stock_code, close) — daily_price와 동일 스키마.
사용: .venv/bin/python scripts/collect_kospi_stocks_yf.py
주의: 종목 많아 수십분. 끊기면 다시 실행(이어받음).
"""
import os
import sys
import time
from pathlib import Path
import pandas as pd
import psycopg2
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
OUT = Path(__file__).parent.parent / "analysis" / "kospi_stocks_2000_2016.csv"
START, END = "1999-12-01", "2017-01-15"   # 2017 초 약간 겹치게(검증용)
BATCH = 80


def get_universe():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    df = pd.read_sql("SELECT DISTINCT stock_code FROM alpha_lab.daily_price ORDER BY stock_code", conn)
    conn.close()
    # 6자리 숫자 코드만 (KOSPI/KOSDAQ 보통주)
    return [c for c in df["stock_code"].astype(str) if c.isdigit() and len(c) == 6]


def done_tickers():
    if not OUT.exists():
        return set()
    try:
        return set(pd.read_csv(OUT, usecols=["stock_code"], dtype={"stock_code": str})["stock_code"].unique())
    except Exception:
        return set()


def main():
    codes = get_universe()
    have = done_tickers()
    todo = [c for c in codes if c not in have]
    print(f"유니버스 {len(codes)}종목, 이미받음 {len(have)}, 남음 {len(todo)}", flush=True)
    header = not OUT.exists()

    for bi in range(0, len(todo), BATCH):
        batch = todo[bi:bi + BATCH]
        tickers = [f"{c}.KS" for c in batch]
        try:
            data = yf.download(tickers, start=START, end=END, progress=False, auto_adjust=False, group_by="ticker", threads=True)
        except Exception as e:
            print(f"  [batch {bi} fail] {str(e)[:60]}", flush=True); continue
        rows = []
        for c in batch:
            tk = f"{c}.KS"
            try:
                if len(tickers) == 1:
                    close = data["Close"]
                else:
                    close = data[tk]["Close"]
                close = close.dropna()
                for dt, v in close.items():
                    if v and v > 0:
                        rows.append((pd.Timestamp(dt).date().isoformat(), c, float(v)))
            except Exception:
                continue
        if rows:
            pd.DataFrame(rows, columns=["dt", "stock_code", "close"]).to_csv(OUT, mode="a", header=header, index=False)
            header = False
        print(f"  진행 {min(bi+BATCH,len(todo))}/{len(todo)}  (+{len(rows)}행)", flush=True)
        time.sleep(1.0)

    df = pd.read_csv(OUT, dtype={"stock_code": str})
    print(f"\n완료: {len(df):,}행, 종목 {df['stock_code'].nunique()}개, {df['dt'].min()}~{df['dt'].max()} → {OUT}")
    print("다음: breadth/newlow를 (이 파일 2000~2016 + daily_price 2017~)로 2000~ 계산 → HMM 장기.")


if __name__ == "__main__":
    main()
