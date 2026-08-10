"""
analysis/false_alarm_cases.py

False Alarm / True Alarm 경보를 *개별 사례*로 출력 (평균이 숨기는 패턴 찾기).
각 경보월의 feature + 당시 상황 + 이후 결과(fwd 2m/6m, dd6) 나란히.
사용: .venv/bin/python analysis/false_alarm_cases.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
WIN = 6
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)


def main():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    def md(ind):
        x = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
        x["dt"] = pd.to_datetime(x["dt"]); return x.set_index("dt")["v"].sort_index()
    kospi = md("kospi"); sox = md("sox"); conn.close()
    d["dt"] = pd.to_datetime(d["dt"])
    wide = pd.concat([h, d[["dt", "stock_code", "close"]]], ignore_index=True).drop_duplicates(["dt", "stock_code"]).pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    ma = wide.rolling(120, min_periods=60).mean(); val = wide.notna() & ma.notna()
    breadth = ((wide > ma) & val).sum(axis=1) / val.sum(axis=1).clip(lower=1); breadth = breadth[val.sum(axis=1) > 100]
    rmn = wide.rolling(252, min_periods=120).min(); newlow = ((wide <= rmn) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1); newlow = newlow.reindex(breadth.index)
    ym = pd.PeriodIndex(kospi.index, freq="M"); last = {}
    for dt_, p in zip(kospi.index, ym): last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1) if len(p) and p.iloc[-1] else np.nan
    rows = []
    for i in range(len(mends) - 1):
        e = mends[i]
        f2 = (kospi.loc[mends[i+2]] / kospi.loc[e] - 1) * 100 if i + 2 < len(mends) else np.nan
        dd6 = np.nan
        if i + 6 < len(mends):
            path = pd.concat([pd.Series([kospi.loc[e]], index=[e]), kospi[(kospi.index > e) & (kospi.index <= mends[i+6])]])
            dd6 = (path / path.cummax() - 1).min() * 100
        rows.append({"ym": pd.Period(e, freq="M").strftime("%Y-%m"),
                     "breadth": asof(breadth, e), "Δbr": asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
                     "newlow": asof(newlow, e), "Δnl": asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
                     "mom_dec": pct(kospi, e, 30) - pct(kospi, e, 180),
                     "sox_rs": pct(sox, e, 60) - pct(kospi, e, 60),
                     "dist_ma": asof(kospi, e) / kospi[kospi.index <= e].iloc[-150:].mean() - 1,
                     "fwd2m": f2, "dd6": dd6})
    df = pd.DataFrame(rows).reset_index(drop=True); n = len(df)
    df["event"] = (((df["dd6"] <= -15) & (df["dd6"].shift(1).fillna(0) > -15))).astype(int)
    events = df.index[df["event"] == 1].tolist()
    def ep(arr, t): hh = arr[:t]; return (hh < arr[t]).mean() if t > 0 else .5
    sig = np.zeros(n, bool)
    for t in range(12, n):
        top = np.mean([1 - ep(df["breadth"].values, t), ep(np.nan_to_num(df["mom_dec"].values), t)])
        sig[t] = (top >= 0.6) or (ep(np.nan_to_num(df["Δbr"].values), t) <= 0.15)
    onsets = [t for t in range(1, n) if sig[t] and not sig[t-1]]

    def show(idxs, title):
        sub = df.iloc[idxs].copy()
        for c in ["breadth", "Δbr", "newlow", "Δnl", "mom_dec", "sox_rs", "dist_ma"]:
            sub[c] = sub[c].round(3)
        sub["fwd2m"] = sub["fwd2m"].round(1); sub["dd6"] = sub["dd6"].round(0)
        print(f"\n{'='*100}\n  {title} ({len(idxs)}개)\n{'='*100}")
        print(sub[["ym", "breadth", "Δbr", "newlow", "Δnl", "mom_dec", "sox_rs", "dist_ma", "fwd2m", "dd6"]].to_string(index=False))

    fa = [s for s in onsets if all(abs(s - ev) > WIN for ev in events)]
    ta = [s for s in onsets if any(abs(s - ev) <= WIN for ev in events)]
    show(fa, "FALSE ALARM (전환 없었음) — fwd2m/dd6 보면 실제로 안 빠짐")
    show(ta, "TRUE ALARM (실제 전환) — 대조용")
    print("\n패턴 찾기: FA에서 dist_ma(고점대비)·sox_rs·Δnl 등이 TA와 다른 군집이 보이는지 눈으로 확인.")


if __name__ == "__main__":
    main()
