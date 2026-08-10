"""
analysis/false_alarm_diagnosis.py

False Alarm 원인 진단: True Alarm vs False Alarm의 feature 분포 비교.
경보 onset(breadth 천장신호) 각각에서 feature 캡처 → TA/FA 라벨 → 어떤 feature가 둘을 가르나.
분리되는 feature 발견 시 → HMM emission 추가 후보.

feature: breadth, Δbreadth, newlow, Δnewlow, mom_decel, sox_rs (+ 추가 진단용 몇 개)
사용: .venv/bin/python analysis/false_alarm_diagnosis.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
WIN = 6


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
        dd6 = np.nan
        if i + 6 < len(mends):
            path = pd.concat([pd.Series([kospi.loc[e]], index=[e]), kospi[(kospi.index > e) & (kospi.index <= mends[i+6])]])
            dd6 = (path / path.cummax() - 1).min() * 100
        rows.append({"ym": pd.Period(e, freq="M").strftime("%Y-%m"),
                     "breadth": asof(breadth, e), "br_chg": asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
                     "newlow": asof(newlow, e), "nl_chg": asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
                     "mom_decel": pct(kospi, e, 30) - pct(kospi, e, 180),
                     "sox_rs": pct(sox, e, 60) - pct(kospi, e, 60),
                     "kospi_mom3m": pct(kospi, e, 90), "dist_ma": asof(kospi, e) / kospi[kospi.index <= e].iloc[-150:].mean() - 1,
                     "dd6": dd6})
    df = pd.DataFrame(rows).reset_index(drop=True); n = len(df)
    df["bear_zone"] = (df["dd6"] <= -15).astype(int)
    df["event"] = ((df["bear_zone"] == 1) & (df["bear_zone"].shift(1).fillna(0) == 0)).astype(int)
    events = df.index[df["event"] == 1].tolist()

    # 천장 신호 onset (top_score>=0.6 OR sudden), 2000~ no-fit 분위수
    def ep(arr, t): hh = arr[:t]; return (hh < arr[t]).mean() if t > 0 else .5
    sig = np.zeros(n, bool)
    for t in range(12, n):
        top = np.mean([1 - ep(df["breadth"].values, t), ep(np.nan_to_num(df["mom_decel"].values), t)])
        sudden = ep(np.nan_to_num(df["br_chg"].values), t) <= 0.15
        sig[t] = (top >= 0.6) or sudden
    onsets = [t for t in range(1, n) if sig[t] and not sig[t-1]]
    labels = []
    for s in onsets:
        labels.append(1 if any(abs(s - ev) <= WIN for ev in events) else 0)  # 1=TA, 0=FA
    labels = np.array(labels)
    print(f"경보 onset {len(onsets)}개: True Alarm {labels.sum()}, False Alarm {(labels==0).sum()}\n")

    fcols = ["breadth", "br_chg", "newlow", "nl_chg", "mom_decel", "sox_rs", "kospi_mom3m", "dist_ma"]
    od = df.iloc[onsets][fcols].reset_index(drop=True)
    print(f"  {'feature':12} {'TA평균':>9} {'FA평균':>9} {'분리|d|':>8} {'AUC':>6}")
    print("  " + "-" * 50)
    res = []
    for c in fcols:
        ta = od[c][labels == 1]; fa = od[c][labels == 0]
        pooled = np.sqrt((ta.var() + fa.var()) / 2) or 1
        dsep = abs(ta.mean() - fa.mean()) / pooled
        try:
            auc = roc_auc_score(labels, od[c])  # TA vs FA 구분
            auc = max(auc, 1 - auc)
        except Exception:
            auc = np.nan
        res.append((c, ta.mean(), fa.mean(), dsep, auc))
    for c, tam, fam, dsep, auc in sorted(res, key=lambda x: -x[3]):
        print(f"  {c:12} {tam:>+9.3f} {fam:>+9.3f} {dsep:>8.2f} {auc:>6.3f}")
    print("\n  분리|d|·AUC 높은 feature = TA/FA 구분자 → HMM emission 추가 후보.")
    print("  (특히 FA에서만 특이한 feature가 있으면 그게 '가짜 경보' 표식)")


if __name__ == "__main__":
    main()
