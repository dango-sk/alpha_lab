"""
analysis/label_compare.py

깨끗한 통제 비교: *같은 표본·같은 feature(가격+US 모멘텀)·같은 모델*,
라벨(예측 대상)만 바꿔 "방향 예측 vs 낙폭 예측" 중 뭐가 더 예측 가능한지.

라벨 4종 (cutoff e 기준 forward):
  up_1m   : 다음 1개월 수익률 > 0      (방향)
  bear_1m : 다음 1개월 수익률 < -3%    (낙폭)
  up_3m   : 다음 3개월 수익률 > 0      (방향)
  dd_3m   : 다음 3개월 최대낙폭 < -10% (낙폭, peak-to-trough)

표본 동일: 4개 라벨 모두 정의되는 월만 사용. feature 동일: 가격+US 9개.
walk-forward LR+GBM, embargo=horizon. AUC 나란히.

입력:  Railway PG (kospi/sox/sp500)
사용:  .venv/bin/python analysis/label_compare.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")
MIN_TRAIN = 48
START = pd.Timestamp("2000-07-01")


def db_daily(conn, ind):
    df = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators "
                     "WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
    df["dt"] = pd.to_datetime(df["dt"])
    return df.dropna().drop_duplicates("dt").set_index("dt")["v"].sort_index()


def asof(s, e):
    sub = s[s.index <= e]
    return sub.iloc[-1] if len(sub) else np.nan


def pct(s, e, days):
    cur = asof(s, e); past = s[s.index <= e - timedelta(days=days)]
    return (cur / past.iloc[-1] - 1) if len(past) and past.iloc[-1] else np.nan


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    kospi = db_daily(conn, "kospi"); sox = db_daily(conn, "sox"); spx = db_daily(conn, "sp500")
    conn.close()

    ym = pd.PeriodIndex(kospi.index, freq="M"); lastd = {}
    for d, p in zip(kospi.index, ym): lastd[p] = d
    mends = [lastd[p] for p in sorted(lastd)]

    rows = []
    for i in range(len(mends) - 1):
        e = mends[i]
        if e < START or i + 3 >= len(mends):   # 3개월 라벨 정의되는 월만 (표본 동일)
            continue
        feat = {
            "kospi_mom1m": pct(kospi, e, 30), "kospi_mom3m": pct(kospi, e, 90),
            "kospi_mom6m": pct(kospi, e, 180),
            "kospi_magap": asof(kospi, e) / kospi[kospi.index <= e].iloc[-150:].mean() - 1,
            "sox_mom1m": pct(sox, e, 30), "sox_mom3m": pct(sox, e, 90),
            "spx_mom1m": pct(spx, e, 30), "spx_mom3m": pct(spx, e, 90),
            "spx_magap": asof(spx, e) / spx[spx.index <= e].iloc[-150:].mean() - 1,
        }
        r1 = kospi.loc[mends[i + 1]] / kospi.loc[e] - 1
        r3 = kospi.loc[mends[i + 3]] / kospi.loc[e] - 1
        path = kospi[(kospi.index > e) & (kospi.index <= mends[i + 3])]
        path = pd.concat([pd.Series([kospi.loc[e]], index=[e]), path])
        dd3 = (path / path.cummax() - 1).min()
        feat["up_1m"] = int(r1 > 0)
        feat["bear_1m"] = int(r1 < -0.03)
        feat["up_3m"] = int(r3 > 0)
        feat["dd_3m"] = int(dd3 < -0.10)
        rows.append(feat)

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    feats = ["kospi_mom1m", "kospi_mom3m", "kospi_mom6m", "kospi_magap",
             "sox_mom1m", "sox_mom3m", "spx_mom1m", "spx_mom3m", "spx_magap"]
    X = df[feats].values
    print(f"표본 {len(df)}개월 ({df.index[0]}~), feature {len(feats)}개 (가격+US 동일)\n")

    def wf(label, horizon):
        y = df[label].values
        lp = np.full(len(df), np.nan); gp = np.full(len(df), np.nan)
        for i in range(MIN_TRAIN, len(df)):
            cut = i - horizon
            if cut < MIN_TRAIN // 2:
                continue
            ytr = y[:cut]
            if ytr.sum() < 8 or (len(ytr) - ytr.sum()) < 8:
                continue
            sc = StandardScaler().fit(X[:cut])
            lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000)
            lr.fit(sc.transform(X[:cut]), ytr); lp[i] = lr.predict_proba(sc.transform(X[i:i+1]))[0, 1]
            gb = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150, random_state=42)
            gb.fit(X[:cut], ytr); gp[i] = gb.predict_proba(X[i:i+1])[0, 1]
        r = {}
        for nm, p in [("LR", lp), ("GBM", gp)]:
            m = ~np.isnan(p)
            r[nm] = roc_auc_score(y[m], p[m]) if m.sum() > 20 and len(set(y[m])) > 1 else np.nan
        return r

    print(f"  {'라벨':14} {'유형':6} {'양성%':>6} {'LR':>7} {'GBM':>7}")
    print("  " + "-" * 46)
    for label, typ, hz in [("up_1m", "방향", 1), ("bear_1m", "낙폭", 1),
                           ("up_3m", "방향", 3), ("dd_3m", "낙폭", 3)]:
        r = wf(label, hz)
        print(f"  {label:14} {typ:6} {df[label].mean()*100:>5.0f}% {r['LR']:>7.3f} {r['GBM']:>7.3f}")
    print("\n  같은 feature·표본·모델. 낙폭(bear/dd) AUC가 방향(up)보다 일관되게 높으면")
    print("  → 게이트를 '방향'에서 '낙폭' 예측으로 바꾸는 게 실제 이득.")


if __name__ == "__main__":
    main()
