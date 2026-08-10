"""
analysis/regime_improve.py

월별 예측력 끌어올리기 최종 시도: 최선 타겟(dd_3m 낙폭)에 *아직 제대로 안 쓴 신호*를 추가.
기준선(가격+US 모멘텀) 대비 marginal 측정.

추가 신호:
  - 12개월 모멘텀 (TSMOM) — 가장 검증된 추세 시그널 (우린 6개월까지만 썼음)
  - 52주 고점대비 위치 (추세 품질)
  - 상승일 비율(63d), MA150 기울기 (추세 지속성/방향)
  - 계절성 (월 sin/cos)
  - 변동성 3m (국면)

타겟: dd_3m(다음3개월 최대낙폭<-10%), bear_1m(다음달<-3%).
walk-forward LR+GBM. 같은 표본·기준선 대비 Δ.

입력:  Railway PG (kospi/sox/sp500)
사용:  .venv/bin/python analysis/regime_improve.py
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
    kret = kospi.pct_change()

    ym = pd.PeriodIndex(kospi.index, freq="M"); lastd = {}
    for d, p in zip(kospi.index, ym): lastd[p] = d
    mends = [lastd[p] for p in sorted(lastd)]

    rows = []
    for i in range(len(mends) - 1):
        e = mends[i]
        if e < START or i + 3 >= len(mends):
            continue
        s = kospi[kospi.index <= e]
        ma150 = s.iloc[-150:].mean()
        ma150_prev = s.iloc[-171:-21].mean() if len(s) >= 171 else np.nan
        r63 = kret[kret.index <= e].iloc[-63:]
        mo = pd.Period(e, freq="M").month
        _hi = s.iloc[-252:].max(); _lo = s.iloc[-252:].min()
        pos12 = (asof(kospi, e) - _lo) / (_hi - _lo) if _hi > _lo else np.nan  # 12개월 가격위치 0~1
        feat = {
            # 기준선
            "kospi_mom1m": pct(kospi, e, 30), "kospi_mom3m": pct(kospi, e, 90),
            "kospi_mom6m": pct(kospi, e, 180), "kospi_magap": asof(kospi, e) / ma150 - 1,
            "sox_mom1m": pct(sox, e, 30), "sox_mom3m": pct(sox, e, 90),
            "spx_mom1m": pct(spx, e, 30), "spx_mom3m": pct(spx, e, 90),
            # 신규
            "kospi_mom12m": pct(kospi, e, 365), "spx_mom12m": pct(spx, e, 365),
            "dist_52w": asof(kospi, e) / s.iloc[-252:].max() - 1,
            "updays_63": float((r63 > 0).mean()),
            "ma_slope": (ma150 / ma150_prev - 1) if ma150_prev else np.nan,
            "vol_3m": kret[kret.index <= e].iloc[-63:].std() * np.sqrt(63),
            "mon_sin": np.sin(2 * np.pi * mo / 12), "mon_cos": np.cos(2 * np.pi * mo / 12),
            "pos12": pos12,
        }
        r3 = kospi.loc[mends[i + 3]] / kospi.loc[e] - 1
        r1 = kospi.loc[mends[i + 1]] / kospi.loc[e] - 1
        path = pd.concat([pd.Series([kospi.loc[e]], index=[e]),
                          kospi[(kospi.index > e) & (kospi.index <= mends[i + 3])]])
        feat["dd_3m"] = int((path / path.cummax() - 1).min() < -0.10)
        feat["bear_1m"] = int(r1 < -0.03)
        rows.append(feat)

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    base = ["kospi_mom1m", "kospi_mom3m", "kospi_mom6m", "kospi_magap",
            "sox_mom1m", "sox_mom3m", "spx_mom1m", "spx_mom3m"]
    new = ["kospi_mom12m", "spx_mom12m", "dist_52w", "updays_63", "ma_slope", "vol_3m", "mon_sin", "mon_cos", "pos12"]
    print(f"표본 {len(df)}개월\n")

    def wf(cols, label, hz):
        X, y = df[cols].values, df[label].values
        lp = np.full(len(df), np.nan); gp = np.full(len(df), np.nan)
        for i in range(MIN_TRAIN, len(df)):
            cut = i - hz
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
        ens = np.nanmean([lp, gp], axis=0)
        r = {}
        for nm, p in [("LR", lp), ("GBM", gp), ("앙상블", ens)]:
            m = ~np.isnan(p)
            r[nm] = roc_auc_score(y[m], p[m]) if m.sum() > 20 and len(set(y[m])) > 1 else np.nan
        return r

    for label, hz, desc in [("dd_3m", 3, "3개월 낙폭<-10%"), ("bear_1m", 1, "다음달<-3%")]:
        print(f"=== {desc} (양성 {df[label].sum()}/{len(df)}) ===")
        b = wf(base, label, hz); f = wf(base + new, label, hz)
        print(f"  기준선(가격+US,{len(base)})   LR {b['LR']:.3f}  GBM {b['GBM']:.3f}  앙상블 {b['앙상블']:.3f}")
        print(f"  +신규신호({len(base+new)})       LR {f['LR']:.3f}  GBM {f['GBM']:.3f}  앙상블 {f['앙상블']:.3f}")
        print(f"  → Δ(best) {max(f.values())-max(b.values()):+.3f}\n")
    print("판정: Δ +0.03↑면 신규신호 효과. best AUC가 0.6 넘으면 의미있는 예측력.")


if __name__ == "__main__":
    main()
