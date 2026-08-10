"""
analysis/regime_improve_detail.py

dd_3m(3개월 낙폭<-10%) + 가격+US+신규신호 LR(C=0.3) 상세 분석.
  - walk-forward OOS AUC / recall / precision
  - 어떤 feature가 작동하나 (표준화 계수, descriptive)
  - 예측 위험도별 실제 낙폭률 + 향후 3개월 수익 (tilt 실효성)
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")
MIN_TRAIN, START = 48, pd.Timestamp("2000-07-01")


def db_daily(conn, ind):
    df = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators "
                     "WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
    df["dt"] = pd.to_datetime(df["dt"])
    return df.dropna().drop_duplicates("dt").set_index("dt")["v"].sort_index()


def asof(s, e):
    sub = s[s.index <= e]; return sub.iloc[-1] if len(sub) else np.nan


def pct(s, e, d):
    cur = asof(s, e); past = s[s.index <= e - timedelta(days=d)]
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
        s = kospi[kospi.index <= e]; ma150 = s.iloc[-150:].mean()
        ma_prev = s.iloc[-171:-21].mean() if len(s) >= 171 else np.nan
        r63 = kret[kret.index <= e].iloc[-63:]; mo = pd.Period(e, freq="M").month
        hi, lo = s.iloc[-252:].max(), s.iloc[-252:].min()
        feat = {
            "kospi_mom1m": pct(kospi, e, 30), "kospi_mom3m": pct(kospi, e, 90),
            "kospi_mom6m": pct(kospi, e, 180), "kospi_magap": asof(kospi, e) / ma150 - 1,
            "sox_mom1m": pct(sox, e, 30), "sox_mom3m": pct(sox, e, 90),
            "spx_mom1m": pct(spx, e, 30), "spx_mom3m": pct(spx, e, 90),
            "kospi_mom12m": pct(kospi, e, 365), "spx_mom12m": pct(spx, e, 365),
            "dist_52w": asof(kospi, e) / hi - 1, "updays_63": float((r63 > 0).mean()),
            "ma_slope": (ma150 / ma_prev - 1) if ma_prev else np.nan,
            "vol_3m": r63.std() * np.sqrt(63),
            "mon_sin": np.sin(2 * np.pi * mo / 12), "mon_cos": np.cos(2 * np.pi * mo / 12),
            "pos12": (asof(kospi, e) - lo) / (hi - lo) if hi > lo else np.nan,
        }
        path = pd.concat([pd.Series([kospi.loc[e]], index=[e]),
                          kospi[(kospi.index > e) & (kospi.index <= mends[i + 3])]])
        feat["dd_3m"] = int((path / path.cummax() - 1).min() < -0.10)
        feat["fwd3m"] = (kospi.loc[mends[i + 3]] / kospi.loc[e] - 1) * 100
        rows.append(feat)

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    cols = ["kospi_mom1m", "kospi_mom3m", "kospi_mom6m", "kospi_magap", "sox_mom1m", "sox_mom3m",
            "spx_mom1m", "spx_mom3m", "kospi_mom12m", "spx_mom12m", "dist_52w", "updays_63",
            "ma_slope", "vol_3m", "mon_sin", "mon_cos", "pos12"]
    X, y = df[cols].values, df["dd_3m"].values
    print(f"표본 {len(df)}개월, dd_3m 양성 {y.sum()} ({y.mean()*100:.0f}%)\n")

    # walk-forward OOS
    prob = np.full(len(df), np.nan)
    for i in range(MIN_TRAIN, len(df)):
        cut = i - 3
        if cut < MIN_TRAIN // 2: continue
        ytr = y[:cut]
        if ytr.sum() < 8 or (len(ytr) - ytr.sum()) < 8: continue
        sc = StandardScaler().fit(X[:cut])
        lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000)
        lr.fit(sc.transform(X[:cut]), ytr)
        prob[i] = lr.predict_proba(sc.transform(X[i:i+1]))[0, 1]
    df["prob"] = prob
    ev = df.dropna(subset=["prob"])
    auc = roc_auc_score(ev["dd_3m"], ev["prob"])
    print(f"[OOS] AUC={auc:.3f}  (평가 {len(ev)}개월)")
    for th in [0.5, 0.6]:
        p = (ev["prob"] >= th).astype(int)
        tp = ((p == 1) & (ev["dd_3m"] == 1)).sum(); fp = ((p == 1) & (ev["dd_3m"] == 0)).sum()
        fn = ((p == 0) & (ev["dd_3m"] == 1)).sum()
        print(f"  임계 {th}: Recall {tp/max(tp+fn,1)*100:.0f}%  Precision {tp/max(tp+fp,1)*100:.0f}%  (경보 {tp+fp}회)")

    # 위험도 tercile별 실제 낙폭률 + 향후 3개월 수익
    ev = ev.copy(); ev["tier"] = pd.qcut(ev["prob"], 3, labels=["저위험", "중", "고위험"])
    print("\n[예측 위험도별 실제 결과]")
    for t in ["저위험", "중", "고위험"]:
        sub = ev[ev["tier"] == t]
        print(f"  {t}: 실제 낙폭발생 {sub['dd_3m'].mean()*100:.0f}%  향후3개월 평균수익 {sub['fwd3m'].mean():+.1f}%  ({len(sub)}개월)")

    # 계수 (전체 fit, descriptive — 어떤 feature가 낙폭 위험 가리키나)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000).fit(sc.transform(X), y)
    coef = pd.Series(lr.coef_[0], index=cols).sort_values()
    print("\n[표준화 계수] (음=낙폭위험↓, 양=낙폭위험↑, descriptive)")
    for k, v in pd.concat([coef.head(5), coef.tail(5)]).items():
        print(f"  {k:14} {v:+.2f}")


if __name__ == "__main__":
    main()
