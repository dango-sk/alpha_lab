"""
analysis/test_signals_added.py

패러독스-탈출 신호(mom_decel, sox_div, riskoff)를 추가해 월별 방향 예측이
기준선(가격+US 모멘텀) 대비 얼마나 오르는지 + 패러독스 없는지 확인.

타겟: 향후 1/3/6개월 수익>0 (상승). walk-forward LR+GBM.
기준선 vs +신규3 의 AUC·정확도 + 예측 tercile별 실제 수익(패러독스 체크).
사용: .venv/bin/python analysis/test_signals_added.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")
MIN_TRAIN, START = 48, pd.Timestamp("2004-07-01")


def db(conn, ind):
    df = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators "
                     "WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
    df["dt"] = pd.to_datetime(df["dt"])
    return df.dropna().drop_duplicates("dt").set_index("dt")["v"].sort_index()


def asof(s, e):
    sub = s[s.index <= e]; return sub.iloc[-1] if len(sub) else np.nan


def pct(s, e, d):
    cur = asof(s, e); past = s[s.index <= e - timedelta(days=d)]
    return (cur / past.iloc[-1] - 1) if len(past) and past.iloc[-1] else np.nan


def chg(s, e, d):
    cur = asof(s, e); past = s[s.index <= e - timedelta(days=d)]
    return (cur - past.iloc[-1]) if len(past) else np.nan


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    kospi = db(conn, "kospi"); sox = db(conn, "sox"); spx = db(conn, "sp500")
    dxy = db(conn, "dxy"); us10y = db(conn, "us10y")
    conn.close()
    ym = pd.PeriodIndex(kospi.index, freq="M"); lastd = {}
    for d, p in zip(kospi.index, ym): lastd[p] = d
    mends = [lastd[p] for p in sorted(lastd)]

    rows = []
    for i in range(len(mends) - 1):
        e = mends[i]
        if e < START or i + 6 >= len(mends):
            continue
        s = kospi[kospi.index <= e]; ma150 = s.iloc[-150:].mean()
        feat = {
            # 기준선 (가격+US)
            "k_mom1m": pct(kospi, e, 30), "k_mom3m": pct(kospi, e, 90),
            "k_mom6m": pct(kospi, e, 180), "k_mom12m": pct(kospi, e, 365),
            "k_magap": asof(kospi, e) / ma150 - 1,
            "sox_mom3m": pct(sox, e, 90), "spx_mom6m": pct(spx, e, 180),
            # 신규 3 (패러독스 탈출)
            "mom_decel": pct(kospi, e, 30) - pct(kospi, e, 180),
            "sox_div": pct(sox, e, 30) - pct(kospi, e, 30),
            "riskoff": pct(dxy, e, 30) - chg(us10y, e, 30) / 100,
        }
        feat["r1"] = (kospi.loc[mends[i + 1]] / kospi.loc[e] - 1) * 100
        feat["r3"] = (kospi.loc[mends[i + 3]] / kospi.loc[e] - 1) * 100
        feat["r6"] = (kospi.loc[mends[i + 6]] / kospi.loc[e] - 1) * 100
        rows.append(feat)

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    base = ["k_mom1m", "k_mom3m", "k_mom6m", "k_mom12m", "k_magap", "sox_mom3m", "spx_mom6m"]
    new = base + ["mom_decel", "sox_div", "riskoff"]
    print(f"표본 {len(df)}개월\n")

    def wf(cols, label, hz):
        X = df[cols].values; y = (df[label] > 0).astype(int).values
        lp = np.full(len(df), np.nan); gp = np.full(len(df), np.nan)
        for i in range(MIN_TRAIN, len(df)):
            cut = i - hz
            if cut < MIN_TRAIN // 2: continue
            ytr = y[:cut]
            if ytr.sum() < 8 or (len(ytr) - ytr.sum()) < 8: continue
            sc = StandardScaler().fit(X[:cut])
            lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000)
            lr.fit(sc.transform(X[:cut]), ytr); lp[i] = lr.predict_proba(sc.transform(X[i:i+1]))[0, 1]
            gb = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150, random_state=42)
            gb.fit(X[:cut], ytr); gp[i] = gb.predict_proba(X[i:i+1])[0, 1]
        ens = np.nanmean([lp, gp], axis=0)
        m = ~np.isnan(ens)
        auc = roc_auc_score(y[m], ens[m]) if m.sum() > 20 and len(set(y[m])) > 1 else np.nan
        acc = ((ens[m] >= 0.5).astype(int) == y[m]).mean()
        return auc, acc, ens

    for label, hz in [("r1", 1), ("r3", 3), ("r6", 6)]:
        ab, accb, _ = wf(base, label, hz)
        an, accn, ens = wf(new, label, hz)
        print(f"=== 향후 {label[1:]}개월 상승 예측 ===")
        print(f"  기준선(가격+US,7)   AUC {ab:.3f}  정확도 {accb*100:.0f}%")
        print(f"  +신규3(10)          AUC {an:.3f}  정확도 {accn*100:.0f}%   ΔAUC {an-ab:+.3f}")
        # 패러독스 체크: 예측 강세확률 tercile별 실제 수익
        d = df.copy(); d["p"] = ens; d = d.dropna(subset=["p"])
        d["t"] = pd.qcut(d["p"], 3, labels=["약세예측", "중", "강세예측"], duplicates="drop")
        lo = d[d["t"] == "약세예측"][label].mean(); hi = d[d["t"] == "강세예측"][label].mean()
        print(f"  예측강세 tercile 실제 {label}={hi:+.1f}%  vs 예측약세={lo:+.1f}%  "
              f"{'✓방향맞음' if hi > lo + 1 else '✗역전'}\n")
    print("판정: ΔAUC↑ & '강세예측>약세예측 실제수익'이면 신규신호가 예측력 개선+패러독스 없음.")


if __name__ == "__main__":
    main()
