"""
analysis/fwd_return_predict.py

dd-패러독스 탈출 시도: 타겟을 "3개월 낙폭"이 아니라 "6/12개월 net 수익<0"(지속 약세)로.
짧은 낙폭은 반등하지만 장기 net 하락은 안 돌아옴 + 모멘텀이 가장 잘 맞히는 horizon.
→ 예측 가능 & 돈 되는(패러독스 없는) 지점인지 확인.

핵심 검증: AUC + "예측 약세 tercile의 실제 향후수익이 *낮은가*" (패러독스 탈출 여부).

타겟: fwd6m_down(6개월<0), fwd6m_loss(6개월<-5%), fwd12m_down(12개월<0)
feature: 모멘텀 + 추세상태. walk-forward LR+GBM, embargo=horizon.
사용: .venv/bin/python analysis/fwd_return_predict.py
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
        if e < START or i + 12 >= len(mends):   # 12개월 타겟 정의되는 월만 (표본 동일)
            continue
        s = kospi[kospi.index <= e]; ma150 = s.iloc[-150:].mean()
        ma_prev = s.iloc[-171:-21].mean() if len(s) >= 171 else np.nan
        hi, lo = s.iloc[-252:].max(), s.iloc[-252:].min()
        feat = {
            "k_mom1m": pct(kospi, e, 30), "k_mom3m": pct(kospi, e, 90),
            "k_mom6m": pct(kospi, e, 180), "k_mom12m": pct(kospi, e, 365),
            "k_magap": asof(kospi, e) / ma150 - 1,
            "ma_slope": (ma150 / ma_prev - 1) if ma_prev else np.nan,
            "pos12": (asof(kospi, e) - lo) / (hi - lo) if hi > lo else np.nan,
            "dist52w": asof(kospi, e) / hi - 1,
            "sox_mom3m": pct(sox, e, 90), "sox_mom12m": pct(sox, e, 365),
            "spx_mom6m": pct(spx, e, 180), "spx_mom12m": pct(spx, e, 365),
        }
        r6 = kospi.loc[mends[i + 6]] / kospi.loc[e] - 1
        r12 = kospi.loc[mends[i + 12]] / kospi.loc[e] - 1
        feat["fwd6m_down"] = int(r6 < 0)
        feat["fwd6m_loss"] = int(r6 < -0.05)
        feat["fwd12m_down"] = int(r12 < 0)
        feat["r6"] = r6 * 100; feat["r12"] = r12 * 100
        rows.append(feat)

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    fcols = ["k_mom1m", "k_mom3m", "k_mom6m", "k_mom12m", "k_magap", "ma_slope",
             "pos12", "dist52w", "sox_mom3m", "sox_mom12m", "spx_mom6m", "spx_mom12m"]
    X = df[fcols].values
    print(f"표본 {len(df)}개월 ({df.index[0]})\n")

    def wf(label, hz):
        y = df[label].values
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
        return lp, gp

    for label, hz, rcol, desc in [("fwd6m_down", 6, "r6", "6개월 net<0"),
                                   ("fwd6m_loss", 6, "r6", "6개월 net<-5%"),
                                   ("fwd12m_down", 12, "r12", "12개월 net<0")]:
        lp, gp = wf(label, hz)
        df["_p"] = np.nanmean([lp, gp], axis=0)
        ev = df.dropna(subset=["_p"])
        y = ev[label]
        auc = roc_auc_score(y, ev["_p"]) if y.nunique() > 1 else float("nan")
        print(f"=== {desc} (양성 {df[label].sum()}/{len(df)}) ===")
        for nm, p in [("LR", lp), ("GBM", gp)]:
            m = ~np.isnan(p)
            a = roc_auc_score(df[label].values[m], p[m]) if m.sum() > 20 else float("nan")
            print(f"    {nm} AUC {a:.3f}", end="  ")
        print(f"앙상블 AUC {auc:.3f}")
        # 패러독스 탈출 검증: 예측 약세 tercile의 실제 수익
        ev = ev.copy(); ev["tier"] = pd.qcut(ev["_p"], 3, labels=["저(강세예측)", "중", "고(약세예측)"], duplicates="drop")
        print(f"    예측위험별 실제 {rcol} 평균:")
        for t in ev["tier"].cat.categories:
            sub = ev[ev["tier"] == t]
            print(f"      {t}: {sub[rcol].mean():+.1f}%  (실제 약세율 {sub[label].mean()*100:.0f}%)")
        print()

    print("핵심: '고(약세예측)' tercile의 실제 수익이 '저'보다 *낮으면* → 패러독스 탈출, 쓸 수 있는 예측 ✓")


if __name__ == "__main__":
    main()
