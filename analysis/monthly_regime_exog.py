"""
analysis/monthly_regime_exog.py

월별 레짐 예측률 개선: *장기 외생 feature*(VIX·미10년·달러인덱스·USD/KRW) 투입.
가격-only는 ~0.55에서 막힘 → 가격에 없는 글로벌 리스크 신호로 올라가나 확인.

이번엔 외생 feature를 1996/2003~로 연장했으므로 *장기 표본*(~21년, Bear 6~8개).
정답지: ① 다음달 수익<0(약세)  ② 다음달<-3%(큰 약세)
비교:   가격-only vs 가격+외생, walk-forward AUC.

모두 일별 데이터 → lookahead는 cutoff(월말) 이하만 사용으로 차단 (매크로 lag 불필요).

입력:  Railway PG (kospi/sox/sp500/vix/us10y/dxy/usd_krw, freq='D')
사용:  .venv/bin/python analysis/monthly_regime_exog.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import psycopg2
import yfinance as yf
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
START = pd.Timestamp("2004-07-01")   # USD/KRW(2003-12~) warmup 후


def daily(conn, ind):
    df = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators "
                     "WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
    df["dt"] = pd.to_datetime(df["dt"])
    return df.dropna().drop_duplicates("dt").set_index("dt")["v"].sort_index()


def asof(s, e):
    sub = s[s.index <= e]
    return sub.iloc[-1] if len(sub) else np.nan


def chg(s, e, days):
    cur = asof(s, e)
    past = s[s.index <= e - timedelta(days=days)]
    return (cur - past.iloc[-1]) if len(past) and np.isfinite(cur) else np.nan


def pct(s, e, days):
    cur = asof(s, e)
    past = s[s.index <= e - timedelta(days=days)]
    return (cur / past.iloc[-1] - 1) if len(past) and past.iloc[-1] else np.nan


def zscore(s, e, days):
    sub = s[(s.index <= e) & (s.index > e - timedelta(days=days))]
    if len(sub) < 10 or sub.std() == 0:
        return np.nan
    return (sub.iloc[-1] - sub.mean()) / sub.std()


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    kospi = daily(conn, "kospi"); sox = daily(conn, "sox"); spx = daily(conn, "sp500")
    vix = daily(conn, "vix"); us10y = daily(conn, "us10y")
    dxy = daily(conn, "dxy"); fx = daily(conn, "usd_krw")
    conn.close()

    # 장단기 금리차 = 미10년(us10y, DB) − 미3개월(^IRX, yfinance)
    irx = yf.download("^IRX", start="1996-01-01", end="2026-06-25",
                      progress=False, auto_adjust=False)["Close"]
    if hasattr(irx, "columns"):
        irx = irx.iloc[:, 0]
    irx.index = pd.to_datetime(irx.index)
    irx = irx.dropna()
    irx_al = irx.reindex(us10y.index, method="ffill")
    term_spread = (us10y - irx_al).dropna()   # 양수=정상, 음수=역전
    print(f"금리차(us10y-us3m): {term_spread.index.min().date()}~{term_spread.index.max().date()}, "
          f"현재 {term_spread.iloc[-1]:.2f}%p")

    ym = pd.PeriodIndex(kospi.index, freq="M"); lastd = {}
    for d, p in zip(kospi.index, ym): lastd[p] = d
    mends = [lastd[p] for p in sorted(lastd)]

    rows = []
    for i in range(len(mends) - 1):
        e, nxt = mends[i], mends[i + 1]
        if e < START:
            continue
        feat = {
            # 가격 (베이스라인)
            "kospi_mom1m": pct(kospi, e, 30), "kospi_mom3m": pct(kospi, e, 90),
            "kospi_mom6m": pct(kospi, e, 180), "kospi_magap": asof(kospi, e) / kospi[kospi.index <= e].iloc[-150:].mean() - 1,
            "sox_mom1m": pct(sox, e, 30), "sox_mom3m": pct(sox, e, 90),
            "spx_mom1m": pct(spx, e, 30), "spx_mom3m": pct(spx, e, 90),
            "spx_magap": asof(spx, e) / spx[spx.index <= e].iloc[-150:].mean() - 1,
            # 외생
            "vix_level": asof(vix, e), "vix_z3m": zscore(vix, e, 90), "vix_chg1m": chg(vix, e, 30),
            "us10y_level": asof(us10y, e), "us10y_chg1m": chg(us10y, e, 30),
            "dxy_chg1m": pct(dxy, e, 30), "dxy_z3m": zscore(dxy, e, 90),
            "fx_chg1m": pct(fx, e, 30),
            # 선행 후보: 장단기 금리차
            "term_spread": asof(term_spread, e), "term_chg1m": chg(term_spread, e, 30),
        }
        ret = (kospi.loc[nxt] / kospi.loc[e] - 1) * 100
        feat["pred_month"] = pd.Period(nxt, freq="M").strftime("%Y-%m")
        feat["down"] = int(ret < 0)
        feat["bear3"] = int(ret < -3)
        rows.append(feat)

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    price = ["kospi_mom1m", "kospi_mom3m", "kospi_mom6m", "kospi_magap",
             "sox_mom1m", "sox_mom3m", "spx_mom1m", "spx_mom3m", "spx_magap"]
    exog = price + ["vix_level", "vix_z3m", "vix_chg1m", "us10y_level", "us10y_chg1m",
                    "dxy_chg1m", "dxy_z3m", "fx_chg1m"]
    lead = exog + ["term_spread", "term_chg1m"]   # 선행: 금리차 추가
    print(f"패널 {len(df)}개월 ({df['pred_month'].iloc[0]}~{df['pred_month'].iloc[-1]})")

    def wf(cols, label):
        X, y = df[cols].values, df[label].values
        lp = np.full(len(df), np.nan); gp = np.full(len(df), np.nan)
        for i in range(MIN_TRAIN, len(df)):
            ytr = y[:i]
            if ytr.sum() < 8 or (len(ytr) - ytr.sum()) < 8:
                continue
            sc = StandardScaler().fit(X[:i])
            lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000)
            lr.fit(sc.transform(X[:i]), ytr)
            lp[i] = lr.predict_proba(sc.transform(X[i:i+1]))[0, 1]
            gb = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150, random_state=42)
            gb.fit(X[:i], ytr)
            gp[i] = gb.predict_proba(X[i:i+1])[0, 1]
        r = {}
        for nm, p in [("LR", lp), ("GBM", gp)]:
            m = ~np.isnan(p)
            r[nm] = roc_auc_score(y[m], p[m]) if m.sum() > 20 and len(set(y[m])) > 1 else np.nan
        return r

    for label, desc in [("down", "다음달 하락(<0)"), ("bear3", "다음달 <-3%")]:
        print(f"\n=== {desc}  (양성 {df[label].sum()}/{len(df)}) ===")
        b = wf(price, label); f = wf(exog, label); g = wf(lead, label)
        print(f"  가격+US지수({len(price)})   LR {b['LR']:.3f}  GBM {b['GBM']:.3f}")
        print(f"  +외생({len(exog)})         LR {f['LR']:.3f}  GBM {f['GBM']:.3f}")
        print(f"  +금리차({len(lead)})        LR {g['LR']:.3f}  GBM {g['GBM']:.3f}  ← 선행")
        best = max(max(f.values()), max(g.values())) - max(b.values())
        print(f"  → 외생/선행 최대 효과: {best:+.3f}  {'개선 ✓' if best > 0.03 else '미미/없음'}")

    print("\n판정: 어떤 세트든 가격대비 +0.05↑면 의미. 아니면 월별 예측은 역시 벽.")


if __name__ == "__main__":
    main()
