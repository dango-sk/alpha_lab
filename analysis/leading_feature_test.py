"""
analysis/leading_feature_test.py

월별 "큰 낙폭(<-3%)" 예측에 *진짜 다른 정보* feature를 하나씩 추가해 marginal 효과 측정.
(다 쏟으면 과적합 → 개별 추가로 어느 게 실제 도움인지 식별)

기준선: 가격+US지수 (KOSPI/SOX/SP500 모멘텀) — bear3에서 GBM 0.576이었음.
후보(yfinance, 아직 미사용):
  credit     : HYG/LQD 비율 (신용위험, 떨어지면 약세)
  coppergold : HG=F/GC=F 비율 (경기 성장, 떨어지면 약세)
  taiwan     : ^TWII 모멘텀 (아시아 수출·반도체)

공통표본 2007~ (HYG 2007-04 제약). walk-forward LR+GBM AUC, 기준선 대비 delta.

입력:  Railway PG(kospi/sox/sp500) + yfinance(HYG/LQD/HG=F/GC=F/^TWII)
사용:  .venv/bin/python analysis/leading_feature_test.py
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
START = pd.Timestamp("2007-07-01")


def db_daily(conn, ind):
    df = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators "
                     "WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
    df["dt"] = pd.to_datetime(df["dt"])
    return df.dropna().drop_duplicates("dt").set_index("dt")["v"].sort_index()


def yf_daily(tk):
    d = yf.download(tk, start="2000-01-01", end="2026-06-25", progress=False, auto_adjust=False)["Close"]
    if hasattr(d, "columns"):
        d = d.iloc[:, 0]
    d.index = pd.to_datetime(d.index)
    return d.dropna()


def asof(s, e):
    sub = s[s.index <= e]
    return sub.iloc[-1] if len(sub) else np.nan


def pct(s, e, days):
    cur = asof(s, e); past = s[s.index <= e - timedelta(days=days)]
    return (cur / past.iloc[-1] - 1) if len(past) and past.iloc[-1] else np.nan


def zscore(s, e, days):
    sub = s[(s.index <= e) & (s.index > e - timedelta(days=days))]
    if len(sub) < 10 or sub.std() == 0:
        return np.nan
    return (sub.iloc[-1] - sub.mean()) / sub.std()


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    kospi = db_daily(conn, "kospi"); sox = db_daily(conn, "sox"); spx = db_daily(conn, "sp500")
    conn.close()
    print("yfinance: HYG/LQD/구리/금/대만 다운로드...")
    hyg = yf_daily("HYG"); lqd = yf_daily("LQD")
    cop = yf_daily("HG=F"); gold = yf_daily("GC=F"); twii = yf_daily("^TWII")
    credit = (hyg.reindex(hyg.index.union(lqd.index)).ffill() /
              lqd.reindex(hyg.index.union(lqd.index)).ffill()).dropna()
    cg = (cop.reindex(cop.index.union(gold.index)).ffill() /
          gold.reindex(cop.index.union(gold.index)).ffill()).dropna()

    ym = pd.PeriodIndex(kospi.index, freq="M"); lastd = {}
    for d, p in zip(kospi.index, ym): lastd[p] = d
    mends = [lastd[p] for p in sorted(lastd)]

    rows = []
    for i in range(len(mends) - 1):
        e, nxt = mends[i], mends[i + 1]
        if e < START:
            continue
        ret = (kospi.loc[nxt] / kospi.loc[e] - 1) * 100
        rows.append({
            # 기준선 가격+US
            "kospi_mom1m": pct(kospi, e, 30), "kospi_mom3m": pct(kospi, e, 90),
            "kospi_mom6m": pct(kospi, e, 180),
            "sox_mom1m": pct(sox, e, 30), "sox_mom3m": pct(sox, e, 90),
            "spx_mom1m": pct(spx, e, 30), "spx_mom3m": pct(spx, e, 90),
            # 후보
            "credit_chg1m": pct(credit, e, 30), "credit_z3m": zscore(credit, e, 90),
            "cg_chg1m": pct(cg, e, 30), "cg_chg3m": pct(cg, e, 90),
            "twii_mom1m": pct(twii, e, 30), "twii_mom3m": pct(twii, e, 90),
            "bear3": int(ret < -3),
        })
    df = pd.DataFrame(rows).dropna().reset_index(drop=True)

    base = ["kospi_mom1m", "kospi_mom3m", "kospi_mom6m", "sox_mom1m", "sox_mom3m", "spx_mom1m", "spx_mom3m"]
    cands = {
        "credit": ["credit_chg1m", "credit_z3m"],
        "coppergold": ["cg_chg1m", "cg_chg3m"],
        "taiwan": ["twii_mom1m", "twii_mom3m"],
    }
    print(f"패널 {len(df)}개월 ({len(df)}), bear3 양성 {df['bear3'].sum()}")

    def wf(cols):
        X, y = df[cols].values, df["bear3"].values
        lp = np.full(len(df), np.nan); gp = np.full(len(df), np.nan)
        for i in range(MIN_TRAIN, len(df)):
            ytr = y[:i]
            if ytr.sum() < 8 or (len(ytr) - ytr.sum()) < 8:
                continue
            sc = StandardScaler().fit(X[:i])
            lr = LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000)
            lr.fit(sc.transform(X[:i]), ytr); lp[i] = lr.predict_proba(sc.transform(X[i:i+1]))[0, 1]
            gb = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150, random_state=42)
            gb.fit(X[:i], ytr); gp[i] = gb.predict_proba(X[i:i+1])[0, 1]
        out = {}
        for nm, p in [("LR", lp), ("GBM", gp)]:
            m = ~np.isnan(p)
            out[nm] = roc_auc_score(y[m], p[m]) if m.sum() > 20 and len(set(y[m])) > 1 else np.nan
        return out

    b = wf(base)
    b_best = max(b.values())
    print(f"\n  기준선(가격+US, {len(base)})        LR {b['LR']:.3f}  GBM {b['GBM']:.3f}")
    print("  ── 후보 하나씩 추가 (marginal 효과) ──")
    for nm, cols in cands.items():
        r = wf(base + cols)
        d = max(r.values()) - b_best
        print(f"  +{nm:10}  LR {r['LR']:.3f}  GBM {r['GBM']:.3f}   Δ {d:+.3f} {'✓' if d > 0.03 else ''}")
    allr = wf(base + sum(cands.values(), []))
    print(f"  +전부          LR {allr['LR']:.3f}  GBM {allr['GBM']:.3f}   Δ {max(allr.values())-b_best:+.3f}")
    print("\n  Δ +0.03↑면 그 feature가 실제 도움. 다 미미하면 월별은 역시 벽.")


if __name__ == "__main__":
    main()
