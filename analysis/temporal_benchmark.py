"""
analysis/temporal_benchmark.py

시퀀스 모델 벤치마크: 최근 K개월 trajectory를 입력하는 windowed 모델 (HMM/snapshot 아닌 다른 클래스).
316개월엔 LSTM이 과적합 → windowed-GBM/Logistic이 통계적으로 올바른 시퀀스 벤치마크.
같은 trade-off면 → 모델 아니라 데이터 정보 한계 확정.

입력: 최근 K=3개월 [breadth,Δbreadth,newlow,Δnewlow,mom_decel] flatten (15차원).
타겟: bear regime(forward 6m dd<=-15). walk-forward. 같은 event 지표.
사용: .venv/bin/python analysis/temporal_benchmark.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
K, MIN_TRAIN, WIN = 3, 48, 6


def main():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    kk = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators WHERE indicator='kospi' AND freq='D' ORDER BY period", conn); conn.close()
    d["dt"] = pd.to_datetime(d["dt"]); kk["dt"] = pd.to_datetime(kk["dt"]); kospi = kk.set_index("dt")["v"].sort_index()
    wide = pd.concat([h, d[["dt", "stock_code", "close"]]], ignore_index=True).drop_duplicates(["dt", "stock_code"]).pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    ma = wide.rolling(120, min_periods=60).mean(); val = wide.notna() & ma.notna()
    breadth = ((wide > ma) & val).sum(axis=1) / val.sum(axis=1).clip(lower=1); breadth = breadth[val.sum(axis=1) > 100]
    rmn = wide.rolling(252, min_periods=120).min(); newlow = ((wide <= rmn) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1); newlow = newlow.reindex(breadth.index)
    ym = pd.PeriodIndex(kospi.index, freq="M"); last = {}
    for dt_, p in zip(kospi.index, ym): last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1) if len(p) and p.iloc[-1] else np.nan
    n = len(mends)
    F = []
    for i in range(n):
        e = mends[i]
        F.append([asof(breadth, e), asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
                  asof(newlow, e), asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
                  pct(kospi, e, 30) - pct(kospi, e, 180)])
    F = np.nan_to_num(np.array(F))
    Px = np.array([asof(kospi, e) for e in mends]); yms = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    dd6 = np.full(n, np.nan)
    for i in range(n - 1):
        if i + 6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path / path.cummax() - 1).min() * 100
    y = (dd6 <= -15).astype(int)
    events = [i for i in range(n) if dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]
    # windowed feature: 최근 K개월 flatten
    Xw = np.full((n, K * 5), np.nan)
    for t in range(K - 1, n):
        Xw[t] = F[t-K+1:t+1].flatten()

    def evalr(reg, label):
        onset = [t for t in range(1, n) if reg[t] == "Bear" and reg[t-1] != "Bear"]
        leads = []; miss = 0
        for ev in events:
            cand = [s for s in onset if abs(s-ev) <= WIN]
            if cand: leads.append(ev - min(cand, key=lambda s: abs(s-ev)))
            else: miss += 1
        fa = sum(1 for s in onset if all(abs(s-ev) > WIN for ev in events))
        whip = sum(1 for t in range(1, n) if reg[t] != reg[t-1])
        print(f"  {label:16} Recall {len(leads)}/{len(events)} (Miss {miss}), Lead {np.mean(leads):+.1f}, FA {fa}, Whipsaw {whip}, Bear월 {sum(1 for x in reg if x=='Bear')}")

    print(f"패널 {n}개월, 위기 {len(events)}개 (windowed K={K}, 15차원 시퀀스)")
    print("(참고 A rule: Recall 13/14, Lead +1.8, FA 17, Whipsaw 75)\n")
    for name, mk in [("windowed-LR", lambda: LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000)),
                     ("windowed-GBM", lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42))]:
        p = np.full(n, np.nan)
        for t in range(MIN_TRAIN, n):
            m = ~np.isnan(Xw[:t]).any(axis=1)
            ytr = y[:t][m]
            if ytr.sum() < 8 or (len(ytr) - ytr.sum()) < 8 or np.isnan(Xw[t]).any(): continue
            sc = StandardScaler().fit(Xw[:t][m]); clf = mk()
            clf.fit(sc.transform(Xw[:t][m]), ytr); p[t] = clf.predict_proba(sc.transform(Xw[t:t+1]))[0, 1]
        reg = ["Bull" if (np.isnan(p[t]) or p[t] < 0.5) else "Bear" for t in range(n)]
        evalr(reg, name)
    print("\n판정: 시퀀스 모델도 같은 trade-off(낮은 FA면 Lead↓, 높은 Recall이면 FA↑)면 → 데이터 정보 한계 확정.")


if __name__ == "__main__":
    main()
