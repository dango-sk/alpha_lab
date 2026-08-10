"""
analysis/trajectory_validate.py

경보 이후 *경로(trajectory)* 가 TA/FA를 구분하나 + OOS holding 하나.
가설: FA는 경보 후 breadth 회복(Δ>0), TA는 지속 악화(Δ<0).
운용 핵심 = 1개월 경로(t→t+1): "Warning→1개월 확인→Bear" 가능성.
OOS 무너지면 → trailing stop이 최선 결론에 합의.
사용: .venv/bin/python analysis/trajectory_validate.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
WIN = 6


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
    # 월별 시리즈
    Br = [asof(breadth, e) for e in mends]; Nl = [asof(newlow, e) for e in mends]
    Md = [pct(kospi, e, 30) - pct(kospi, e, 180) for e in mends]; Brc = [asof(breadth, mends[i]) - asof(breadth, mends[i] - timedelta(days=30)) for i in range(len(mends))]
    Px = [asof(kospi, e) for e in mends]
    n = len(mends)
    dd6 = [np.nan] * n
    for i in range(n - 1):
        if i + 6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path / path.cummax() - 1).min() * 100
    ev_flag = [int(dd6[i] is not None and dd6[i] <= -15 and (i == 0 or not (dd6[i-1] is not None and dd6[i-1] <= -15))) for i in range(n)]
    events = [i for i in range(n) if ev_flag[i]]
    def ep(arr, t):
        hh = [a for a in arr[:t] if a == a]; return (np.array(hh) < arr[t]).mean() if hh else .5
    sig = [False] * n
    for t in range(12, n):
        top = np.mean([1 - ep(Br, t), ep([m if m == m else 0 for m in Md], t)])
        sig[t] = (top >= 0.6) or (ep([b if b == b else 0 for b in Brc], t) <= 0.15)
    onsets = [t for t in range(1, n) if sig[t] and not sig[t-1] and t + 3 < n]
    lab = np.array([1 if any(abs(s - ev) <= WIN for ev in events) else 0 for s in onsets])  # TA=1

    # trajectory feature (경보 후)
    rows = []
    for s in onsets:
        rows.append({
            "dbr1": Br[s+1] - Br[s], "dbr3": Br[s+3] - Br[s],          # 회복(+)/악화(-)
            "dnl1": Nl[s+1] - Nl[s], "dmd1": Md[s+1] - Md[s],
            "ret1": (Px[s+1] / Px[s] - 1) * 100,
        })
    td = pd.DataFrame(rows); td["TA"] = lab
    print(f"경보 {len(onsets)}개 (TA {lab.sum()}, FA {(lab==0).sum()})\n")
    print("=== 경보 후 1개월 경로: TA vs FA 평균 (가설: FA는 회복) ===")
    for c in ["dbr1", "dbr3", "dnl1", "dmd1", "ret1"]:
        ta = td[c][td["TA"] == 1].mean(); fa = td[c][td["TA"] == 0].mean()
        print(f"  {c:6} TA {ta:+.3f}  FA {fa:+.3f}  (FA가 {'회복=+' if c.startswith('dbr') or c=='ret1' else ''} 쪽이면 가설 맞음)")

    # OOS: 1개월 경로로 FA 구분
    half = len(td) // 2
    e_, l_ = td.iloc[:half], td.iloc[half:]
    feats = ["dbr1", "dnl1", "dmd1", "ret1"]
    Xe, ye = e_[feats].values, 1 - e_["TA"].values; Xl, yl = l_[feats].values, 1 - l_["TA"].values
    print(f"\n전반부 {half}개 / 후반부 {len(td)-half}개")
    if len(set(ye)) > 1 and len(set(yl)) > 1:
        sc = StandardScaler().fit(Xe); lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Xe), ye)
        ins = roc_auc_score(ye, lr.predict_proba(sc.transform(Xe))[:, 1]); oos = roc_auc_score(yl, lr.predict_proba(sc.transform(Xl))[:, 1])
        print(f"  1개월경로 LR FA예측: 전반부(IS) {ins:.3f} → 후반부(OOS) {oos:.3f}")
    # dbr1 단독 OOS
    for c in ["dbr1", "dbr3"]:
        ae = roc_auc_score(1 - e_["TA"], -e_[c]) if e_["TA"].nunique() > 1 else np.nan
        al = roc_auc_score(1 - l_["TA"], -l_[c]) if l_["TA"].nunique() > 1 else np.nan
        print(f"  {c} 단독 FA설명 AUC: 전반부 {ae:.3f} → 후반부(OOS) {al:.3f}")
    print("\n판정: 1개월경로 OOS AUC > 0.6 → 'Warning→1개월확인→Bear' 가능. ~0.5 → trailing stop 최선.")


if __name__ == "__main__":
    main()
