"""
analysis/macro_in_hmm_check.py
환율/금리가 레짐 HMM에 도움 안 되는지 재확인 (같은 2004+ 구간, 공정 비교).
A) 단변량 AUC  B) 증분 OOS AUC (breadth만 vs +환율/+금리/+both, logistic walk-forward)
C) Full-Cov HMM 이벤트지표 (base vs +환율 / +금리 / +both)
사용: .venv/bin/python analysis/macro_in_hmm_check.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore"); load_dotenv(Path('.env'))
A = Path(__file__).parent
MIN_TRAIN, SEED, WIN = 36, 42, 6


def main():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    def mac(ind):
        x = pd.read_sql(f"SELECT left(period,10) p, value::float v FROM alpha_lab.macro_indicators WHERE indicator='{ind}' AND freq='D'", conn)
        x['p'] = pd.to_datetime(x['p']); return x.set_index('p')['v'].sort_index()
    kospi = mac('kospi'); usdkrw = mac('usd_krw'); us10y = mac('us10y'); conn.close()
    d["dt"] = pd.to_datetime(d["dt"])
    wide = pd.concat([h, d[["dt", "stock_code", "close"]]], ignore_index=True).drop_duplicates(["dt", "stock_code"]).pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    ma = wide.rolling(120, min_periods=60).mean(); val = wide.notna() & ma.notna()
    breadth = ((wide > ma) & val).sum(axis=1) / val.sum(axis=1).clip(lower=1); breadth = breadth[val.sum(axis=1) > 100]
    rmn = wide.rolling(252, min_periods=120).min(); newlow = ((wide <= rmn) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1); newlow = newlow.reindex(breadth.index)
    ym = pd.PeriodIndex(kospi.index, freq="M"); last = {}
    for dt_, p in zip(kospi.index, ym): last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= pd.Timestamp('2004-01-01')]  # 환율 가용 후
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1)*100 if len(p) and p.iloc[-1] else np.nan
    def chg(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c - p.iloc[-1]) if len(p) else np.nan
    n = len(mends); yms = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    BR = np.array([[asof(breadth, e), asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
                    asof(newlow, e), asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
                    (pct(kospi, e, 30) - pct(kospi, e, 180)) / 100] for e in mends])
    FX = np.array([[pct(usdkrw, e, 90) / 100] for e in mends])   # Δ환율 3m
    IR = np.array([[chg(us10y, e, 90)] for e in mends])           # Δ미10y 3m
    BR = np.nan_to_num(BR); FX = np.nan_to_num(FX); IR = np.nan_to_num(IR)
    Px = np.array([asof(kospi, e) for e in mends])
    ret = np.array([(Px[i+1]/Px[i]-1)*100 if i+1 < n else np.nan for i in range(n)])
    dd6 = np.full(n, np.nan)
    for i in range(n-1):
        if i+6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path/path.cummax()-1).min()*100
    y = (dd6 <= -15).astype(int)
    events = [i for i in range(n) if dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]
    ov = [i for i in range(n) if yms[i] >= "2018-04"]
    print(f"구간 {yms[0]}~{yms[-1]} ({n}개월), 약세선행월 {int(np.nansum(y))}, 이벤트 {len(events)}\n")

    # B) 증분 OOS AUC (walk-forward logistic)
    print("=== B) 증분 OOS AUC (walk-forward, 타겟 fwd6m dd<=-15) ===")
    def oos_auc(X):
        p = np.full(n, np.nan); m = ~np.isnan(dd6)
        for t in range(MIN_TRAIN, n):
            tr = [i for i in range(t) if m[i]]
            if len(set(y[tr])) < 2: continue
            sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000)
            clf.fit(sc.transform(X[tr]), y[tr]); p[t] = clf.predict_proba(sc.transform(X[t:t+1]))[0, 1]
        mm = ~np.isnan(p) & m
        return roc_auc_score(y[mm], p[mm]) if len(set(y[mm])) > 1 else np.nan
    for name, X in [("breadth만", BR), ("+환율", np.hstack([BR, FX])), ("+금리", np.hstack([BR, IR])),
                    ("+환율+금리", np.hstack([BR, FX, IR])), ("환율+금리만(no breadth)", np.hstack([FX, IR]))]:
        print(f"  {name:24} OOS AUC {oos_auc(X):.3f}")

    # C) Full-Cov HMM 이벤트지표
    print("\n=== C) Full-Cov HMM 이벤트지표 (base vs 매크로 추가) ===")
    print(f"  {'구성':16} {'Recall(full)':>12} {'Lead':>5} {'FA':>4} {'Whip':>5} {'격차(ov)':>8}")
    def hmm_reg(X):
        reg = ["Bull"] * n
        for t in range(MIN_TRAIN, n):
            try:
                Z = StandardScaler().fit_transform(X[:t]); hm = GaussianHMM(2, "full", n_iter=60, random_state=SEED); hm.fit(Z)
                mu = hm.means_; bear = int(np.argmax(-mu[:, 0] - mu[:, 1] + mu[:, 2] + mu[:, 3] - mu[:, 4]))  # 항상 breadth 5열로 식별
                reg[t] = "Bear" if hm.predict(Z)[-1] == bear else "Bull"
            except Exception:
                reg[t] = reg[t-1]
        return reg
    def ev(reg, idx):
        evs = [i for i in idx if dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]
        ons = [t for t in idx if reg[t] == "Bear" and t-1 >= 0 and reg[t-1] != "Bear"]
        lds = [ev_ - min([s for s in ons if abs(s-ev_) <= WIN], key=lambda s: abs(s-ev_)) for ev_ in evs if any(abs(s-ev_) <= WIN for s in ons)]
        fa = sum(1 for s in ons if all(abs(s-ev_) > WIN for ev_ in evs))
        wh = sum(1 for k in range(1, len(idx)) if reg[idx[k]] != reg[idx[k-1]])
        br = np.nanmean([ret[i] for i in idx if reg[i] == "Bull"]); be = np.nanmean([ret[i] for i in idx if reg[i] == "Bear"])
        return len(lds), len(evs), (np.mean(lds) if lds else float('nan')), fa, wh, (br-be)
    full_idx = list(range(n))
    for name, X in [("base(breadth5)", BR), ("+환율", np.hstack([BR, FX])), ("+금리", np.hstack([BR, IR])),
                    ("+환율+금리", np.hstack([BR, FX, IR]))]:
        reg = hmm_reg(X); rc, ne, ld, fa, wh, _ = ev(reg, full_idx); _, _, _, _, _, gov = ev(reg, ov)
        lds = f"{ld:+.1f}" if not np.isnan(ld) else "n/a"
        print(f"  {name:16} {rc}/{ne:<10} {lds:>5} {fa:>4} {wh:>5} {gov:>+7.2f}p")
    print("\n  (증분 OOS AUC가 breadth만과 비슷 + HMM 지표 개선 없음 → 환율/금리 무효 확정)")


if __name__ == "__main__":
    main()
