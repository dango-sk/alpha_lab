"""
analysis/hmm_structure_compare.py

HMM 구조 개선 비교 (rule 추가 X, 모델에 레짐 지속성 내장):
  diag       : 기존 Gaussian diag (baseline)
  sticky     : transmat_prior 대각 강화 → self-transition↑ (whipsaw↓)
  full       : full covariance (feature 상관 반영)
  gmm        : GMM-HMM (fat-tail 근사, Student-t 대용)
emission=[breadth,Δbreadth,newlow,Δnewlow,mom_decel], 2-state walk-forward.
평가: Recall / FalseAlarm / Lead / Whipsaw / Bear월 + 위기별 detection.
목표: A(rule)의 Recall13·Lead+1.8 유지하며 Whipsaw75·FA17 줄이기.
사용: .venv/bin/python analysis/hmm_structure_compare.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM, GMMHMM
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
MIN_TRAIN, SEED, WIN = 36, 42, 6


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
    rows = []
    for i in range(n):
        e = mends[i]
        rows.append([asof(breadth, e), asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
                     asof(newlow, e), asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
                     pct(kospi, e, 30) - pct(kospi, e, 180)])
    X = pd.DataFrame(rows, columns=["breadth", "brc", "newlow", "nlc", "md"]).fillna(0).values
    yms = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    Px = np.array([asof(kospi, e) for e in mends])
    dd6 = np.full(n, np.nan)
    for i in range(n - 1):
        if i + 6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path / path.cummax() - 1).min() * 100
    events = [i for i in range(n) if dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]

    def make(kind):
        if kind == "diag": return GaussianHMM(n_components=2, covariance_type="diag", n_iter=80, random_state=SEED)
        if kind == "sticky": return GaussianHMM(n_components=2, covariance_type="diag", n_iter=80, random_state=SEED, transmat_prior=np.array([[20., 1.], [1., 20.]]))
        if kind == "full": return GaussianHMM(n_components=2, covariance_type="full", n_iter=80, random_state=SEED)
        if kind == "gmm": return GMMHMM(n_components=2, n_mix=2, covariance_type="diag", n_iter=80, random_state=SEED)

    def regime_of(kind):
        reg = ["Bull"] * n
        for t in range(MIN_TRAIN, n):
            try:
                sc = StandardScaler().fit(X[:t]); Z = sc.transform(X[:t])
                hm = make(kind); hm.fit(Z); mu = hm.means_
                if mu.ndim == 3: mu = mu.mean(axis=1)   # GMM: [state,mix,feat]→[state,feat]
                bear = int(np.argmax(-mu[:, 0] - mu[:, 1] + mu[:, 2] + mu[:, 3] - mu[:, 4]))
                reg[t] = "Bear" if hm.predict(Z)[-1] == bear else "Bull"
            except Exception:
                reg[t] = reg[t-1]
        return reg

    def evalr(reg, label):
        onset = [t for t in range(1, n) if reg[t] == "Bear" and reg[t-1] != "Bear"]
        leads = []; miss = 0; crisis = []
        for ev in events:
            cand = [s for s in onset if abs(s - ev) <= WIN]
            if cand:
                b = min(cand, key=lambda s: abs(s-ev)); leads.append(ev - b); crisis.append((yms[ev], dd6[ev], ev - b))
            else:
                miss += 1; crisis.append((yms[ev], dd6[ev], None))
        fa = sum(1 for s in onset if all(abs(s-ev) > WIN for ev in events))
        whip = sum(1 for t in range(1, n) if reg[t] != reg[t-1])
        print(f"  {label:8} Recall {len(leads)}/{len(events)} (Miss {miss}), Lead {np.mean(leads):+.1f}, FA {fa}, Whipsaw {whip}, Bear월 {sum(1 for x in reg if x=='Bear')}")
        return crisis

    print(f"패널 {n}개월, 위기 {len(events)}개")
    print("\n(참고 A rule: Recall 13/14, Lead +1.8, FA 17, Whipsaw 75, Bear월 114)\n")
    crisis_tables = {}
    for kind in ["diag", "sticky", "full", "gmm"]:
        crisis_tables[kind] = evalr(regime_of(kind), kind)

    print("\n=== 위기별 detection (Lead, +선행 / X놓침) ===")
    print(f"  {'위기':9} {'낙폭':>5} | " + " ".join(f"{k:>7}" for k in ["diag", "sticky", "full", "gmm"]))
    base = crisis_tables["diag"]
    for idx in range(len(base)):
        ymv, ddv, _ = base[idx]
        cells = []
        for k in ["diag", "sticky", "full", "gmm"]:
            ld = crisis_tables[k][idx][2]
            cells.append(f"{ld:+d}" if ld is not None else "X")
        print(f"  {ymv:9} {ddv:>5.0f} | " + " ".join(f"{c:>7}" for c in cells))
    print("\n목표: A 대비 Whipsaw·FA↓ 하면서 Recall·Lead 유지하는 구조 찾기.")


if __name__ == "__main__":
    main()
