"""
analysis/hsmm_robustness.py
hsmm 분별력이 seed 전반 안정적인지 무조건 확인 (base가 seed42 운빨이었던 게 드러난 뒤).
구성: hsmm(breadth5) vs hsmm+환율, 5 seed, 2000+ (메인 동일). seed별 + 평균 + 표준편차.
사용: .venv/bin/python analysis/hsmm_robustness.py
"""
import os, sys, warnings
import numpy as np, pandas as pd, psycopg2
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore"); load_dotenv(Path('.env'))
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
from analysis.hmm_fullcov_stabilize import bear_state, nb_logpmf_fn, hsmm_last_state
MIN_TRAIN, WIN = 36, 6
SEEDS = [0, 1, 7, 42, 123]
# 참고(이미 측정): base(breadth5) 평균 격차(ov) -1.35p, +환율 +0.84p


def build():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    def mac(ind):
        x = pd.read_sql(f"SELECT left(period,10) p, value::float v FROM alpha_lab.macro_indicators WHERE indicator='{ind}' AND freq='D'", conn)
        x['p'] = pd.to_datetime(x['p']); return x.set_index('p')['v'].sort_index()
    kospi = mac('kospi'); usdkrw = mac('usd_krw'); conn.close()
    d["dt"] = pd.to_datetime(d["dt"])
    wide = pd.concat([h, d[["dt", "stock_code", "close"]]], ignore_index=True).drop_duplicates(["dt", "stock_code"]).pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    ma = wide.rolling(120, min_periods=60).mean(); val = wide.notna() & ma.notna()
    breadth = ((wide > ma) & val).sum(axis=1) / val.sum(axis=1).clip(lower=1); breadth = breadth[val.sum(axis=1) > 100]
    rmn = wide.rolling(252, min_periods=120).min(); newlow = ((wide <= rmn) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1); newlow = newlow.reindex(breadth.index)
    ym = pd.PeriodIndex(kospi.index, freq="M"); last = {}
    for dt_, p in zip(kospi.index, ym): last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1)*100 if len(p) and p.iloc[-1] else np.nan
    n = len(mends); yms = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    BR = np.nan_to_num(np.array([[asof(breadth, e), asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
        asof(newlow, e), asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
        pct(kospi, e, 30) - pct(kospi, e, 180)] for e in mends]))
    FX = np.nan_to_num(np.array([[pct(usdkrw, e, 90)] for e in mends]))
    Px = np.array([asof(kospi, e) for e in mends])
    ret = np.array([(Px[i+1]/Px[i]-1)*100 if i+1 < n else np.nan for i in range(n)])
    dd6 = np.full(n, np.nan)
    for i in range(n-1):
        if i+6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path/path.cummax()-1).min()*100
    return BR, FX, yms, n, ret, dd6


def hsmm_reg(X, n, seed):
    reg = ["Bull"] * n
    for t in range(MIN_TRAIN, n):
        try:
            Z = StandardScaler().fit_transform(X[:t])
            hm = GaussianHMM(2, "full", n_iter=60, random_state=seed); hm.fit(Z)
            b = bear_state(hm.means_)
            logB = hm._compute_log_likelihood(Z); logstart = np.log(np.clip(hm.startprob_, 1e-12, 1))
            path = hm.predict(Z); durs = {0: [], 1: []}; rs, rl = path[0], 1
            for k in range(1, len(path)):
                if path[k] == rs: rl += 1
                else: durs[rs].append(rl); rs, rl = path[k], 1
            durs[rs].append(rl)
            logD = {i: nb_logpmf_fn(durs[i]) for i in range(2)}
            reg[t] = "Bear" if hsmm_last_state(logB, logstart, logD, b) == b else "Bull"
        except Exception:
            reg[t] = reg[t-1]
    return reg


def ev(reg, idx, n, ret, dd6):
    evs = [i for i in idx if dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]
    ons = [t for t in idx if reg[t] == "Bear" and t-1 >= 0 and reg[t-1] != "Bear"]
    lds = [e - min([s for s in ons if abs(s-e) <= WIN], key=lambda s: abs(s-e)) for e in evs if any(abs(s-e) <= WIN for s in ons)]
    fa = sum(1 for s in ons if all(abs(s-e) > WIN for e in evs))
    wh = sum(1 for k in range(1, len(idx)) if reg[idx[k]] != reg[idx[k-1]])
    br = np.nanmean([ret[i] for i in idx if reg[i] == "Bull"]); be = np.nanmean([ret[i] for i in idx if reg[i] == "Bear"])
    return len(lds), len(evs), (np.mean(lds) if lds else float('nan')), fa, wh, (br-be)


def main():
    BR, FX, yms, n, ret, dd6 = build()
    full_idx = list(range(n)); ov = [i for i in range(n) if yms[i] >= "2018-04"]
    print(f"구간 {yms[0]}~{yms[-1]} ({n}개월), seeds={SEEDS}\n")
    configs = [("hsmm", BR), ("hsmm+환율", np.hstack([BR, FX]))]
    agg = {k: [] for k, _ in configs}
    for seed in SEEDS:
        print(f"--- seed {seed} ---")
        print(f"  {'구성':10} {'Recall':>7} {'Lead':>5} {'FA':>4} {'Whip':>5} {'격차(full)':>9} {'격차(ov)':>8}")
        for name, X in configs:
            reg = hsmm_reg(X, n, seed)
            rc, ne, ld, fa, wh, gf = ev(reg, full_idx, n, ret, dd6)
            _, _, _, _, _, gov = ev(reg, ov, n, ret, dd6)
            agg[name].append((rc, ld, fa, wh, gf, gov))
            lds = f"{ld:+.1f}" if not np.isnan(ld) else "n/a"
            print(f"  {name:10} {rc}/{ne:<5} {lds:>5} {fa:>4} {wh:>5} {gf:>+8.2f}p {gov:>+7.2f}p", flush=True)
    print(f"\n=== seed 평균 ± 표준편차 (n={len(SEEDS)}) ===")
    print(f"  {'구성':10} {'Recall':>7} {'FA':>11} {'Whip':>12} {'격차(ov)':>14}")
    for name in agg:
        a = np.array(agg[name], float)
        print(f"  {name:10} {a[:,0].mean():.1f}/14   {a[:,2].mean():.1f}±{a[:,2].std():.1f}   "
              f"{a[:,3].mean():.1f}±{a[:,3].std():.1f}   {a[:,5].mean():+.2f}±{a[:,5].std():.2f}p")
    print("\n  격차(ov) 부호: 5 seed 전부 + 면 안정(채택가능), 섞이면 불안정.")
    for name in agg:
        govs = [r[5] for r in agg[name]]
        pos = sum(1 for g in govs if g > 0)
        print(f"  {name:10} 격차(ov) seed별 = {['%+.2f' % g for g in govs]}  (양수 {pos}/{len(govs)})")


if __name__ == "__main__":
    main()
