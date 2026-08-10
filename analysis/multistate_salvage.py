"""
analysis/multistate_salvage.py
다상태 salvage: 상태→Bull/Bear 그룹을 bear_score가 아니라 *forward 수익률*로 매핑.
walk-forward로 각 fit 시점까지 실현된 수익만 사용(lookahead 없음): 학습구간 상태별 평균 forward수익 하위 floor(N/2)=Bear group.
2-state(base) / 3-state / 4-state 비교 + 상태별 수익률 centroid.
사용: DATABASE_URL=<ip> .venv/bin/python analysis/multistate_salvage.py
"""
import os, sys, warnings
import numpy as np, pandas as pd, psycopg2
from datetime import timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
from analysis.hmm_fullcov_stabilize import nb_logpmf_fn, hsmm_last_state
MIN_TRAIN, WIN, OV = 36, 6, "2018-04"
SEEDS = [0, 1, 7, 42, 123]


def bear_scores(mu):
    return -mu[:, 0] - mu[:, 1] + mu[:, 2] + mu[:, 3] - mu[:, 4]


def build():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    def macD(ind):
        x = pd.read_sql(f"SELECT left(period,10) p, value::float v FROM alpha_lab.macro_indicators WHERE indicator='{ind}' AND freq='D'", conn)
        x['p'] = pd.to_datetime(x['p']); return x.set_index('p')['v'].sort_index()
    kospi = macD('kospi'); usdkrw = macD('usd_krw'); conn.close()
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
    n = len(mends); yms = [pd.Period(e, 'M').strftime("%Y-%m") for e in mends]
    F = np.nan_to_num(np.array([[asof(breadth, e), asof(breadth, e)-asof(breadth, e-timedelta(days=30)),
        asof(newlow, e), asof(newlow, e)-asof(newlow, e-timedelta(days=30)), pct(kospi, e, 30)-pct(kospi, e, 180),
        pct(usdkrw, e, 90)] for e in mends]))
    Px = np.array([asof(kospi, e) for e in mends])
    ret = np.array([(Px[i+1]/Px[i]-1)*100 if i+1 < n else np.nan for i in range(n)])
    dd6 = np.full(n, np.nan)
    for i in range(n-1):
        if i+6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path/path.cummax()-1).min()*100
    return F, yms, n, ret, dd6


def hsmm_group_ret(F, ret, n, seed, N):
    """N-state HSMM, 그룹 = 학습구간 상태별 평균 forward수익 하위 floor(N/2)=Bear."""
    nbear = max(1, N // 2); bear = np.zeros(n, int)
    for t in range(MIN_TRAIN, n):
        try:
            Z = StandardScaler().fit_transform(F[:t]); hm = GaussianHMM(N, "full", n_iter=80, random_state=seed); hm.fit(Z)
            path = hm.predict(Z)
            mret = np.full(N, np.inf)  # 샘플없으면 inf(=Bull쪽)
            for j in range(N):
                vals = [ret[k] for k in range(t) if path[k] == j and not np.isnan(ret[k])]
                if vals: mret[j] = np.mean(vals)
            beargrp = set(np.argsort(mret)[:nbear].tolist())  # 저수익 하위 = Bear
            logB = hm._compute_log_likelihood(Z); ls = np.log(np.clip(hm.startprob_, 1e-12, 1))
            durs = {i: [] for i in range(N)}; rs, rl = path[0], 1
            for k in range(1, len(path)):
                if path[k] == rs: rl += 1
                else: durs[rs].append(rl); rs, rl = path[k], 1
            durs[rs].append(rl)
            logD = {i: nb_logpmf_fn(durs[i] if durs[i] else [6]) for i in range(N)}
            st = hsmm_last_state(logB, ls, logD, None)
            bear[t] = 1 if st in beargrp else 0
        except Exception:
            bear[t] = bear[t-1]
    return bear


def ev(reg, idx, n, ret, dd6):
    evs = [i for i in idx if dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]
    ons = [t for t in idx if reg[t] == "Bear" and t-1 >= 0 and reg[t-1] != "Bear"]
    lds = [e - min([s for s in ons if abs(s-e) <= WIN], key=lambda s: abs(s-e)) for e in evs if any(abs(s-e) <= WIN for s in ons)]
    fa = sum(1 for s in ons if all(abs(s-e) > WIN for e in evs))
    wh = sum(1 for k in range(1, len(idx)) if reg[idx[k]] != reg[idx[k-1]])
    br = np.nanmean([ret[i] for i in idx if reg[i] == "Bull"]); be = np.nanmean([ret[i] for i in idx if reg[i] == "Bear"])
    return len(lds), len(evs), (np.mean(lds) if lds else float('nan')), fa, wh, (br-be)


def centroid(F, ret, n, N):
    sc = StandardScaler().fit(F); Z = sc.transform(F)
    hm = GaussianHMM(N, "full", n_iter=80, random_state=42); hm.fit(Z)
    mo = sc.inverse_transform(hm.means_); bs = bear_scores(hm.means_); path = hm.predict(Z)
    mret = np.array([np.mean([ret[k] for k in range(n) if path[k] == j and not np.isnan(ret[k])] or [np.nan]) for j in range(N)])
    nbear = max(1, N // 2); beargrp = set(np.argsort(np.where(np.isnan(mret), np.inf, mret))[:nbear].tolist())
    print(f"  [{N}-state] (Breadth%/ΔBr/Mom/FX | bear_score | 평균fwd수익 | group[수익기준])")
    for j in np.argsort(bs):
        print(f"    s{j}: {mo[j,0]*100:5.1f} {mo[j,1]*100:+5.1f} {mo[j,4]:+6.1f} {mo[j,5]:+4.1f} | {bs[j]:+5.2f} | fwd {mret[j]:+5.2f}% | {'Bear' if j in beargrp else 'Bull'}")


def main():
    F, yms, n, ret, dd6 = build()
    ov = [i for i in range(n) if yms[i] >= OV]; yidx = {y: i for i, y in enumerate(yms)}; idxf = list(range(n))
    print(f"패널 {n}개월, overlap {len(ov)}, seeds={SEEDS}\n")
    print("=== 상태별 수익률 centroid (full fit, seed42; group=forward수익 기준) ===")
    for N in [2, 3, 4]: centroid(F, ret, n, N)

    print("\n=== 이벤트 지표 (forward수익 그룹매핑, overlap, consensus + μ±std) ===")
    print(f"  {'구조':10}{'Recall':>9}{'Lead':>6}{'FA':>5}{'Whip':>6}{'격차(ov)':>18}")
    cons_maps = {}
    for N in [2, 3, 4]:
        per = []; bsum = np.zeros(n)
        for s in SEEDS:
            b = hsmm_group_ret(F, ret, n, s, N); bsum += b
            per.append(ev(["Bear" if b[t] else "Bull" for t in range(n)], ov, n, ret, dd6))
        cons = ["Bear" if bsum[t] >= 3 else "Bull" for t in range(n)]; cons_maps[N] = cons
        rc, ne, ld, fa, wh, gp = ev(cons, ov, n, ret, dd6); gaps = [p[5] for p in per]
        print(f"  {N}-state{'':3}{rc}/{ne:<6}{ld:>+5.1f}{fa:>5}{wh:>6}   {gp:>+5.2f}p (μ{np.mean(gaps):+.2f}±{np.std(gaps):.2f})", flush=True)

    def first_bear(c, lo, hi): return next((y for y in yms if lo <= y <= hi and c[yidx[y]] == "Bear"), "—")
    def last_bear(c, lo, hi):
        r = "—"
        for y in yms:
            if lo <= y <= hi and c[yidx[y]] == "Bear": r = y
        return r
    def elead(c, lo, hi):
        evs = [i for i in idxf if lo <= yms[i] <= hi and dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]
        ons = [t for t in idxf if c[t] == "Bear" and t-1 >= 0 and c[t-1] != "Bear"]
        if not evs: return None
        e = evs[0]; cand = [s for s in ons if abs(s-e) <= WIN]
        return (e - min(cand, key=lambda s: abs(s-e))) if cand else None
    print(f"\n=== 전환 타이밍 (consensus) ===")
    print(f"  {'구조':10}{'2018lead':>9}{'2022lead':>9}{'2023Bear월':>10}{'2023막Bear':>11}")
    for N, c in cons_maps.items():
        l18 = elead(c, "2018-01", "2019-06"); l22 = elead(c, "2021-10", "2022-12")
        n23 = sum(1 for y in yms if "2023-01" <= y <= "2023-12" and c[yidx[y]] == "Bear")
        f18 = f"{l18:+d}" if l18 is not None else "·"; f22 = f"{l22:+d}" if l22 is not None else "·"
        print(f"  {N}-state{'':3}{f18:>9}{f22:>9}{n23:>10}{last_bear(c,'2022-10','2023-12'):>11}")
    print("\n  판정: 수익기준 매핑서 3/4-state 격차 양수+안정 & 2022/2023 개선이면 구조 유효, 아니면 2-state+환율 확정.")


if __name__ == "__main__":
    main()
