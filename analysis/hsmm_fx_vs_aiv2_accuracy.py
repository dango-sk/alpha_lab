"""
analysis/hsmm_fx_vs_aiv2_accuracy.py
최종 모델 hsmm+환율 (5 seed 다수결 consensus) vs ai_v2 : 예측 정확도 비교 (overlap 2018~2026).
이벤트지표(Recall/Miss/Lead/FA/Whip) + 수익격차 + 분류정확도(precrisis P/R/F1) + 방향 적중률.
사용: .venv/bin/python analysis/hsmm_fx_vs_aiv2_accuracy.py
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
from analysis.hmm_fullcov_stabilize import bear_state, nb_logpmf_fn, hsmm_last_state, ai_v2_map
MIN_TRAIN, WIN = 36, 6
SEEDS = [0, 1, 7, 42, 123]
OV = "2018-04"


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


def full_bear(X, n, seed):
    """plain Full-Cov HMM (Viterbi), 환율 포함 X."""
    bear = np.zeros(n, int)
    for t in range(MIN_TRAIN, n):
        try:
            Z = StandardScaler().fit_transform(X[:t]); hm = GaussianHMM(2, "full", n_iter=60, random_state=seed); hm.fit(Z)
            b = bear_state(hm.means_)
            bear[t] = 1 if hm.predict(Z)[-1] == b else 0
        except Exception:
            bear[t] = bear[t-1]
    return bear


def hsmm_bear(X, n, seed):
    bear = np.zeros(n, int)
    for t in range(MIN_TRAIN, n):
        try:
            Z = StandardScaler().fit_transform(X[:t]); hm = GaussianHMM(2, "full", n_iter=60, random_state=seed); hm.fit(Z)
            b = bear_state(hm.means_); logB = hm._compute_log_likelihood(Z); ls = np.log(np.clip(hm.startprob_, 1e-12, 1))
            path = hm.predict(Z); durs = {0: [], 1: []}; rs, rl = path[0], 1
            for k in range(1, len(path)):
                if path[k] == rs: rl += 1
                else: durs[rs].append(rl); rs, rl = path[k], 1
            durs[rs].append(rl); logD = {i: nb_logpmf_fn(durs[i]) for i in range(2)}
            bear[t] = 1 if hsmm_last_state(logB, ls, logD, b) == b else 0
        except Exception:
            bear[t] = bear[t-1]
    return bear


def metrics(reg, idx, n, ret, dd6):
    evs = [i for i in idx if dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]
    ons = [t for t in idx if reg[t] == "Bear" and t-1 >= 0 and reg[t-1] != "Bear"]
    lds, miss = [], 0
    for e in evs:
        c = [s for s in ons if abs(s-e) <= WIN]
        if c: lds.append(e - min(c, key=lambda s: abs(s-e)))
        else: miss += 1
    fa = sum(1 for s in ons if all(abs(s-e) > WIN for e in evs))
    wh = sum(1 for k in range(1, len(idx)) if reg[idx[k]] != reg[idx[k-1]])
    br = np.nanmean([ret[i] for i in idx if reg[i] == "Bull"]); be = np.nanmean([ret[i] for i in idx if reg[i] == "Bear"])
    # 분류정확도: precrisis(fwd6m dd<=-15) vs Bear 예측 (월단위)
    ci = [i for i in idx if not np.isnan(dd6[i])]
    tp = sum(1 for i in ci if reg[i] == "Bear" and dd6[i] <= -15); fp = sum(1 for i in ci if reg[i] == "Bear" and dd6[i] > -15)
    fn = sum(1 for i in ci if reg[i] != "Bear" and dd6[i] <= -15)
    prec = tp/(tp+fp) if tp+fp else float('nan'); rec = tp/(tp+fn) if tp+fn else float('nan')
    f1 = 2*prec*rec/(prec+rec) if prec and rec and not np.isnan(prec) and not np.isnan(rec) and (prec+rec) else float('nan')
    # 방향 적중: Bear월 다음달<0 비율, Bull월 다음달>=0 비율
    ri = [i for i in idx if not np.isnan(ret[i])]
    bh = [ret[i] < 0 for i in ri if reg[i] == "Bear"]; uh = [ret[i] >= 0 for i in ri if reg[i] == "Bull"]
    bear_hit = np.mean(bh) if bh else float('nan'); bull_hit = np.mean(uh) if uh else float('nan')
    acc = np.mean([(reg[i] == "Bear") == (ret[i] < 0) for i in ri])
    return dict(rec_ev=len(lds), n_ev=len(evs), miss=miss, lead=(np.mean(lds) if lds else float('nan')),
                fa=fa, whip=wh, bull=br, bear=be, gap=br-be, prec=prec, rec_cls=rec, f1=f1,
                bear_hit=bear_hit, bull_hit=bull_hit, acc=acc, bearm=sum(1 for i in idx if reg[i] == "Bear"))


def main():
    BR, FX, yms, n, ret, dd6 = build()
    XF = np.hstack([BR, FX])
    ov = [i for i in range(n) if yms[i] >= OV]
    print(f"overlap {yms[ov[0]]}~{yms[ov[-1]]} ({len(ov)}개월). full+환율 / hsmm+환율 5 seed 집계...\n", flush=True)
    gaps = {"full+환율": [], "hsmm+환율": []}; bsum = {"full+환율": np.zeros(n), "hsmm+환율": np.zeros(n)}
    for s in SEEDS:
        for name, fn in [("full+환율", full_bear), ("hsmm+환율", hsmm_bear)]:
            b = fn(XF, n, s); bsum[name] += b
            reg_s = ["Bear" if b[t] else "Bull" for t in range(n)]
            gaps[name].append(metrics(reg_s, ov, n, ret, dd6)['gap'])
        print(f"  seed {s} done (full {gaps['full+환율'][-1]:+.2f}p / hsmm {gaps['hsmm+환율'][-1]:+.2f}p)", flush=True)
    cons = {name: ["Bear" if bsum[name][t] >= 3 else "Bull" for t in range(n)] for name in bsum}  # 다수결 >=3/5
    aiv = ai_v2_map(yms)

    M = {"full+환율": metrics(cons["full+환율"], ov, n, ret, dd6),
         "hsmm+환율": metrics(cons["hsmm+환율"], ov, n, ret, dd6),
         "ai_v2": metrics(aiv, ov, n, ret, dd6)}
    cols = ["full+환율", "hsmm+환율", "ai_v2"]
    print(f"\n{'='*86}\n  예측 정확도 비교 (overlap {yms[ov[0]]}~{yms[ov[-1]]})\n{'='*86}")
    fmt = {"Recall(이벤트)": lambda m: f"{m['rec_ev']}/{m['n_ev']}", "Miss": lambda m: m['miss'],
           "Lead(개월)": lambda m: f"{m['lead']:+.1f}", "False Alarm": lambda m: m['fa'],
           "Whipsaw": lambda m: m['whip'], "Bear월수": lambda m: m['bearm'],
           "Bull 월평균": lambda m: f"{m['bull']:+.2f}%", "Bear 월평균": lambda m: f"{m['bear']:+.2f}%",
           "수익격차": lambda m: f"{m['gap']:+.2f}p", "분류 Precision": lambda m: f"{m['prec']:.2f}",
           "분류 Recall": lambda m: f"{m['rec_cls']:.2f}", "분류 F1": lambda m: f"{m['f1']:.2f}",
           "Bear 방향적중": lambda m: f"{m['bear_hit']:.0%}", "Bull 방향적중": lambda m: f"{m['bull_hit']:.0%}",
           "월방향 정확도": lambda m: f"{m['acc']:.0%}"}
    print(f"  {'지표':16}" + "".join(f"{c:>13}" for c in cols))
    for name, f in fmt.items():
        print(f"  {name:16}" + "".join(f"{str(f(M[c])):>13}" for c in cols))
    print(f"\n  격차 seed별: full {['%+.2f' % g for g in gaps['full+환율']]} (평균 {np.mean(gaps['full+환율']):+.2f}±{np.std(gaps['full+환율']):.2f}p)")
    print(f"             hsmm {['%+.2f' % g for g in gaps['hsmm+환율']]} (평균 {np.mean(gaps['hsmm+환율']):+.2f}±{np.std(gaps['hsmm+환율']):.2f}p)")
    print("  분류 = precrisis(fwd6m dd<=-15) 월을 Bear로 맞췄나. 방향적중 = 그 레짐의 다음달 부호 일치율.")


if __name__ == "__main__":
    main()
