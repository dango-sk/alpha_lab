"""
analysis/hsmm_confusion.py
Confusion Matrix: hsmm+환율(5 seed 다수결 consensus) vs ai_v2.
ground truth = 실제 위험국면(fwd6m 낙폭<=-15%). overlap 2018~2026.
Bull을 Bull답게(Specificity) / Bear를 Bear답게(Recall) 분류하는지 명확히.
consensus 레짐은 analysis/regime_hsmm_fx_consensus.json 로 저장(FCF 재사용).
사용: .venv/bin/python analysis/hsmm_confusion.py
"""
import os, sys, json, warnings
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
MIN_TRAIN = 36
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
    dd6 = np.full(n, np.nan)
    for i in range(n-1):
        if i+6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path/path.cummax()-1).min()*100
    return BR, FX, yms, n, dd6


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


def confusion(reg, idx, dd6, label):
    ci = [i for i in idx if not np.isnan(dd6[i])]
    tp = sum(1 for i in ci if reg[i] == "Bear" and dd6[i] <= -15)   # 위험인데 Bear (정답)
    fp = sum(1 for i in ci if reg[i] == "Bear" and dd6[i] > -15)    # 안전인데 Bear (오경보)
    fn = sum(1 for i in ci if reg[i] == "Bull" and dd6[i] <= -15)   # 위험인데 Bull (놓침)
    tn = sum(1 for i in ci if reg[i] == "Bull" and dd6[i] > -15)    # 안전인데 Bull (정답)
    prec = tp/(tp+fp) if tp+fp else float('nan')
    rec = tp/(tp+fn) if tp+fn else float('nan')      # Bear를 Bear답게 (Sensitivity)
    spec = tn/(tn+fp) if tn+fp else float('nan')     # Bull을 Bull답게 (Specificity)
    f1 = 2*prec*rec/(prec+rec) if prec and rec and (prec+rec) else float('nan')
    acc = (tp+tn)/(tp+fp+fn+tn) if ci else float('nan')
    print(f"\n{'='*60}\n  [{label}]  (overlap, n={len(ci)}개월)\n{'='*60}")
    print(f"                        실제 위험(dd<=-15)   실제 안전")
    print(f"    예측 Bear              TP = {tp:>3}            FP = {fp:>3}")
    print(f"    예측 Bull              FN = {fn:>3}            TN = {tn:>3}")
    print(f"\n    Bear를 Bear답게 (Recall/Sensitivity) = {rec:.0%}   (위험 {tp+fn}개월 중 {tp}개 잡음)")
    print(f"    Bull을 Bull답게 (Specificity)        = {spec:.0%}   (안전 {tn+fp}개월 중 {tn}개 맞춤)")
    print(f"    Precision = {prec:.0%}  |  F1 = {f1:.2f}  |  Accuracy = {acc:.0%}")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=prec, rec=rec, spec=spec, f1=f1, acc=acc)


def main():
    BR, FX, yms, n, dd6 = build()
    XF = np.hstack([BR, FX])
    ov = [i for i in range(n) if yms[i] >= OV]
    print(f"overlap {yms[ov[0]]}~{yms[ov[-1]]} ({len(ov)}개월). hsmm+환율 5 seed 집계...", flush=True)
    bsum = np.zeros(n)
    for s in SEEDS:
        bsum += hsmm_bear(XF, n, s); print(f"  seed {s} done", flush=True)
    cons = ["Bear" if bsum[t] >= 3 else "Bull" for t in range(n)]
    (A / "regime_hsmm_fx_consensus.json").write_text(json.dumps(dict(zip(yms, cons)), ensure_ascii=False))
    aiv = ai_v2_map(yms)

    confusion(cons, ov, dd6, "hsmm+환율 (5 seed 다수결)")
    confusion(aiv, ov, dd6, "ai_v2")
    nrisk = sum(1 for i in ov if not np.isnan(dd6[i]) and dd6[i] <= -15)
    print(f"\n  (참고: overlap 위험국면 base rate = {nrisk}/{sum(1 for i in ov if not np.isnan(dd6[i]))}개월)")
    print("  consensus 레짐 저장 → analysis/regime_hsmm_fx_consensus.json")


if __name__ == "__main__":
    main()
