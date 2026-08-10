"""
analysis/fcf_threshold_sweep.py
consensus threshold sweep: Bear 발동 기준을 3/5·4/5·5/5로 바꿔 수익/방어 trade-off 탐색.
비교: bull_only / hsmm+환율 3/5 / 4/5 / 5/5 / ai_v2.
지표: 누적수익·CAGR·Sharpe·MDD·Calmar·turnover·Whipsaw·Bear개월 (+ Event Recall/FA 참고).
bsum(seed별 Bear수)는 analysis/hsmm_fx_bearcount.json 저장(risk overlay 단계 재사용).
사용: DATABASE_URL=<ip> .venv/bin/python analysis/fcf_threshold_sweep.py
"""
import os, json, sys, warnings
import numpy as np, pandas as pd, psycopg2
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
from analysis.hmm_fullcov_stabilize import bear_state, nb_logpmf_fn, hsmm_last_state
MIN_TRAIN, WIN, OV = 36, 6, "2018-04"
SEEDS = [0, 1, 7, 42, 123]
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"


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


def regime_stats(reg, yms, n, dd6):
    ov = [i for i in range(n) if yms[i] >= OV]
    whip = sum(1 for k in range(1, len(ov)) if reg[ov[k]] != reg[ov[k-1]])
    bearm = sum(1 for i in ov if reg[i] == "Bear")
    evs = [i for i in ov if dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]
    ons = [t for t in ov if reg[t] == "Bear" and t-1 >= 0 and reg[t-1] != "Bear"]
    caught = sum(1 for e in evs if any(abs(s-e) <= WIN for s in ons))
    fa = sum(1 for s in ons if all(abs(s-e) > WIN for e in evs))
    return whip, bearm, caught, len(evs), fa


def main():
    from lib.data import run_regime_combo_backtest
    BR, FX, yms, n, dd6 = build()
    XF = np.hstack([BR, FX])
    print("5 seed Bear-count 집계...", flush=True)
    bsum = np.zeros(n, int)
    for s in SEEDS:
        bsum += hsmm_bear(XF, n, s); print(f"  seed {s} done", flush=True)
    (A / "hsmm_fx_bearcount.json").write_text(json.dumps({yms[i]: int(bsum[i]) for i in range(n)}, ensure_ascii=False))

    maps = {
        "bull_only": {y: "Bull" for y in yms},
        "hsmm+환율 3/5": {yms[i]: ("Bear" if bsum[i] >= 3 else "Bull") for i in range(n)},
        "hsmm+환율 4/5": {yms[i]: ("Bear" if bsum[i] >= 4 else "Bull") for i in range(n)},
        "hsmm+환율 5/5": {yms[i]: ("Bear" if bsum[i] >= 5 else "Bull") for i in range(n)},
    }
    slot = A / "regime_rf_map.json"; bak = A / "regime_rf_map.json.swbak"
    if slot.exists() and not bak.exists(): bak.write_text(slot.read_text())
    res = {}
    for name, mp in maps.items():
        slot.write_text(json.dumps(mp, ensure_ascii=False))
        print(f"[FCF] {name} ...", flush=True)
        c = (run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="rf") or {}).get("REGIME_COMBO", {})
        reg = [mp[y] for y in yms]; res[name] = (c, regime_stats(reg, yms, n, dd6))
    if bak.exists(): slot.write_text(bak.read_text())
    print("[FCF] ai_v2 ...", flush=True)
    c = (run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="ai") or {}).get("REGIME_COMBO", {})
    # ai_v2 regime stats
    aip = json.load(open(A / "regime_agent_results.json")); ers = {r["as_of"][:7]: r.get("expected_return", 0) for r in aip}
    areg = {}; prev = "Bull"
    for y in sorted(ers):
        er = ers[y]; cur = ("Bear" if er <= -2 else "Bull") if prev == "Bull" else ("Bull" if er >= 1 else "Bear"); areg[y] = cur; prev = cur
    res["ai_v2"] = (c, regime_stats([areg.get(y, "Bull") for y in yms], yms, n, dd6))

    print(f"\n{'='*104}\n  consensus threshold sweep (FCF, overlap 2018~2026)\n{'='*104}")
    print(f"  {'모델':16}{'누적수익':>9}{'CAGR':>7}{'Sharpe':>7}{'MDD':>8}{'Calmar':>7}{'turnover':>9}{'Whip':>6}{'Bear월':>6}{'Recall':>8}{'FA':>4}")
    base_tr = res["bull_only"][0].get("total_return")
    for name, (c, st) in res.items():
        if not c: print(f"  {name:16}(no result)"); continue
        tr, cg, md, sh, to = c.get("total_return"), c.get("cagr"), c.get("mdd"), c.get("sharpe"), c.get("avg_turnover")
        cal = (cg/abs(md)) if (cg is not None and md) else float('nan')
        whip, bearm, caught, nev, fa = st
        print(f"  {name:16}{tr*100:>8.0f}%{cg*100:>6.1f}%{sh:>7.2f}{md*100:>7.1f}%{cal:>7.2f}{(to or 0):>9.2f}{whip:>6}{bearm:>6}{caught:>5}/{nev:<2}{fa:>4}")
    print(f"\n  목표: 누적/CAGR ≥ bull_only({base_tr*100:.0f}%) & MDD 의미있게↓ & Sharpe/Calmar↑ & Whip↓.")
    print("  bsum 저장: analysis/hsmm_fx_bearcount.json")


if __name__ == "__main__":
    main()
