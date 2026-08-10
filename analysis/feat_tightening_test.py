"""
analysis/feat_tightening_test.py
긴축형 Bear(2018·2022) 보강용 피처 테스트 — 노이즈 안 늘리는지 검증.
구성: base(breadth5+FX) / +DXYΔ3m / +US10YΔ3m / +DXY+US10Y / +DXY+US10Y+KR10YΔ3m.
5 seed HSMM. 지표: Event Recall/Lead/FA/Whipsaw/격차 (mean±std + consensus).
별도: 2018·2022 조기탐지(lead), 2023 recovery Bear 잔류, FA/Whipsaw 증가 여부.
사용: DATABASE_URL=<ip> .venv/bin/python analysis/feat_tightening_test.py
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


def build():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    def macD(ind):
        x = pd.read_sql(f"SELECT left(period,10) p, value::float v FROM alpha_lab.macro_indicators WHERE indicator='{ind}' AND freq='D'", conn)
        x['p'] = pd.to_datetime(x['p']); return x.set_index('p')['v'].sort_index()
    kospi = macD('kospi'); usdkrw = macD('usd_krw'); dxy = macD('dxy'); us10 = macD('us10y'); kr10 = macD('bond_10y'); conn.close()
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
    def chg(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c - p.iloc[-1]) if len(p) else np.nan
    n = len(mends); yms = [pd.Period(e, 'M').strftime("%Y-%m") for e in mends]
    BR = np.nan_to_num(np.array([[asof(breadth, e), asof(breadth, e)-asof(breadth, e-timedelta(days=30)),
        asof(newlow, e), asof(newlow, e)-asof(newlow, e-timedelta(days=30)), pct(kospi, e, 30)-pct(kospi, e, 180)] for e in mends]))
    FX = np.nan_to_num(np.array([[pct(usdkrw, e, 90)] for e in mends]))
    DXY = np.nan_to_num(np.array([[pct(dxy, e, 90)] for e in mends]))
    US = np.nan_to_num(np.array([[chg(us10, e, 90)] for e in mends]))
    KR = np.nan_to_num(np.array([[chg(kr10, e, 90)] for e in mends]))
    Px = np.array([asof(kospi, e) for e in mends])
    ret = np.array([(Px[i+1]/Px[i]-1)*100 if i+1 < n else np.nan for i in range(n)])
    dd6 = np.full(n, np.nan)
    for i in range(n-1):
        if i+6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path/path.cummax()-1).min()*100
    return BR, FX, DXY, US, KR, yms, n, ret, dd6


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


def ev(reg, idx, n, ret, dd6):
    evs = [i for i in idx if dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]
    ons = [t for t in idx if reg[t] == "Bear" and t-1 >= 0 and reg[t-1] != "Bear"]
    lds = [e - min([s for s in ons if abs(s-e) <= WIN], key=lambda s: abs(s-e)) for e in evs if any(abs(s-e) <= WIN for s in ons)]
    fa = sum(1 for s in ons if all(abs(s-e) > WIN for e in evs))
    wh = sum(1 for k in range(1, len(idx)) if reg[idx[k]] != reg[idx[k-1]])
    br = np.nanmean([ret[i] for i in idx if reg[i] == "Bull"]); be = np.nanmean([ret[i] for i in idx if reg[i] == "Bear"])
    return len(lds), len(evs), (np.mean(lds) if lds else float('nan')), fa, wh, (br-be)


def main():
    BR, FX, DXY, US, KR, yms, n, ret, dd6 = build()
    base = np.hstack([BR, FX])
    CONFIGS = {
        "base(+FX)": base,
        "+DXY": np.hstack([base, DXY]),
        "+US10Y": np.hstack([base, US]),
        "+DXY+US10Y": np.hstack([base, DXY, US]),
        "+DXY+US10Y+KR10Y": np.hstack([base, DXY, US, KR]),
    }
    ov = [i for i in range(n) if yms[i] >= OV]; idxf = list(range(n))
    yidx = {y: i for i, y in enumerate(yms)}
    print(f"패널 {n}개월, overlap {len(ov)}개월, seeds={SEEDS}\n", flush=True)

    hdr = f"  {'구성':18}{'Recall(ov)':>11}{'Lead':>6}{'FA':>5}{'Whip':>6}{'격차(ov)':>16}"
    print("=== 이벤트 지표 (overlap 2018~, consensus + seed 평균±std) ===")
    print(hdr)
    cons_maps = {}
    for name, X in CONFIGS.items():
        per = []; bsum = np.zeros(n)
        for s in SEEDS:
            b = hsmm_bear(X, n, s); bsum += b
            reg = ["Bear" if b[t] else "Bull" for t in range(n)]
            per.append(ev(reg, ov, n, ret, dd6))
        cons = ["Bear" if bsum[t] >= 3 else "Bull" for t in range(n)]
        cons_maps[name] = (cons, bsum)
        rc, ne, ld, fa, wh, gp = ev(cons, ov, n, ret, dd6)
        gaps = [p[5] for p in per]; fas = [p[3] for p in per]; whs = [p[4] for p in per]; lds = [p[2] for p in per]
        print(f"  {name:18}{rc}/{ne:<8}{ld:>+5.1f}{fa:>5}{wh:>6}   {gp:>+5.2f}p (μ{np.mean(gaps):+.2f}±{np.std(gaps):.2f})", flush=True)
        (A / f"regime_feat_{name.replace('+','_').replace('(','').replace(')','')}.json").write_text(
            json.dumps(dict(zip(yms, cons)), ensure_ascii=False))

    # 긴축형 Bear 조기탐지 + 2023 recovery
    def first_bear(cons, lo, hi):
        for y in yms:
            if lo <= y <= hi and cons[yidx[y]] == "Bear": return y
        return "—"
    def last_bear(cons, lo, hi):
        r = "—"
        for y in yms:
            if lo <= y <= hi and cons[yidx[y]] == "Bear": r = y
        return r
    def event_lead(cons, lo, hi):  # 그 구간 event onset 대비 lead
        evs = [i for i in idxf if lo <= yms[i] <= hi and dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]
        ons = [t for t in idxf if cons[t] == "Bear" and t-1 >= 0 and cons[t-1] != "Bear"]
        if not evs: return None
        e = evs[0]; cand = [s for s in ons if abs(s-e) <= WIN]
        return (e - min(cand, key=lambda s: abs(s-e))) if cand else None

    print(f"\n=== 긴축형 Bear 조기탐지 & 2023 recovery (consensus) ===")
    print(f"  {'구성':18}{'2018첫Bear':>11}{'2018lead':>9}{'2022첫Bear':>11}{'2022lead':>9}{'2023Bear월':>10}{'2023막Bear':>11}")
    for name, (cons, _) in cons_maps.items():
        b18 = first_bear(cons, "2018-01", "2018-12"); l18 = event_lead(cons, "2018-01", "2019-06")
        b22 = first_bear(cons, "2021-12", "2022-12"); l22 = event_lead(cons, "2021-10", "2022-12")
        n23 = sum(1 for y in yms if "2023-01" <= y <= "2023-12" and cons[yidx[y]] == "Bear")
        lb = last_bear(cons, "2022-10", "2023-12")
        f18 = f"{l18:+d}" if l18 is not None else "·"; f22 = f"{l22:+d}" if l22 is not None else "·"
        print(f"  {name:18}{b18:>11}{f18:>9}{b22:>11}{f22:>9}{n23:>10}{lb:>11}")
    print("\n  목표: 2018·2022 첫Bear/lead 빨라지고, 2023막Bear 안 늘고, FA/Whip 안 늘면 → 보강 성공.")
    print("  (lead + = event보다 선행, 2023Bear월↑ = recovery에 과잉잔류)")


if __name__ == "__main__":
    main()
