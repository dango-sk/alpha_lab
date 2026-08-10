"""
analysis/hsmm_fx_casestudy.py
환율이 *왜* 좋아졌는가 — 사건별 분석.
5 seed 전반에서 hsmm(no FX) vs hsmm+환율의 Bear 판정이 일관되게 갈린 달만 추출(구조적 효과).
  - FX→Bull: no-FX=Bear인데 +FX=Bull (거짓 약세 제거)
  - FX→Bear: no-FX=Bull인데 +FX=Bear (위기 조기/추가 포착)
각 달에 Δ환율(3m)·breadth·mom_decel·다음달수익·향후6m낙폭 붙임.
사용: .venv/bin/python analysis/hsmm_fx_casestudy.py
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
MIN_TRAIN = 36
SEEDS = [0, 1, 7, 42, 123]
HI, LO = 0.6, 0.4  # seed 합의 임계: Bear율>=0.6 vs <=0.4


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
    brd = np.array([asof(breadth, e) for e in mends]); md = np.array([pct(kospi, e, 30) - pct(kospi, e, 180) for e in mends])
    BR = np.nan_to_num(np.array([[asof(breadth, e), asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
        asof(newlow, e), asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
        pct(kospi, e, 30) - pct(kospi, e, 180)] for e in mends]))
    fxraw = np.array([pct(usdkrw, e, 90) for e in mends]); FX = np.nan_to_num(fxraw.reshape(-1, 1))
    Px = np.array([asof(kospi, e) for e in mends])
    ret = np.array([(Px[i+1]/Px[i]-1)*100 if i+1 < n else np.nan for i in range(n)])
    dd6 = np.full(n, np.nan)
    for i in range(n-1):
        if i+6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path/path.cummax()-1).min()*100
    return BR, FX, fxraw, brd, md, yms, n, ret, dd6


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


def main():
    BR, FX, fxraw, brd, md, yms, n, ret, dd6 = build()
    XF = np.hstack([BR, FX])
    print(f"구간 {yms[0]}~{yms[-1]} ({n}개월), seeds={SEEDS}. seed 합의 Bear율 집계 중...\n", flush=True)
    nb = np.zeros(n); fb = np.zeros(n)
    for s in SEEDS:
        nb += hsmm_bear(BR, n, s); fb += hsmm_bear(XF, n, s); print(f"  seed {s} done", flush=True)
    nb /= len(SEEDS); fb /= len(SEEDS)  # Bear율 0~1

    def show(rows, title):
        print(f"\n{'='*92}\n  {title}  ({len(rows)}건)\n{'='*92}")
        print(f"  {'월':9} {'Δ환율3m':>8} {'breadth':>8} {'mom_dec':>8} {'no-FX Bear율':>11} {'+FX Bear율':>10} {'다음달':>7} {'향후6m낙폭':>9}")
        for t in rows:
            crisis = " ★위기" if dd6[t] <= -15 else ""
            print(f"  {yms[t]:9} {fxraw[t]:>+7.2f}% {brd[t]:>7.2f} {md[t]:>+7.2f} {nb[t]:>10.0%} {fb[t]:>9.0%} "
                  f"{ret[t]:>+6.2f}% {dd6[t]:>+8.1f}%{crisis}")
        if rows:
            print(f"  ── 평균: Δ환율 {np.mean([fxraw[t] for t in rows]):+.2f}%, 다음달수익 {np.nanmean([ret[t] for t in rows]):+.2f}%, 향후6m낙폭 {np.nanmean([dd6[t] for t in rows]):+.1f}%")

    idx = [t for t in range(n) if yms[t] >= "2004-01" and not np.isnan(dd6[t])]
    fx2bull = [t for t in idx if nb[t] >= HI and fb[t] <= LO]   # 환율이 Bear 제거 → Bull
    fx2bear = [t for t in idx if nb[t] <= LO and fb[t] >= HI]   # 환율이 Bear 추가/조기
    show(fx2bull, "① FX→Bull : 환율 없으면 Bear, 환율 넣으면 Bull (거짓 약세 제거)")
    show(fx2bear, "② FX→Bear : 환율 없으면 Bull, 환율 넣으면 Bear (위기 조기/추가 포착)")

    print(f"\n{'='*92}\n  해석 근거\n{'='*92}")
    if fx2bull:
        print(f"  ① FX→Bull {len(fx2bull)}건: Δ환율 평균 {np.mean([fxraw[t] for t in fx2bull]):+.2f}% (KRW 안정/강세), "
              f"실제 다음달 {np.nanmean([ret[t] for t in fx2bull]):+.2f}% → breadth만 보면 약세로 오판할 구간을 환율이 '원화 안정'으로 걸러 Bull 유지.")
    if fx2bear:
        print(f"  ② FX→Bear {len(fx2bear)}건: Δ환율 평균 {np.mean([fxraw[t] for t in fx2bear]):+.2f}% (KRW 급격 약세), "
              f"향후6m낙폭 평균 {np.nanmean([dd6[t] for t in fx2bear]):+.1f}% → 원화 급락이 자본유출 risk-off 신호로 약세를 조기 확정.")
    print("\n  (Δ환율3m>0 = 원화 약세/달러 강세. ★ = 실제 fwd6m dd<=-15 위기월)")


if __name__ == "__main__":
    main()
