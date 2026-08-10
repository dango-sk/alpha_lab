"""
analysis/fx_revalidate_stable.py
환율 재검증: 안정 base(2000년 시작, 메인 동일) + multi-seed 에서 base vs +환율(Δ usd_krw 3m).
안정된 base에서도 환율이 격차/FA/Whip을 *일관되게* 개선하는지 → 채택 여부 판단.
사용: .venv/bin/python analysis/fx_revalidate_stable.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore"); load_dotenv(Path('.env'))
A = Path(__file__).parent
MIN_TRAIN, WIN = 36, 6
SEEDS = [0, 1, 7, 42, 123]


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
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]  # 메인과 동일(2000+)
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1)*100 if len(p) and p.iloc[-1] else np.nan
    n = len(mends); yms = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    BR = np.nan_to_num(np.array([[asof(breadth, e), asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
        asof(newlow, e), asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
        pct(kospi, e, 30) - pct(kospi, e, 180)] for e in mends]))  # 메인과 동일 스케일
    FX = np.nan_to_num(np.array([[pct(usdkrw, e, 90)] for e in mends]))  # Δ환율 3m
    Px = np.array([asof(kospi, e) for e in mends])
    ret = np.array([(Px[i+1]/Px[i]-1)*100 if i+1 < n else np.nan for i in range(n)])
    dd6 = np.full(n, np.nan)
    for i in range(n-1):
        if i+6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path/path.cummax()-1).min()*100
    return BR, FX, yms, n, ret, dd6


def hmm_reg(X, n, seed):
    reg = ["Bull"] * n
    for t in range(MIN_TRAIN, n):
        try:
            Z = StandardScaler().fit_transform(X[:t]); hm = GaussianHMM(2, "full", n_iter=60, random_state=seed); hm.fit(Z)
            mu = hm.means_; bear = int(np.argmax(-mu[:, 0] - mu[:, 1] + mu[:, 2] + mu[:, 3] - mu[:, 4]))
            reg[t] = "Bear" if hm.predict(Z)[-1] == bear else "Bull"
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
    print(f"구간 {yms[0]}~{yms[-1]} ({n}개월, 메인 동일 2000+), seeds={SEEDS}\n")
    configs = [("base", BR), ("+환율", np.hstack([BR, FX]))]
    agg = {k: [] for k, _ in configs}
    for seed in SEEDS:
        print(f"--- seed {seed} ---")
        print(f"  {'구성':8} {'Recall':>7} {'Lead':>5} {'FA':>4} {'Whip':>5} {'격차(full)':>9} {'격차(ov)':>8}")
        for name, X in configs:
            reg = hmm_reg(X, n, seed)
            rc, ne, ld, fa, wh, gf = ev(reg, full_idx, n, ret, dd6)
            _, _, _, _, _, gov = ev(reg, ov, n, ret, dd6)
            agg[name].append((rc, ne, ld, fa, wh, gf, gov))
            lds = f"{ld:+.1f}" if not np.isnan(ld) else "n/a"
            print(f"  {name:8} {rc}/{ne:<5} {lds:>5} {fa:>4} {wh:>5} {gf:>+8.2f}p {gov:>+7.2f}p")
    print(f"\n=== seed 평균 (n={len(SEEDS)}) ===")
    print(f"  {'구성':8} {'Recall':>7} {'Lead':>5} {'FA':>5} {'Whip':>6} {'격차(full)':>9} {'격차(ov)':>8}")
    for name in agg:
        a = np.array(agg[name], float)
        print(f"  {name:8} {a[:,0].mean():.1f}/{a[:,1].mean():.0f}  {a[:,2].mean():>+4.1f} {a[:,3].mean():>5.1f} {a[:,4].mean():>6.1f} {a[:,5].mean():>+8.2f}p {a[:,6].mean():>+7.2f}p")
    b, f = np.array(agg["base"], float), np.array(agg["+환율"], float)
    print(f"\n  Δ(+환율 - base) 평균: FA {f[:,3].mean()-b[:,3].mean():+.1f}, Whip {f[:,4].mean()-b[:,4].mean():+.1f}, "
          f"격차(full) {f[:,5].mean()-b[:,5].mean():+.2f}p, 격차(ov) {f[:,6].mean()-b[:,6].mean():+.2f}p")
    print("  판정: FA·Whip↓ & 격차 유지/↑ 가 seed 전반 일관되면 환율 채택. base 안정성(격차 +면 정상)도 함께 확인.")


if __name__ == "__main__":
    main()
