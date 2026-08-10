"""
analysis/hmm_state_centroids.py
hsmm+환율 모델(full-cov HMM)의 두 상태 centroid(피처 평균)와 Bear 라벨링 기준 출력.
전체기간 1회 적합(seed 42), 2004+ (환율 가용구간)으로 centroid 해석 깨끗하게.
사용: DATABASE_URL=<ip> .venv/bin/python analysis/hmm_state_centroids.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
SEED = 42
NAMES = ["Breadth(%)", "ΔBreadth", "NewLow(%)", "ΔNewLow", "Mom(%)", "FX_Δ3m(%)"]


def main():
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
    mends = [last[p] for p in sorted(last) if last[p] >= pd.Timestamp('2004-01-01')]
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1)*100 if len(p) and p.iloc[-1] else np.nan
    F = np.array([[asof(breadth, e)*100, (asof(breadth, e) - asof(breadth, e - timedelta(days=30)))*100,
        asof(newlow, e)*100, (asof(newlow, e) - asof(newlow, e - timedelta(days=30)))*100,
        pct(kospi, e, 30) - pct(kospi, e, 180), pct(usdkrw, e, 90)] for e in mends])
    F = np.nan_to_num(F)

    sc = StandardScaler().fit(F); Z = sc.transform(F)
    hm = GaussianHMM(2, "full", n_iter=60, random_state=SEED); hm.fit(Z)
    mu_z = hm.means_                       # 표준화 단위
    mu_o = sc.inverse_transform(mu_z)      # 원단위
    # Bear 라벨링 점수 (표준화 means, breadth 5개만; FX 제외)
    score = -mu_z[:, 0] - mu_z[:, 1] + mu_z[:, 2] + mu_z[:, 3] - mu_z[:, 4]
    bear = int(np.argmax(score)); bull = 1 - bear

    print(f"적합: full-cov HMM, 2004~2026 {len(F)}개월, seed {SEED}\n")
    print("=== 상태별 centroid (원단위) ===")
    print(f"  {'피처':12} {'State0':>10} {'State1':>10}")
    for j, nm in enumerate(NAMES):
        print(f"  {nm:12} {mu_o[0, j]:>10.2f} {mu_o[1, j]:>10.2f}")
    print("\n=== 상태별 centroid (표준화, 라벨링에 사용) ===")
    print(f"  {'피처':12} {'State0':>10} {'State1':>10}")
    for j, nm in enumerate(NAMES):
        tag = "  ←라벨링 사용" if j < 5 else "  (라벨링 제외)"
        print(f"  {nm:12} {mu_z[0, j]:>+10.2f} {mu_z[1, j]:>+10.2f}{tag}")
    print(f"\n=== Bear 라벨링 ===")
    print(f"  약세점수 = −Breadth −ΔBreadth +NewLow +ΔNewLow −Mom (표준화 means, FX 제외)")
    print(f"  State0 점수 = {score[0]:+.2f},  State1 점수 = {score[1]:+.2f}")
    print(f"  → 점수 높은 쪽 = Bear:  State{bear} = Bear,  State{bull} = Bull")


if __name__ == "__main__":
    main()
