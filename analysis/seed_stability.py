# -*- coding: utf-8 -*-
"""
analysis/seed_stability.py
hsmm_final의 cold-start seed 민감도 검증 (웜스타트가 초기조건을 씻어내는지).
build_features 1회 → seed만 바꿔 walk_forward N회 → 오버레이 MDD/CAGR/Sharpe 분산·pbear 상관·라벨 일치율.
사용: DATABASE_URL 세팅 후  .venv/bin/python analysis/seed_stability.py
"""
import os, sys
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
import numpy as np
import hsmm_final as H
from lib.data import load_strategy

SEEDS = [42, 0, 1, 7, 13, 99, 123, 2024]
BULL = "FCF_YIELD추가전략"


def prior_ym(ym):
    y, m = int(ym[:4]), int(ym[5:7]); m -= 1
    if m == 0: y -= 1; m = 12
    return f"{y:04d}-{m:02d}"


def main():
    bd = load_strategy(BULL, rebal_type="monthly", universe="KOSPI")
    r = bd["results"]
    fret, frd = r["monthly_returns"], r["rebalance_dates"]
    fmonths = [frd[i][:7] for i in range(len(fret))]
    CUT = "2026-07"; fmonths = [y for y in fmonths if y <= CUT]
    fcf = {frd[i][:7]: fret[i] for i in range(len(fret))}

    df, yms, n, _ret, rvol, dvol, _dd6 = H.build_features()   # seed 무관 → 1회만

    def path_for_seed(seed):
        H.SEED = seed
        pbear_raw, start = H.walk_forward(df, yms, n)
        idx = list(range(start, n))
        pbear = pbear_raw.copy()
        for t in idx[1:]:
            pbear[t] = H.PBEAR_EMA * pbear_raw[t] + (1 - H.PBEAR_EMA) * pbear[t - 1]
        # pbear-only 익스포저(채택안) + 리밸밴드
        raw = np.clip(1 - pbear, H.EXP_FLOOR, 1.0)
        exp = raw.copy(); held = None
        for t in idx:
            if held is None or abs(raw[t] - held) >= H.REBAL_BAND:
                held = round(raw[t] / 0.05) * 0.05
            exp[t] = min(max(held, H.EXP_FLOOR), 1.0)
        edic = {yms[t]: exp[t] for t in idx}
        ov = np.array([edic.get(prior_ym(y), 1.0) * fcf[y] for y in fmonths])
        m = H.perf(ov)
        bearcnt = int(sum(1 for t in idx if pbear[t] >= 0.5))
        return pbear[idx[0]:], m, bearcnt, np.array([1 if pbear[t] >= 0.5 else 0 for t in idx])

    res = {}
    for sd in SEEDS:
        pb, m, bc, lbl = path_for_seed(sd)
        res[sd] = dict(pbear=pb, m=m, bear=bc, lbl=lbl)
        print(f"[seed {sd}] MDD {m['mdd']*100:.1f}% CAGR {m['cagr']*100:.1f}% Sharpe {m['sharpe']:.2f} Bear월 {bc}", flush=True)

    base = res[42]["lbl"]; L = min(len(res[s]["pbear"]) for s in SEEDS)
    mdds = np.array([res[s]["m"]["mdd"] * 100 for s in SEEDS])
    cagrs = np.array([res[s]["m"]["cagr"] * 100 for s in SEEDS])
    shps = np.array([res[s]["m"]["sharpe"] for s in SEEDS])
    # pbear 경로 상관 (vs seed42)
    p42 = res[42]["pbear"][:L]
    corrs = [np.corrcoef(res[s]["pbear"][:L], p42)[0, 1] for s in SEEDS if s != 42]
    # Bull/Bear 라벨 일치율 (vs seed42)
    agrees = [np.mean(res[s]["lbl"][:L] == base[:L]) for s in SEEDS if s != 42]

    print(f"\n{'='*66}\n  SEED 안정성 요약 ({len(SEEDS)}개 seed, pbear-only 오버레이)\n{'='*66}")
    print(f"  오버레이 MDD    : {mdds.mean():.1f}% ± {mdds.std():.1f}%  (범위 {mdds.min():.1f} ~ {mdds.max():.1f})")
    print(f"  오버레이 CAGR   : {cagrs.mean():.1f}% ± {cagrs.std():.1f}%  (범위 {cagrs.min():.1f} ~ {cagrs.max():.1f})")
    print(f"  오버레이 Sharpe : {shps.mean():.2f} ± {shps.std():.2f}   (범위 {shps.min():.2f} ~ {shps.max():.2f})")
    print(f"  pbear 경로 상관 (vs seed42): 평균 {np.mean(corrs):.3f}  최소 {np.min(corrs):.3f}")
    print(f"  Bull/Bear 라벨 일치율(vs seed42): 평균 {np.mean(agrees)*100:.1f}%  최소 {np.min(agrees)*100:.1f}%")
    print(f"\n  판정: MDD 표준편차 {mdds.std():.1f}%p · 라벨 일치 {np.mean(agrees)*100:.0f}% → "
          f"{'웜스타트가 seed 초기조건을 씻어냄(안정)' if mdds.std() < 2 and np.mean(agrees) > 0.9 else '재확인 필요(seed 취약 가능)'}")


if __name__ == "__main__":
    main()
