# -*- coding: utf-8 -*-
"""
analysis/slowbear_features.py  (실험, production 미수정)
현재 emission(레벨만: breadth·newlow·trend) vs 증강(+Δbreadth·Δ신저가·다이버전스) 비교.
목적: 2022형 slow bear를 더 빨리 잡는지 (P(bear) 경로·선행개월), 다른 위기 안 깨지는지, FCF 낙폭.
사용: DATABASE_URL 세팅 후 .venv/bin/python analysis/slowbear_features.py
"""
import os, sys
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd
import hsmm_final as H

BASE = ["breadth", "newlow", "trend"]
AUG = BASE + ["d_breadth", "d_newlow", "diverg"]


def augment(df):
    d = df.copy()
    d["d_breadth"] = df["breadth"].diff(3).fillna(0.0)          # 3개월 변화(악화 속도)
    d["d_newlow"] = df["newlow"].diff(3).fillna(0.0)
    w = 36
    def z(s): return ((s - s.rolling(w, min_periods=12).mean()) / (s.rolling(w, min_periods=12).std() + 1e-9)).fillna(0.0)
    d["diverg"] = (z(df["trend"]) - z(df["breadth"])).clip(-4, 4)   # 지수>breadth = 약세 다이버전스
    return d


def run(emis, dframe, yms, n):
    old = H.EMIS_COLS
    H.EMIS_COLS = emis
    try:
        pbear_raw, start = H.walk_forward(dframe, yms, n)
    finally:
        H.EMIS_COLS = old
    idx = list(range(start, n)); pb = pbear_raw.copy()
    for t in idx[1:]:
        pb[t] = H.PBEAR_EMA * pbear_raw[t] + (1 - H.PBEAR_EMA) * pb[t - 1]
    reg = ["Bull"] * n; p = "Bull"
    for t in idx:
        p = ("Bear" if pb[t] >= H.T_OUT else "Bull") if p == "Bear" else ("Bear" if pb[t] >= H.T_IN else "Bull")
        reg[t] = p
    return pb, reg, start, idx


def fcf_overlay_mdd(pb, reg, idx, yms):
    """pbear-only 오버레이를 FCF 전략에 적용한 MDD/CAGR (현금0)."""
    s = pd.read_csv(REPO / "analysis" / "fcf_overlay_series.csv")
    fmonths = list(s["ym"]); fcf = dict(zip(s["ym"], s["bench"]))
    raw = np.clip(1 - pb, H.EXP_FLOOR, 1.0); exp = raw.copy(); held = None
    for t in idx:
        if held is None or abs(raw[t] - held) >= H.REBAL_BAND:
            held = round(raw[t] / 0.05) * 0.05
        exp[t] = min(max(held, H.EXP_FLOOR), 1.0)
    edic = {yms[t]: exp[t] for t in idx}
    def pm(y):
        yy, mm = int(y[:4]), int(y[5:7]); mm -= 1
        if mm == 0: yy -= 1; mm = 12
        return f"{yy:04d}-{mm:02d}"
    ov = np.array([edic.get(pm(y), 1.0) * fcf[y] for y in fmonths])
    eq = np.cumprod(1 + ov); y = len(ov) / 12
    return eq[-1] ** (1 / y) - 1, (eq / np.maximum.accumulate(eq) - 1).min()


def detect(reg, idx, start, dd6, yms):
    evs = [i for i in idx if dd6[i] <= -15 and (i == start or not (dd6[i - 1] <= -15))]
    onset = [t for t in idx if reg[t] == "Bear" and reg[t - 1] != "Bear"]
    out = {}
    for ev in evs:
        cand = [o for o in onset if abs(o - ev) <= H.WIN]
        out[yms[ev]] = (dd6[ev], (ev - min(cand, key=lambda o: abs(o - ev))) if cand else None)
    return out


def main():
    df, yms, n, ret, rvol, dvol, dd6 = H.build_features()
    dfa = augment(df)
    print("피처:", BASE, "→", AUG, flush=True)
    pbB, regB, stB, idxB = run(BASE, df, yms, n)
    pbA, regA, stA, idxA = run(AUG, dfa, yms, n)

    # 1) 2021~2023 P(bear) 경로
    print(f"\n=== 2021-01 ~ 2023-06 P(bear) 경로 (2022 slow bear) ===")
    print(f"  {'월':9}{'baseline':>10}{'augmented':>11}{'  레짐B/A':>12}")
    for t in range(n):
        if "2021-01" <= yms[t] <= "2023-06":
            print(f"  {yms[t]:9}{pbB[t]:>10.2f}{pbA[t]:>11.2f}    {regB[t]:>4}/{regA[t]:<4}")

    # 2) 위기 detection 비교
    dB, dA = detect(regB, idxB, stB, dd6, yms), detect(regA, idxA, stA, dd6, yms)
    NAMES = {"2018-10": "2018 하락", "2020-02": "2020 코로나", "2021-11": "2021~22 하락",
             "2022-01": "2022 하락", "2024-08": "2024 급락", "2025-09": "2025 하락"}
    print(f"\n=== 위기 detection 비교 (값=선행개월 +빠름, X=놓침) ===")
    print(f"  {'이벤트':14}{'낙폭':>7}{'baseline':>11}{'augmented':>11}")
    for ev in sorted(set(dB) | set(dA)):
        dd, lb = dB.get(ev, (dd6[yms.index(ev)] if ev in yms else 0, None))
        _, la = dA.get(ev, (0, None))
        def fmt(l): return f"{l:+d}개월" if l is not None else "X(놓침)"
        print(f"  {NAMES.get(ev, ev):14}{dd:>6.0f}%{fmt(lb):>11}{fmt(la):>11}")

    # 3) FCF 오버레이 성과 (안 깨지는지)
    cB, mB = fcf_overlay_mdd(pbB, regB, idxB, yms)
    cA, mA = fcf_overlay_mdd(pbA, regA, idxA, yms)
    bearB = sum(1 for t in idxB if regB[t] == "Bear"); bearA = sum(1 for t in idxA if regA[t] == "Bear")
    print(f"\n=== FCF 오버레이(pbear-only) 전체 성과 ===")
    print(f"  {'':10}{'CAGR':>8}{'MDD':>9}{'Bear월수':>10}")
    print(f"  {'baseline':10}{cB*100:>7.1f}%{mB*100:>8.1f}%{bearB:>8}")
    print(f"  {'augmented':10}{cA*100:>7.1f}%{mA*100:>8.1f}%{bearA:>8}")


if __name__ == "__main__":
    main()
