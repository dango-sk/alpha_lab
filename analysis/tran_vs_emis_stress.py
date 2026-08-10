# -*- coding: utf-8 -*-
"""
analysis/tran_vs_emis_stress.py  (실험, production 무수정)

변화율/다이버전스 피처를 어느 채널에 넣느냐 3-arm 비교:
  BASE      : emis=레벨3(breadth·newlow·trend),           stress=base(fx3m·fflow)
  EMIS-aug  : emis=레벨3 + 변화3(Δbreadth·Δnewlow·diverg), stress=base      ← 현 slowbear
  TRAN-aug  : emis=레벨3,                                   stress=base + 변화3  ← 핵심 검증

가설: 같은 신호라도 emission(상태정의)이 아니라 transition(exit hazard 변조)에 넣으면
      slow bear 조기탐지(Lead)는 유지하면서 회복장 과민방어(revert lag)는 줄어든다.

추가 피처: d_breadth=Δ3m, d_newlow=Δ3m, diverg=z(trend)-z(breadth) (지수>breadth=약세발산).
평가: ①2022 P(bear)경로 ②위기 detection(Lead/놓침) ③회복 되돌림 지연 ④FCF오버레이(CAGR/MDD/Bear월).
      ②는 [feedback_regime_eval_metric], ③이 과민방어 직접 지표(신규).

사용: DATABASE_URL 세팅 후  .venv/bin/python analysis/tran_vs_emis_stress.py
"""
import os, sys
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import hsmm_final as H

BASE_EMIS = ["breadth", "newlow", "trend"]
AUG_EMIS = BASE_EMIS + ["d_breadth", "d_newlow", "diverg"]
NAMES = {"2018-10": "2018 하락", "2020-02": "2020 코로나", "2021-11": "2021~22 하락",
         "2022-01": "2022 하락", "2024-08": "2024 급락", "2025-09": "2025 하락"}


def augment(df):
    """slowbear_features.augment와 동일 정의 (변화율·발산)."""
    d = df.copy()
    d["d_breadth"] = df["breadth"].diff(3).fillna(0.0)          # breadth 3M 변화(악화 속도)
    d["d_newlow"] = df["newlow"].diff(3).fillna(0.0)            # 신저가율 3M 변화
    w = 36
    def z(s): return ((s - s.rolling(w, min_periods=12).mean()) / (s.rolling(w, min_periods=12).std() + 1e-9)).fillna(0.0)
    d["diverg"] = (z(df["trend"]) - z(df["breadth"])).clip(-4, 4)   # 지수>breadth = 약세 다이버전스
    return d


def build_stress(dfa):
    """transition stress 스칼라. Bull 이탈 압력↑ = 값↑.
       base = Z(fx3m) - Z(fflow)   (환율↑=스트레스, 외국인유입=완화)  ← H.walk_forward와 동일
       tran = base  - Z(d_breadth) + Z(d_newlow) + Z(diverg)
              (breadth 악화=음수→-Z / 신저가 상승=+Z / 발산=+Z ; 각 Z는 이미 단위분산, 등가중)"""
    cols = ["fx3m", "fflow", "d_breadth", "d_newlow", "diverg"]
    Z = H.roll_z(dfa, cols)
    base = Z["fx3m"].values - Z["fflow"].values
    tran = base - Z["d_breadth"].values + Z["d_newlow"].values + Z["diverg"].values
    return base, tran


def walk_forward_exp(df, yms, n, emis_cols, stress_raw):
    """H.walk_forward를 (emission컬럼·stress벡터) 인자화한 복제본. 엔진 함수는 전부 H 재사용."""
    start = yms.index(H.DECIDE_START) if H.DECIDE_START in yms else 12
    EM = df[emis_cols].values
    pbear = np.full(n, np.nan); params, sc, last_refit = None, None, -10 ** 9
    for t in range(start, n):
        lo = max(0, t + 1 - H.WINDOW_M); Xr = EM[lo:t + 1]; nw = len(Xr)
        w = 0.5 ** ((nw - 1 - np.arange(nw)) / H.HL)               # 시간감쇠
        if params is None or (t - last_refit) >= H.REFIT_EVERY:    # 웜스타트(연1회 재추정)
            sc = StandardScaler().fit(Xr)
            params = H.fit_emission(sc.transform(Xr), w,
                                    H.cold_emission(sc.transform(Xr)) if params is None else params,
                                    40 if params is None else 10)
            last_refit = t
        Xz = sc.transform(Xr)
        means, covs = params['means'], params['covs']
        logB = H.emis_logB(Xz, means, covs)
        bear = int(np.argmax(H.bear_score(means)))
        sw = stress_raw[lo:t + 1]; sw = (sw - sw.mean()) / (sw.std() + H.EPS)   # 창내 표준화
        if H.USE_DURATION:                                         # HSMM(명시적 지속기간) — 채택본
            gamma, _ = H.forward_backward(logB, params['Amat'], params['pi'], w)
            durs = H.state_durations(gamma.argmax(1))
            haz = np.vstack([H.to_hazard(H.dur_pmf(durs[0])), H.to_hazard(H.dur_pmf(durs[1]))])
            filt = H.hsmm_filter(logB, sw, haz, bear, params['pi'])
        else:
            filt = H.plain_hmm_filter(logB, sw, params['Amat'], params['pi'], bear)
        pbear[t] = filt[-1, bear]
    return pbear, start


def to_regime(pbear_raw, start, n):
    """EMA 스무딩 → 히스테리시스 이산레짐 (H와 동일 파라미터)."""
    idx = list(range(start, n)); pb = pbear_raw.copy()
    for t in idx[1:]:
        pb[t] = H.PBEAR_EMA * pbear_raw[t] + (1 - H.PBEAR_EMA) * pb[t - 1]
    reg = ["Bull"] * n; p = "Bull"
    for t in idx:
        p = ("Bear" if pb[t] >= H.T_OUT else "Bull") if p == "Bear" else ("Bear" if pb[t] >= H.T_IN else "Bull")
        reg[t] = p
    return pb, reg, idx


def detect(reg, idx, start, dd6, yms):
    """위기(향후6M -15%↓) 대비 Bear 전환 선행개월. +빠름 / None=놓침."""
    evs = [i for i in idx if dd6[i] <= -15 and (i == start or not (dd6[i - 1] <= -15))]
    onset = [t for t in idx if reg[t] == "Bear" and reg[t - 1] != "Bear"]
    out = {}
    for ev in evs:
        cand = [o for o in onset if abs(o - ev) <= H.WIN]
        out[yms[ev]] = (dd6[ev], (ev - min(cand, key=lambda o: abs(o - ev))) if cand else None)
    return out, evs


def revert_lag(reg, ret, evs, yms):
    """과민방어 지표: 각 위기 저점(trough) 이후 Bull 복귀까지 개월수. 클수록 과민방어.
       trough = 누적지수 최소점(이벤트~+10M). None=데이터 내 미복귀."""
    n = len(reg); P = np.ones(n)
    for t in range(1, n):
        r = ret[t - 1]; P[t] = P[t - 1] * (1 + (r if not np.isnan(r) else 0.0))
    out = {}
    for ev in evs:
        seg = list(range(ev, min(ev + 10, n)))
        tr = min(seg, key=lambda t: P[t])
        rev = next((t for t in range(tr, n) if reg[t] == "Bull"), None)
        out[yms[ev]] = (rev - tr) if rev is not None else None
    return out


def fcf_overlay_mdd(pb, idx, yms):
    """pbear-only 오버레이를 FCF불 전략 월수익에 적용한 CAGR/MDD (현금0). slowbear와 동일."""
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
    eq = np.cumprod(1 + ov); yr = len(ov) / 12
    return eq[-1] ** (1 / yr) - 1, (eq / np.maximum.accumulate(eq) - 1).min()


def main():
    df, yms, n, ret, rvol, dvol, dd6 = H.build_features()
    dfa = augment(df)
    base_stress, tran_stress = build_stress(dfa)
    arms = [("BASE", BASE_EMIS, base_stress),
            ("EMIS-aug", AUG_EMIS, base_stress),
            ("TRAN-aug", BASE_EMIS, tran_stress)]

    R = {}
    for name, emis, stress in arms:
        tag = "base" if stress is base_stress else "base+변화3"
        print(f"\n>>> {name}: emis={emis} | stress={tag}", flush=True)
        pbr, start = walk_forward_exp(dfa, yms, n, emis, stress)
        pb, reg, idx = to_regime(pbr, start, n)
        det, evs = detect(reg, idx, start, dd6, yms)
        lag = revert_lag(reg, ret, evs, yms)
        cagr, mdd = fcf_overlay_mdd(pb, idx, yms)
        R[name] = dict(pb=pb, reg=reg, idx=idx, det=det, lag=lag, cagr=cagr, mdd=mdd,
                       bearm=sum(1 for t in idx if reg[t] == "Bear"))

    anames = [a[0] for a in arms]

    # ① 2021-01~2023-06 P(bear) 경로 + 레짐
    print(f"\n{'='*70}\n① 2021-01~2023-06 P(bear) 경로 (2022 slow bear)\n{'='*70}")
    print(f"  {'월':9}" + "".join(f"{a:>20}" for a in anames))
    for t in range(n):
        if "2021-01" <= yms[t] <= "2023-06":
            cells = "".join(f"{R[a]['pb'][t]:>12.2f} {R[a]['reg'][t]:>7}" for a in anames)
            print(f"  {yms[t]:9}{cells}")

    # ② 위기 detection (Lead)
    allev = sorted(set().union(*[set(R[a]['det']) for a in anames]))
    print(f"\n{'='*70}\n② 위기 detection (값=선행개월 +빠름, X=놓침)\n{'='*70}")
    print(f"  {'이벤트':14}{'낙폭':>7}" + "".join(f"{a:>12}" for a in anames))
    for ev in allev:
        dd = next((R[a]['det'][ev][0] for a in anames if ev in R[a]['det']), 0.0)
        def fl(a):
            v = R[a]['det'].get(ev)
            return "X" if (v is None or v[1] is None) else f"{v[1]:+d}m"
        print(f"  {NAMES.get(ev, ev):14}{dd:>6.0f}%" + "".join(f"{fl(a):>12}" for a in anames))

    # ③ 회복 되돌림 지연 (과민방어 직접 지표)
    print(f"\n{'='*70}\n③ 회복 되돌림 지연 = 저점 후 Bull복귀 개월 (클수록 과민방어)\n{'='*70}")
    print(f"  {'이벤트':14}" + "".join(f"{a:>12}" for a in anames))
    for ev in allev:
        def fl(a):
            v = R[a]['lag'].get(ev)
            return "미복귀" if v is None else f"{v}m"
        print(f"  {NAMES.get(ev, ev):14}" + "".join(f"{fl(a):>12}" for a in anames))

    # ④ FCF 오버레이 성과 (안 깨지는지)
    print(f"\n{'='*70}\n④ FCF 오버레이(pbear-only) 성과\n{'='*70}")
    print(f"  {'arm':10}{'CAGR':>9}{'MDD':>10}{'Bear월':>9}")
    for a in anames:
        print(f"  {a:10}{R[a]['cagr']*100:>8.1f}%{R[a]['mdd']*100:>9.1f}%{R[a]['bearm']:>8}")

    print(f"\n  판정: TRAN-aug 성공 = ②Lead가 EMIS-aug만큼 조기 + ③revert lag이 BASE수준으로 짧음 + ④CAGR이 BASE 근처.")

    # 경로 CSV 저장
    out = pd.DataFrame({"ym": yms})
    for a in anames:
        out[f"pbear_{a}"] = R[a]["pb"]; out[f"reg_{a}"] = R[a]["reg"]
    out.to_csv(REPO / "analysis" / "tran_vs_emis_path.csv", index=False, encoding="utf-8-sig")
    print(f"\n  경로 저장 → analysis/tran_vs_emis_path.csv")


if __name__ == "__main__":
    main()
