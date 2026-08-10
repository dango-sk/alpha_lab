# -*- coding: utf-8 -*-
"""
analysis/recovery_overlay.py  (실험, production 무수정)

목적: FULL_EMIS HSMM(P(bear)·duration·EMA·state)은 그대로 두고, **가격 급회복이 확인될 때만**
      최종 exposure를 floor로 선복구 → 2020 V반등 손실을 줄이되 2022 가짜반등·2025~26 cascade에서
      false recovery를 만들지 않는지 동시 검증. "price overlay가 정답"이라 가정하지 않음 — 나쁘면 기각.

무래앞: 모든 score는 t시점까지의 P/breadth/trend로 계산 → t→t+1 수익에 적용.
        true/false recovery 라벨은 **사후평가 전용**(신호에 안 들어감).
threshold: 사후최적화 X, 경제적 후보 소수 고정 (200MA+5%, 3M+15%, 1M+8%, Δ3M breadth+0.25).

결합: final_exp = max(base_exp, floor),  floor = FLOOR_MAX·score,  해제 = max(floor, REL·직전floor).
      기존 exposure를 대체하지 않고 '바닥만' 올림.

arm: FULL_BASE / TREND_LEVEL(진단용) / TREND_CROSS / PRICE_MOM / TREND_PLUS_MOM / PRICE_PLUS_THRUST.
     + TREND_PLUS_MOM에 floor(0.4/0.6/0.8)·해제(즉시/감쇠/하드) 소수 민감도.

사용: DATABASE_URL 세팅 후  .venv/bin/python analysis/recovery_overlay.py
"""
import os, sys
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import hsmm_final as H

EMIS = ["breadth", "newlow", "trend", "d_breadth", "d_newlow", "diverg"]
FLOOR, BAND = H.EXP_FLOOR, H.REBAL_BAND
# 경제적 스케일(고정, 사후최적화 아님)
T0, M3, M1, TB = 0.05, 0.15, 0.08, 0.25   # 200MA+5% / 3M+15% / 1M+8% / Δ3M breadth+0.25
FLOOR_MAX, REL = 0.6, 0.5                   # 기본 floor 상한 / 해제 감쇠(1~2M)
PERIODS = [("2020 V반등", "2020-02", "2021-01"), ("2021 slow-grind 방어", "2021-06", "2022-02"),
           ("2022 bear rally/본체", "2022-03", "2022-12"), ("2025~26 cascade", "2025-01", "2026-07")]
DETAIL = ["TREND_CROSS", "PRICE_MOM", "TREND_PLUS_MOM", "PRICE_PLUS_THRUST"]


def augment(df):
    d = df.copy()
    d["d_breadth"] = df["breadth"].diff(3).fillna(0.0)
    d["d_newlow"] = df["newlow"].diff(3).fillna(0.0)
    w = 36
    def z(s): return ((s - s.rolling(w, min_periods=12).mean()) / (s.rolling(w, min_periods=12).std() + 1e-9)).fillna(0.0)
    d["diverg"] = (z(df["trend"]) - z(df["breadth"])).clip(-4, 4)
    return d


def base_stress(df):
    Z = H.roll_z(df, ["fx3m", "fflow"]); return Z["fx3m"].values - Z["fflow"].values


def shift_ym(y, k):
    yy, mm = int(y[:4]), int(y[5:7]); mm += k
    yy += (mm - 1) // 12; mm = (mm - 1) % 12 + 1
    return f"{yy:04d}-{mm:02d}"


def walk_forward_full(df, yms, n, stress):
    """FULL_EMIS pbear (production과 동일 파이프라인). 여기서만 HSMM 1회 실행."""
    start = yms.index(H.DECIDE_START) if H.DECIDE_START in yms else 12
    EM = df[EMIS].values; pbear = np.full(n, np.nan); params, sc, last_refit = None, None, -10 ** 9
    for t in range(start, n):
        lo = max(0, t + 1 - H.WINDOW_M); Xr = EM[lo:t + 1]; nw = len(Xr)
        w = 0.5 ** ((nw - 1 - np.arange(nw)) / H.HL)
        if params is None or (t - last_refit) >= H.REFIT_EVERY:
            sc = StandardScaler().fit(Xr)
            params = H.fit_emission(sc.transform(Xr), w,
                                    H.cold_emission(sc.transform(Xr)) if params is None else params,
                                    40 if params is None else 10)
            last_refit = t
        Xz = sc.transform(Xr); means, covs = params['means'], params['covs']
        logB = H.emis_logB(Xz, means, covs); bear = int(np.argmax(H.bear_score(means)))
        sw = stress[lo:t + 1]; sw = (sw - sw.mean()) / (sw.std() + H.EPS)
        gamma, _ = H.forward_backward(logB, params['Amat'], params['pi'], w)
        durs = H.state_durations(gamma.argmax(1))
        haz = np.vstack([H.to_hazard(H.dur_pmf(durs[0])), H.to_hazard(H.dur_pmf(durs[1]))])
        filt = H.hsmm_filter(logB, sw, haz, bear, params['pi']); pbear[t] = filt[-1, bear]
    return pbear, start


def base_exposure(pbear, start, n):
    idx = list(range(start, n)); pb = pbear.copy()
    for t in idx[1:]:
        pb[t] = H.PBEAR_EMA * pbear[t] + (1 - H.PBEAR_EMA) * pb[t - 1]
    exp = np.ones(n); raw = np.clip(1 - pb, FLOOR, 1.0); held = None
    for t in idx:
        if held is None or abs(raw[t] - held) >= BAND: held = round(raw[t] / 0.05) * 0.05
        exp[t] = min(max(held, FLOOR), 1.0)
    return pb, exp


def build_scores(df, P, n):
    """모든 recovery score = 0~1 연속. t까지 정보만."""
    trend = df["trend"].values; dbre = df["d_breadth"].values
    mom1 = np.array([(P[t] / P[t - 1] - 1) if t >= 1 else 0.0 for t in range(n)])
    mom3 = np.array([(P[t] / P[t - 3] - 1) if t >= 3 else 0.0 for t in range(n)])
    s = {"FULL_BASE": np.zeros(n),
         "TREND_LEVEL": np.clip(trend / T0, 0, 1)}                       # level → 상시 발동(진단)
    cross = np.zeros(n); active = False                                  # neg→pos 전환 후에만
    for t in range(1, n):
        if trend[t] > 0 and trend[t - 1] <= 0: active = True
        if trend[t] <= 0: active = False
        cross[t] = np.clip(trend[t] / T0, 0, 1) if active else 0.0
    s["TREND_CROSS"] = cross
    s["PRICE_MOM"] = np.clip(mom3 / M3, 0, 1)                            # 3M 반등강도
    s["TREND_PLUS_MOM"] = np.clip(mom3 / M3, 0, 1) * (trend > 0)         # 중기추세 회복 × 단기반등
    s["PRICE_PLUS_THRUST"] = np.clip(mom1 / M1, 0, 1) * np.clip(dbre / TB, 0, 1)  # 급반등 × breadth 가속
    return s, mom1, mom3


def overlay(base_exp, score, start, n, fmax=FLOOR_MAX, rel=REL):
    floor = np.zeros(n); final = base_exp.copy(); prev = 0.0
    for t in range(start, n):
        fe = max(fmax * score[t], rel * prev); floor[t] = fe; prev = fe
        final[t] = max(base_exp[t], fe)
    return final, floor


def perf(exp, yms, fmonths, fcf):
    ebym = {yms[t]: exp[t] for t in range(len(yms))}
    ov = np.array([ebym.get(shift_ym(y, -1), 1.0) * fcf[y] for y in fmonths])
    eq = np.cumprod(1 + ov); yr = len(ov) / 12; cagr = eq[-1] ** (1 / yr) - 1
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(cagr=cagr, mdd=mdd, sharpe=(ov.mean() * 12) / (ov.std() * np.sqrt(12) + 1e-12),
                calmar=cagr / abs(mdd) if mdd else float('nan'))


def main():
    df0, yms, n, ret, rvol, dvol, dd6 = H.build_features()
    df = augment(df0); stress = base_stress(df)
    s = pd.read_csv(REPO / "analysis" / "fcf_overlay_series.csv")
    fmonths, fcf = list(s["ym"]), dict(zip(s["ym"], s["bench"]))
    fcf_next = np.array([fcf.get(shift_ym(yms[t], 1), np.nan) for t in range(n)])
    P = np.ones(n)
    for t in range(1, n): P[t] = P[t - 1] * (1 + (ret[t - 1] if not np.isnan(ret[t - 1]) else 0.0))
    fwd3 = np.array([(P[t + 3] / P[t] - 1) if t + 3 < n else np.nan for t in range(n)])  # 사후평가용

    pbear_raw, start = walk_forward_full(df, yms, n, stress)
    pbear, base_exp = base_exposure(pbear_raw, start, n)
    scores, mom1, mom3 = build_scores(df, P, n)

    ARMS = list(scores.keys())
    A = {}
    for a in ARMS:
        final, floor = overlay(base_exp, scores[a], start, n)
        A[a] = dict(final=final, floor=floor, score=scores[a], **perf(final, yms, fmonths, fcf))

    def idxwin(a, b): return [t for t in range(n) if a <= yms[t] <= b]

    # ═══ 전체 성과 ═══
    print(f"\n{'='*92}\n[1] 전체 성과 (FCF 오버레이, floor_max={FLOOR_MAX}, 해제감쇠={REL})\n{'='*92}")
    print(f"  {'arm':20}{'CAGR':>8}{'MDD':>9}{'Sharpe':>8}{'Calmar':>8}{'평균exp':>8}{'turn':>7}{'발동月':>7}{'평균지속':>8}")
    for a in ARMS:
        final = A[a]['final']; act = [t for t in range(start, n) if final[t] - base_exp[t] > 0.02]
        turn = sum(abs(final[t] - final[t - 1]) for t in range(start + 1, n))
        runs, r = [], 0
        for t in range(start, n):
            if t in act: r += 1
            elif r: runs.append(r); r = 0
        if r: runs.append(r)
        print(f"  {a:20}{A[a]['cagr']*100:>7.1f}%{A[a]['mdd']*100:>8.1f}%{A[a]['sharpe']:>8.2f}"
              f"{A[a]['calmar']:>8.2f}{np.mean(final[start:]):>8.2f}{turn:>7.1f}{len(act):>7}"
              f"{(np.mean(runs) if runs else 0):>8.1f}")

    # ═══ 회복 성능 & 방어 훼손 ═══
    tr = yms.index("2020-03"); m2020 = idxwin("2020-04", "2020-12")
    m2021 = idxwin("2021-06", "2022-02"); m2022 = idxwin("2022-03", "2022-12"); mcas = idxwin("2025-01", "2026-07")
    def first_ge(exp, thr):
        t = next((t for t in range(tr, n) if exp[t] >= thr), None); return (t - tr) if t is not None else None
    miss_base = sum((1 - base_exp[t]) * (fcf_next[t] if not np.isnan(fcf_next[t]) else 0) for t in m2020)
    def_base = sum((1 - base_exp[t]) * (-(fcf_next[t] if not np.isnan(fcf_next[t]) else 0)) for t in m2021)
    print(f"\n{'='*92}\n[2] 회복 성능 / 방어 훼손 (FULL_BASE 대비)\n{'='*92}")
    print(f"  {'arm':20}{'→0.5':>6}{'→0.8':>6}{'2020회복률':>10}{'2020ΔCAGR':>10}"
          f"{'2021반납':>9}{'2022가짜손':>10}{'25-26whip':>10}{'falseRec%':>10}")
    for a in ARMS:
        final = A[a]['final']
        miss_a = sum((1 - final[t]) * (fcf_next[t] if not np.isnan(fcf_next[t]) else 0) for t in m2020)
        rec_ratio = (miss_base - miss_a) / miss_base if abs(miss_base) > 1e-9 else 0.0
        def_a = sum((1 - final[t]) * (-(fcf_next[t] if not np.isnan(fcf_next[t]) else 0)) for t in m2021)
        f22 = sum((final[t] - base_exp[t]) * (fcf_next[t] if not np.isnan(fcf_next[t]) else 0) for t in m2022 if (final[t] - base_exp[t]) > 0.02 and fcf_next[t] < 0)
        fcas = sum((final[t] - base_exp[t]) * (fcf_next[t] if not np.isnan(fcf_next[t]) else 0) for t in mcas if (final[t] - base_exp[t]) > 0.02 and fcf_next[t] < 0)
        act = [t for t in range(start, n) if final[t] - base_exp[t] > 0.02]
        fr = (sum(1 for t in act if not np.isnan(fwd3[t]) and fwd3[t] < 0) / len(act)) if act else 0.0
        print(f"  {a:20}{str(first_ge(final,0.5))+'m':>6}{str(first_ge(final,0.8))+'m':>6}"
              f"{rec_ratio*100:>9.0f}%{(A[a]['cagr']-A['FULL_BASE']['cagr'])*100:>+9.1f}p"
              f"{(def_base-def_a)*100:>8.1f}%{f22*100:>9.1f}%{fcas*100:>9.1f}%{fr*100:>9.0f}%")
    print("  ※ 2020회복률=놓친상승 회복비율 / 2021반납=slow-grind 방어이익 감소 / 2022·25-26=가짜반등 재진입 손실")

    # ═══ 구간별 월별 attribution ═══
    for label, a0, a1 in PERIODS:
        wl = idxwin(a0, a1)
        print(f"\n{'='*92}\n[3] {label} ({a0}~{a1})\n{'='*92}")
        print(f"  {'월':8}{'KOSPI':>7}{'FCF+1':>7}{'pbear':>7}{'baseE':>7}", end="")
        for a in DETAIL: print(f"{a[:10]+'F':>12}", end="")
        print()
        for t in wl:
            rk = f"{ret[t]*100:>6.1f}" if not np.isnan(ret[t]) else "   n/a"
            rf = f"{fcf_next[t]*100:>6.1f}" if not np.isnan(fcf_next[t]) else "   n/a"
            print(f"  {yms[t]:8}{rk}{rf}{pbear[t]:>7.2f}{base_exp[t]:>7.2f}", end="")
            for a in DETAIL:
                fin = A[a]['final'][t]; act = fin - base_exp[t] > 0.02
                tf = "" if not act else ("T" if (not np.isnan(fwd3[t]) and fwd3[t] > 0) else "F")
                print(f"{fin:>10.2f}{tf:>2}", end="")
            print()
        # 구간 기여 합
        print(f"  {'[구간기여Δ]':8}{'':>28}", end="")
        for a in DETAIL:
            c = sum((A[a]['final'][t] - base_exp[t]) * (fcf_next[t] if not np.isnan(fcf_next[t]) else 0) for t in wl)
            print(f"{c*100:>10.1f}% ", end="")
        print("\n  ※ 각 arm열=최종exposure, T/F=발동시 사후 fwd3M(+/-). [구간기여]=Σ(final-base)·FCF")

    # ═══ 민감도 (TREND_PLUS_MOM: floor·해제) ═══
    print(f"\n{'='*92}\n[4] 민감도 — TREND_PLUS_MOM (floor·해제, 소수만)\n{'='*92}")
    print(f"  {'설정':22}{'CAGR':>8}{'MDD':>9}{'2020회복률':>10}{'2022가짜손':>10}")
    sc = scores["TREND_PLUS_MOM"]
    for fmax in (0.4, 0.6, 0.8):
        for rel, tag in ((0.0, "즉시"), (0.5, "감쇠")):
            fin, _ = overlay(base_exp, sc, start, n, fmax=fmax, rel=rel)
            pm = perf(fin, yms, fmonths, fcf)
            miss_a = sum((1 - fin[t]) * (fcf_next[t] if not np.isnan(fcf_next[t]) else 0) for t in m2020)
            rr = (miss_base - miss_a) / miss_base if abs(miss_base) > 1e-9 else 0
            f22 = sum((fin[t] - base_exp[t]) * (fcf_next[t] if not np.isnan(fcf_next[t]) else 0) for t in m2022 if (fin[t] - base_exp[t]) > 0.02 and fcf_next[t] < 0)
            print(f"  floor{fmax}/{tag:14}{pm['cagr']*100:>7.1f}%{pm['mdd']*100:>8.1f}%{rr*100:>9.0f}%{f22*100:>9.1f}%")
    # 하드스위치 대조 (연속 vs 0/1)
    hard = (sc > 0.33).astype(float)   # score>0.33 ≈ 3M +5%↑ & trend>0
    fin_h, _ = overlay(base_exp, hard, start, n)
    pm = perf(fin_h, yms, fmonths, fcf)
    print(f"  {'하드스위치(0/1)':22}{pm['cagr']*100:>7.1f}%{pm['mdd']*100:>8.1f}%")

    print(f"\n{'='*92}\n[5] 최종 판정 질문 (위 표로 답):\n{'='*92}")
    print("  Q1 2020손실 회복? = [2] 2020회복률·2020ΔCAGR")
    print("  Q2 2021 MDD 반납? = [2] 2021반납 + [1] MDD 변화")
    print("  Q3 2022 false recovery? = [2] 2022가짜손·falseRec% + [3]2022표 F표시")
    print("  Q4 어느 신호 안정? = falseRec% 낮고 2020회복률 높은 arm")
    print("  Q5 hard vs continuous? = [4] turnover·MDD")
    print("  Q6 특정이벤트 의존? = [3] 2020 외 2025~26에서도 +기여 반복되나")
    print("  Q7 구조적 정당 vs 사후최적? = 스케일 고정·무래앞이나, [2]falseRec 크면 기각")

    out = pd.DataFrame({"ym": yms, "kospi": ret, "fcf_next": fcf_next, "pbear": pbear,
                        "base_exp": base_exp, "trend": df["trend"].values, "mom3": mom3, "d_breadth": df["d_breadth"].values})
    for a in ARMS:
        out[f"score_{a}"] = A[a]["score"]; out[f"final_{a}"] = A[a]["final"]
    out.to_csv(REPO / "analysis" / "recovery_overlay_path.csv", index=False, encoding="utf-8-sig")
    print(f"\n  경로 저장 → analysis/recovery_overlay_path.csv")


if __name__ == "__main__":
    main()
