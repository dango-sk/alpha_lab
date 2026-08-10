# -*- coding: utf-8 -*-
"""
analysis/overstay_mechanism.py  (진단, production 무수정)

FULL_EMIS의 2020 V반등 over-stay 원인 분해: emission vs duration vs EMA.
핵심: emission 적합(means/covs)은 duration·EMA와 무관 → logB(Bull/Bear)는 4-arm 공유.
      duration filter(HSMM) vs plain filter, EMA 0.5 vs 1.0 만 다름 → 3요인 clean 분리.

4-arm (emission=FULL_EMIS 6피처 공통, transition=base stress 공통):
  FULL         duration ON,  EMA 0.5
  NO_DURATION  duration OFF, EMA 0.5
  NO_EMA       duration ON,  EMA 1.0
  NO_BOTH      duration OFF, EMA 1.0

사용: DATABASE_URL 세팅 후  .venv/bin/python analysis/overstay_mechanism.py
"""
import os, sys
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import hsmm_final as H

EMIS = ["breadth", "newlow", "trend", "d_breadth", "d_newlow", "diverg"]
FLOOR, BAND, T_IN, T_OUT = H.EXP_FLOOR, H.REBAL_BAND, H.T_IN, H.T_OUT
W_LO, W_HI = "2020-02", "2021-01"   # 상세 표 구간


def augment(df):
    d = df.copy()
    d["d_breadth"] = df["breadth"].diff(3).fillna(0.0)
    d["d_newlow"] = df["newlow"].diff(3).fillna(0.0)
    w = 36
    def z(s): return ((s - s.rolling(w, min_periods=12).mean()) / (s.rolling(w, min_periods=12).std() + 1e-9)).fillna(0.0)
    d["diverg"] = (z(df["trend"]) - z(df["breadth"])).clip(-4, 4)
    return d


def base_stress(df):
    Z = H.roll_z(df, ["fx3m", "fflow"])
    return Z["fx3m"].values - Z["fflow"].values


def shift_ym(y, k):
    yy, mm = int(y[:4]), int(y[5:7]); mm += k
    yy += (mm - 1) // 12; mm = (mm - 1) % 12 + 1
    return f"{yy:04d}-{mm:02d}"


def hsmm_filter_capture(logB, stress, haz, bear, pi):
    """H.hsmm_filter 충실 복제 + 마지막달 확장상태 질량 a(2,D) 반환."""
    T = logB.shape[0]; D = haz.shape[1]; bull = 1 - bear
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    haz_l = np.log(haz / (1 - haz))
    filt = np.zeros((T, 2)); a = np.zeros((2, D)); a[:, 0] = pi * B[0]; a /= a.sum() + H.EPS; filt[0] = a.sum(1)
    for t in range(1, T):
        z = stress[t]; he = np.empty_like(haz)
        he[bull] = 1 / (1 + np.exp(-(haz_l[bull] + H.KAPPA * z)))
        he[bear] = 1 / (1 + np.exp(-(haz_l[bear] - H.KAPPA * z)))
        he = np.clip(he, 1e-6, 1 - 1e-6); he[:, -1] = 1.0
        cont = a * (1 - he); endm = (a * he).sum(1)
        nxt = np.zeros((2, D)); nxt[:, 1:] = cont[:, :-1]
        nxt[bull, 0] = endm[bear]; nxt[bear, 0] = endm[bull]
        nxt = nxt * B[t][:, None]; nxt /= nxt.sum() + H.EPS
        a = nxt; filt[t] = a.sum(1)
    return filt, a


def run_shared(df, yms, n, stress):
    """emission 1회 적합(공유) → 같은 logB에 duration/plain 두 필터 동시 실행."""
    start = yms.index(H.DECIDE_START) if H.DECIDE_START in yms else 12
    EM = df[EMIS].values
    llb = np.full(n, np.nan); llr = np.full(n, np.nan)          # logB Bull / Bear (당월)
    raw_dur = np.full(n, np.nan); raw_plain = np.full(n, np.nan)
    ed_bear = np.full(n, np.nan); frac_long = np.full(n, np.nan)  # E[경과d|Bear], d>=6 질량비
    params, sc, last_refit = None, None, -10 ** 9
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
        logB = H.emis_logB(Xz, means, covs)
        bear = int(np.argmax(H.bear_score(means))); bull = 1 - bear
        llb[t] = logB[-1, bull]; llr[t] = logB[-1, bear]
        sw = stress[lo:t + 1]; sw = (sw - sw.mean()) / (sw.std() + H.EPS)
        # duration filter
        gamma, _ = H.forward_backward(logB, params['Amat'], params['pi'], w)
        durs = H.state_durations(gamma.argmax(1))
        haz = np.vstack([H.to_hazard(H.dur_pmf(durs[0])), H.to_hazard(H.dur_pmf(durs[1]))])
        filt_d, a_fin = hsmm_filter_capture(logB, sw, haz, bear, params['pi'])
        raw_dur[t] = filt_d[-1, bear]
        bm = a_fin[bear]; tot = bm.sum() + H.EPS
        ed_bear[t] = ((np.arange(1, len(bm) + 1)) * bm).sum() / tot
        frac_long[t] = bm[5:].sum() / tot
        # plain filter (지속기간 제약 없음)
        filt_p = H.plain_hmm_filter(logB, sw, params['Amat'], params['pi'], bear)
        raw_plain[t] = filt_p[-1, bear]
    return dict(start=start, llb=llb, llr=llr, raw_dur=raw_dur, raw_plain=raw_plain,
                ed_bear=ed_bear, frac_long=frac_long)


def smooth_exposure(raw, start, n, ema):
    """EMA(1.0=끔) → pbear-only 익스포저(리밸밴드) + 히스테리시스 이산레짐."""
    idx = list(range(start, n)); pb = raw.copy()
    for t in idx[1:]:
        pb[t] = ema * raw[t] + (1 - ema) * pb[t - 1]
    exp = np.ones(n); rw = np.clip(1 - pb, FLOOR, 1.0); held = None
    for t in idx:
        if held is None or abs(rw[t] - held) >= BAND: held = round(rw[t] / 0.05) * 0.05
        exp[t] = min(max(held, FLOOR), 1.0)
    reg = ["Bull"] * n; p = "Bull"
    for t in idx:
        p = ("Bear" if pb[t] >= T_OUT else "Bull") if p == "Bear" else ("Bear" if pb[t] >= T_IN else "Bull")
        reg[t] = p
    return pb, exp, reg


def overlay_perf(exp, yms, fmonths, fcf):
    ebym = {yms[t]: exp[t] for t in range(len(yms))}
    ov = np.array([ebym.get(shift_ym(y, -1), 1.0) * fcf[y] for y in fmonths])
    eq = np.cumprod(1 + ov); yr = len(ov) / 12
    return dict(cagr=eq[-1] ** (1 / yr) - 1, mdd=(eq / np.maximum.accumulate(eq) - 1).min(),
                sharpe=(ov.mean() * 12) / (ov.std() * np.sqrt(12) + 1e-12))


def main():
    df0, yms, n, ret, rvol, dvol, dd6 = H.build_features()
    df = augment(df0); stress = base_stress(df)
    s = pd.read_csv(REPO / "analysis" / "fcf_overlay_series.csv")
    fmonths, fcf = list(s["ym"]), dict(zip(s["ym"], s["bench"]))
    fcf_next = np.array([fcf.get(shift_ym(yms[t], 1), np.nan) for t in range(n)])

    S = run_shared(df, yms, n, stress); start = S["start"]
    ARMS = [("FULL", "raw_dur", 0.5), ("NO_DURATION", "raw_plain", 0.5),
            ("NO_EMA", "raw_dur", 1.0), ("NO_BOTH", "raw_plain", 1.0)]
    A = {}
    for name, rawkey, ema in ARMS:
        pb, exp, reg = smooth_exposure(S[rawkey], start, n, ema)
        A[name] = dict(pb=pb, exp=exp, reg=reg, raw=S[rawkey], ema=ema, **overlay_perf(exp, yms, fmonths, fcf))
    anames = [a[0] for a in ARMS]
    lldiff = S["llb"] - S["llr"]     # + = Bull emission 우세

    def win(a, b):
        return [t for t in range(n) if a <= yms[t] <= b]
    wl = win(W_LO, W_HI)

    # ── 표 A: 공유 emission + raw pbear (duration vs plain) ──
    print(f"\n{'='*88}\n[A] 공유 emission likelihood + raw P(bear)  ({W_LO}~{W_HI})\n{'='*88}")
    print(f"  {'월':8}{'logB_Bull':>10}{'logB_Bear':>10}{'lldiff':>8}{'raw_dur':>9}{'raw_plain':>10}{'E[d|Bear]':>10}{'d>=6비':>8}")
    for t in wl:
        print(f"  {yms[t]:8}{S['llb'][t]:>10.2f}{S['llr'][t]:>10.2f}{lldiff[t]:>8.2f}"
              f"{S['raw_dur'][t]:>9.2f}{S['raw_plain'][t]:>10.2f}{S['ed_bear'][t]:>10.1f}{S['frac_long'][t]:>8.2f}")
    print("  ※ lldiff>0 = emission이 Bull 선호 / raw_dur vs raw_plain 격차 = duration 고착분")

    # ── 표 B: arm별 smoothed P(bear) + exposure ──
    print(f"\n{'='*88}\n[B] arm별 smoothed P(bear) / exposure  ({W_LO}~{W_HI})\n{'='*88}")
    hdr = "".join(f"{a[:8]:>10}{'exp':>6}" for a in anames)
    print(f"  {'월':8}" + hdr)
    for t in wl:
        cells = "".join(f"{A[a]['pb'][t]:>10.2f}{A[a]['exp'][t]:>6.2f}" for a in anames)
        print(f"  {yms[t]:8}{cells}")

    # ── 표 C: 피처 + 수익 ──
    print(f"\n{'='*88}\n[C] 피처(level+변화율) + 수익  ({W_LO}~{W_HI})\n{'='*88}")
    print(f"  {'월':8}{'breadth':>8}{'newlow':>8}{'trend':>8}{'d_brea':>8}{'d_newl':>8}{'diverg':>8}{'KOSPI':>7}{'FCF+1':>7}")
    for t in wl:
        rk = f"{ret[t]*100:>6.1f}" if not np.isnan(ret[t]) else "   n/a"
        rf = f"{fcf_next[t]*100:>6.1f}" if not np.isnan(fcf_next[t]) else "   n/a"
        print(f"  {yms[t]:8}{df['breadth'].iloc[t]:>8.2f}{df['newlow'].iloc[t]:>8.2f}{df['trend'].iloc[t]:>8.3f}"
              f"{df['d_breadth'].iloc[t]:>8.3f}{df['d_newlow'].iloc[t]:>8.3f}{df['diverg'].iloc[t]:>8.2f}{rk}{rf}")

    # ── 7개 질문 자동 판정 ──
    tr = yms.index("2020-03")
    print(f"\n{'='*88}\n[D] 7개 질문 판정\n{'='*88}")
    q1 = next((yms[t] for t in range(tr, n) if lldiff[t] > 0), "없음")
    print(f"  Q1 Bear emission이 Bull보다 낮아지는 첫 달(lldiff>0): {q1}")
    q2d = next((yms[t] for t in range(tr + 1, n) if S['raw_dur'][t] < S['raw_dur'][tr] * 0.5), "?")
    q2p = next((yms[t] for t in range(tr + 1, n) if S['raw_plain'][t] < S['raw_plain'][tr] * 0.5), "?")
    print(f"  Q2 raw P(bear) 반감 시점:  duration={q2d}  plain={q2p}")
    def first_ge(exp, thr):
        t = next((t for t in range(tr, n) if exp[t] >= thr), None)
        return (t - tr) if t is not None else None
    print(f"  Q3 EMA 지연: FULL vs NO_EMA exp>=0.5 도달 = "
          f"{first_ge(A['FULL']['exp'],0.5)}m vs {first_ge(A['NO_EMA']['exp'],0.5)}m "
          f"(duration off: NO_DURATION {first_ge(A['NO_DURATION']['exp'],0.5)}m vs NO_BOTH {first_ge(A['NO_BOTH']['exp'],0.5)}m)")
    print(f"  Q4 duration off시 Bear질량 소멸(raw_plain<0.3) 첫 달: "
          f"{next((yms[t] for t in range(tr, n) if S['raw_plain'][t] < 0.3), '?')}")
    print(f"  Q5/Q6: 표A에서 lldiff 부호전환({q1}) 시점에 breadth/trend/raw 값 대조 (아래 요약)")
    # level 피처가 붙잡는지: lldiff>0 된 뒤에도 trend<0(200일선 아래)이면 level이 잡음
    if q1 != "없음":
        qi = yms.index(q1)
        print(f"     {q1} 시점 trend={df['trend'].iloc[qi]:.3f} (음수=지수 200일선 아래=level이 Bear쪽) "
              f"breadth={df['breadth'].iloc[qi]:.2f}")
    # Q7 상호작용: 단독효과 합 vs 결합효과
    b_full = first_ge(A['FULL']['exp'], 0.8); b_noema = first_ge(A['NO_EMA']['exp'], 0.8)
    b_nodur = first_ge(A['NO_DURATION']['exp'], 0.8); b_noboth = first_ge(A['NO_BOTH']['exp'], 0.8)
    print(f"  Q7 exp>=0.8 회복(개월): FULL={b_full} NO_EMA={b_noema} NO_DURATION={b_nodur} NO_BOTH={b_noboth}")
    print(f"     EMA단독효과={None if None in (b_full,b_noema) else b_full-b_noema}  "
          f"DUR단독효과={None if None in (b_full,b_nodur) else b_full-b_nodur}  "
          f"(둘 합 ≈ FULL-NO_BOTH={None if None in (b_full,b_noboth) else b_full-b_noboth} 이면 독립, 크게 벗어나면 상호작용)")

    # ── 성과 + trade-off ──
    m24 = win("2020-04", "2020-12"); m21 = win("2021-06", "2022-09")
    print(f"\n{'='*88}\n[E] 성과 & trade-off\n{'='*88}")
    print(f"  {'arm':13}{'CAGR':>8}{'MDD':>9}{'Sharpe':>8}{'Bear월':>7}{'평균exp':>8}"
          f"{'→0.5':>6}{'→0.8':>6}{'2020놓침':>9}{'2021방어':>9}")
    for a in anames:
        exp = A[a]['exp']
        miss = sum((1 - exp[t]) * (fcf_next[t] if not np.isnan(fcf_next[t]) else 0) for t in m24)
        defn = sum((1 - exp[t]) * (-(fcf_next[t] if not np.isnan(fcf_next[t]) else 0)) for t in m21)
        bm = sum(1 for t in range(start, n) if A[a]['reg'][t] == "Bear")
        r05, r08 = first_ge(exp, 0.5), first_ge(exp, 0.8)
        print(f"  {a:13}{A[a]['cagr']*100:>7.1f}%{A[a]['mdd']*100:>8.1f}%{A[a]['sharpe']:>8.2f}{bm:>7}"
              f"{np.mean(exp[start:]):>8.2f}{str(r05)+'m':>6}{str(r08)+'m':>6}{miss*100:>8.1f}%{defn*100:>8.1f}%")
    print("  ※ 2020놓침=(1-exp)×FCF, 2020-04~12 상승 미참여 / 2021방어=(1-exp)×(-FCF), 2021-06~2022-09 grind 방어이익")

    # ── 판정 가이드 ──
    print(f"\n{'='*88}\n[F] 판정 가이드\n{'='*88}")
    print("  · NO_DURATION/NO_BOTH에서 over-stay 사라지고 [E]2021방어 유지 → 2-state 내부수정 우선")
    print("  · NO_BOTH에서도 over-stay 남음(exp 오래 0.2) → emission/level피처 원인 (표A lldiff·trend 확인)")
    print("  · persistence 제거가 [E]2021방어/MDD를 크게 반납 → 단순제거 X, 조건부 recovery overlay 검토")

    out = pd.DataFrame({"ym": yms, "logB_bull": S["llb"], "logB_bear": S["llr"], "lldiff": lldiff,
                        "raw_dur": S["raw_dur"], "raw_plain": S["raw_plain"], "E_d_bear": S["ed_bear"],
                        "frac_long": S["frac_long"], "breadth": df["breadth"].values, "newlow": df["newlow"].values,
                        "trend": df["trend"].values, "d_breadth": df["d_breadth"].values,
                        "d_newlow": df["d_newlow"].values, "diverg": df["diverg"].values,
                        "kospi_ret": ret, "fcf_next": fcf_next})
    for a in anames:
        out[f"pb_{a}"] = A[a]["pb"]; out[f"exp_{a}"] = A[a]["exp"]
    out.to_csv(REPO / "analysis" / "overstay_mechanism_path.csv", index=False, encoding="utf-8-sig")
    print(f"\n  전체 경로 저장 → analysis/overstay_mechanism_path.csv")


if __name__ == "__main__":
    main()
