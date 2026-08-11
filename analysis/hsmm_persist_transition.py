"""
analysis/hsmm_persist_transition.py

지속성 신호를 **emission이 아니라 transition(duration hazard) 변조**에 넣는 실험.
2-state 유지 · newlow 원본 유지 · production 파일 미수정.

■ 왜 이 자리인가 (앞선 진단의 귀결)
  2022 slow bear에서 breadth·trend는 crash 상태와 사실상 동일(|z| 0.07 / 0.01)했는데
  newlow가 5.59σ로 단독 veto를 행사해 진입이 막혔다.
  emission을 손대는 처방은 전부 실패했다:
    · ever_newlow 교체 → veto 제거되며 신호까지 소실(항상 Bear)
    · newlow 압축(zclip2/pct60) → veto는 85%→34%로 완화됐으나 crash 상태 자체가 소멸
    · ma200_slope emission 추가 → veto 그대로라 무효
    · 3-state → 세 번째 상태가 slow bear가 아니라 짧은 crash로 형성
  → emission은 '값의 거리'를 다루는 층이라 newlow와 z² 경쟁을 피할 수 없다.
    slow bear의 본질은 '값의 극단'이 아니라 **낮은 상태의 지속**이므로
    duration hazard 층이 표현해야 할 성질이다. 이 층은 우도와 경쟁하지 않아 veto를 우회한다.

■ 구현
  hsmm_filter의 exit hazard 변조에 지속성 항 p를 하나 더한다(원본 stress 항은 그대로).

      he[bull] = sigmoid( logit(h_bull) + KAPPA*z_stress + LAMBDA*p )   지속성↑ → Bull 이탈↑
      he[bear] = sigmoid( logit(h_bear) - KAPPA*z_stress - LAMBDA*p )   지속성↑ → Bear 유지↑

  LAMBDA=0이면 production과 **완전히 동일**하다(회귀 검증에 사용).

■ 지속성 후보 (모두 causal — trailing만 사용, 학습창 안에서 표준화)
  slope     : -ma200_slope_60           (200일선이 내려가는 중 = 약세 지속)
  bread6    : -(breadth 6개월 이동평균)   (저breadth가 이어지는 정도)
  negrun    : trend<0 연속 개월수         (추세 음수 지속)
  ddpeak    : KOSPI 12개월 고점 대비 낙폭  (고점에서 얼마나 눌려 있나)

■ 평가
  production과 같은 이산 레짐(0.6/0.4)·EMA. slow bear(2021-11~2022-12) P_bear 경로와
  lift/분별력/FP/Bear비율/crash 유지/lead를 함께 본다.
  LAMBDA 민감도(0.4/0.8/1.2)를 같이 찍어 **한 점에서만 좋아지는 과적합**을 걸러낸다.

■ 사용 / 산출
  .venv/bin/python analysis/hsmm_persist_transition.py            (production 패널)
  .venv/bin/python analysis/hsmm_persist_transition.py --long     (장기 패널 robustness)
  analysis/results/hsmm_persist_compare.csv / _path.csv
"""
import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent.parent
A_DIR = Path(__file__).parent
OUT = A_DIR / "results"; OUT.mkdir(exist_ok=True)
SHORT_PANEL = A_DIR / ".cache" / "hsmm_features.pkl"
LONG_PANEL = A_DIR / ".cache" / "hsmm_longrun_features.pkl"
SLOPE_CACHE = A_DIR / ".cache" / "ma200_slope_persist.pkl"
sys.path.insert(0, str(BASE))
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ModuleNotFoundError:
    pass

import importlib.util
_sp = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_sp); sys.modules["hsmm_final"] = HF; _sp.loader.exec_module(HF)

EMIS, TRAN = HF.EMIS_COLS, HF.TRAN_COLS
WINDOW_M, REFIT_EVERY, HL, EPS, KAPPA = HF.WINDOW_M, HF.REFIT_EVERY, HF.HL, HF.EPS, HF.KAPPA
PBEAR_EMA, T_IN, T_OUT, WIN, DMAX = HF.PBEAR_EMA, HF.T_IN, HF.T_OUT, HF.WIN, HF.DMAX
SLOW_A, SLOW_B = "2021-11", "2022-12"


# ─────────────────── 지속성 신호 (causal) ───────────────────
def load_slope(yms):
    if SLOPE_CACHE.exists():
        s = pd.read_pickle(SLOPE_CACHE)
        if set(yms) <= set(s.index):
            return s.reindex(yms)
    conn = HF._connect()
    k = pd.read_sql("SELECT period p, value::float v FROM alpha_lab.macro_indicators "
                    "WHERE indicator='kospi' AND freq='D'", conn)
    conn.close()
    k["p"] = pd.to_datetime(k.p.str.slice(0, 10))
    ks = k.set_index("p")["v"].sort_index()
    ma = ks.rolling(200, min_periods=100).mean()
    sl = ma / ma.shift(60) - 1
    mend = {}
    for d, pp in zip(ks.index, pd.PeriodIndex(ks.index, freq="M")):
        mend[pp.strftime("%Y-%m")] = d
    out = pd.Series({y: (sl[sl.index <= mend[y]].iloc[-1] if y in mend else np.nan) for y in yms}).fillna(0.0)
    old = pd.read_pickle(SLOPE_CACHE) if SLOPE_CACHE.exists() else pd.Series(dtype=float)
    pd.to_pickle(pd.concat([old, out[~out.index.isin(old.index)]]), SLOPE_CACHE)
    return out


def build_persist(df, yms, ret):
    """지속성 후보들. 모두 trailing 정보만 사용한다."""
    P = {}
    P["slope"] = -load_slope(yms).values                       # 내려갈수록 큰 값
    P["bread6"] = -df["breadth"].rolling(6, min_periods=2).mean().fillna(method="bfill").values
    tr = df["trend"].values
    run = np.zeros(len(tr))
    for i in range(len(tr)):
        run[i] = (run[i - 1] + 1) if (i > 0 and tr[i] < 0) else (1.0 if tr[i] < 0 else 0.0)
    P["negrun"] = run
    px = np.ones(len(ret) + 1)                                 # ret은 forward → 가격지수 복원
    for i in range(len(ret)):
        px[i + 1] = px[i] * (1 + (ret[i] if not np.isnan(ret[i]) else 0.0))
    px = px[:len(ret)]
    s = pd.Series(px)
    P["ddpeak"] = -(s / s.rolling(12, min_periods=3).max() - 1).fillna(0.0).values   # 낙폭 클수록 큰 값
    return P


# ─────────────────── 2-state HSMM 필터 (지속성 항 추가) ───────────────────
def hsmm_filter_p(logB, stress, persist, haz, bear, pi, lam):
    """원본 HF.hsmm_filter와 동일하되 exit hazard에 LAMBDA*persist 항을 더한다.
       lam=0이면 원본과 완전히 같다."""
    T = logB.shape[0]; D = haz.shape[1]; bull = 1 - bear
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    haz_l = np.log(haz / (1 - haz))
    filt = np.zeros((T, 2))
    a = np.zeros((2, D)); a[:, 0] = pi * B[0]; a /= a.sum() + EPS; filt[0] = a.sum(1)
    for t in range(1, T):
        z, p = stress[t], persist[t]
        he = np.empty_like(haz)
        he[bull] = 1 / (1 + np.exp(-(haz_l[bull] + KAPPA * z + lam * p)))
        he[bear] = 1 / (1 + np.exp(-(haz_l[bear] - KAPPA * z - lam * p)))
        he = np.clip(he, 1e-6, 1 - 1e-6); he[:, -1] = 1.0
        cont = a * (1 - he); endm = (a * he).sum(1)
        nxt = np.zeros((2, D))
        nxt[:, 1:] = cont[:, :-1]
        nxt[bull, 0] = endm[bear]; nxt[bear, 0] = endm[bull]
        nxt = nxt * B[t][:, None]; nxt /= nxt.sum() + EPS
        a = nxt; filt[t] = a.sum(1)
    return filt


def walk_forward(df, yms, n, persist_raw, lam, decide_start):
    """production walk_forward와 동일 골격. 지속성만 추가로 변조."""
    start = yms.index(decide_start) if decide_start in yms else 12
    TRz = HF.roll_z(df, TRAN).values
    EM = df[EMIS].values
    stress_raw = TRz[:, 0] - TRz[:, 1]
    pbear = np.full(n, np.nan)
    params, sc, last_refit = None, None, -10 ** 9
    for t in range(start, n):
        lo = max(0, t + 1 - WINDOW_M); Xr = EM[lo:t + 1]; nw = len(Xr)
        w = 0.5 ** ((nw - 1 - np.arange(nw)) / HL)
        if params is None or (t - last_refit) >= REFIT_EVERY:
            sc = StandardScaler().fit(Xr)
            params = HF.fit_emission(sc.transform(Xr), w,
                                     HF.cold_emission(sc.transform(Xr)) if params is None else params,
                                     40 if params is None else 10)
            last_refit = t
        Xz = sc.transform(Xr)
        logB = HF.emis_logB(Xz, params["means"], params["covs"])
        bear = int(np.argmax(HF.bear_score(params["means"])))
        gamma, _ = HF.forward_backward(logB, params["Amat"], params["pi"], w)
        durs = HF.state_durations(gamma.argmax(1))
        haz = np.vstack([HF.to_hazard(HF.dur_pmf(durs[0])), HF.to_hazard(HF.dur_pmf(durs[1]))])
        sw = stress_raw[lo:t + 1]; sw = (sw - sw.mean()) / (sw.std() + EPS)
        pw = persist_raw[lo:t + 1]; pw = (pw - pw.mean()) / (pw.std() + EPS)   # 학습창 내 표준화
        filt = hsmm_filter_p(logB, sw, pw, haz, bear, params["pi"], lam)
        pbear[t] = filt[-1, bear]
    return pbear, start


def regimes(pb, n, start):
    reg = ["Bull"] * n; p = "Bull"
    for t in range(start, n):
        p = ("Bear" if pb[t] >= T_OUT else "Bull") if p == "Bear" else ("Bear" if pb[t] >= T_IN else "Bull")
        reg[t] = p
    return reg


def evaluate(pb, reg, yms, n, start, dd6, thr):
    idx = list(range(start, n))
    evs = [i for i in idx if dd6[i] <= thr and (i == start or not (dd6[i - 1] <= thr))]
    onset = [t for t in idx if reg[t] == "Bear" and reg[t - 1] != "Bear"]
    leads, nhit = [], 0
    for ev in evs:
        cand = [s for s in onset if abs(s - ev) <= WIN]
        if cand:
            nhit += 1; leads.append(ev - min(cand, key=lambda s: abs(s - ev)))
    ok = [t for t in idx if not np.isnan(dd6[t])]
    tp = sum(1 for t in ok if dd6[t] <= thr and reg[t] == "Bear")
    fn = sum(1 for t in ok if dd6[t] <= thr and reg[t] != "Bear")
    fp = sum(1 for t in ok if dd6[t] > thr and reg[t] == "Bear")
    tn = sum(1 for t in ok if dd6[t] > thr and reg[t] != "Bear")
    prec = tp / (tp + fp) if tp + fp else np.nan
    base = (tp + fn) / len(ok) if ok else np.nan
    lift = prec / base if (base and not np.isnan(prec)) else np.nan
    bdd = np.mean([dd6[t] for t in ok if reg[t] == "Bear"]) if (tp + fp) else np.nan
    udd = np.mean([dd6[t] for t in ok if reg[t] != "Bear"]) if (tn + fn) else np.nan
    disc = udd - bdd if not (np.isnan(bdd) or np.isnan(udd)) else np.nan
    sb = [t for t in idx if SLOW_A <= yms[t] <= SLOW_B]
    v = pb[sb]
    return dict(rec=nhit / len(evs) if evs else np.nan, lead=np.mean(leads) if leads else np.nan,
                lift=lift, disc=disc, fp=fp, bear_ratio=(tp + fp) / len(ok),
                sb_mean=float(np.nanmean(v)), sb_max=float(np.nanmax(v)),
                sb_gt05=int((v > 0.5).sum()), sb_bear=sum(1 for t in sb if reg[t] == "Bear"),
                evs=evs, onset=onset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--thr", type=float, default=-15.0)
    ap.add_argument("--decide-start", default=None)
    args = ap.parse_args()
    thr = -abs(args.thr)

    if args.long:
        if not LONG_PANEL.exists():
            print("!! 장기 패널 없음 — analysis/hsmm_longrun.py 먼저 실행"); return
        df, yms, n, ret, _rv, _dv, dd6, _raw = pd.read_pickle(LONG_PANEL)
        ds = args.decide_start or "2009-01"
        tag = "장기(2004~)"
    else:
        if not SHORT_PANEL.exists():
            print("!! production 패널 없음 — analysis/hsmm_final.py 먼저 실행"); return
        df, yms, n, ret, _rv, _dv, dd6 = pd.read_pickle(SHORT_PANEL)
        ds = args.decide_start or HF.DECIDE_START
        tag = "production(2017~)"

    print(f"패널 {tag}  {yms[0]}~{yms[-1]} ({n}개월)  판정 {ds}  2-state · newlow 원본")
    P = build_persist(df, yms, ret)
    S = df.loc[ds:]
    print("\n지속성 후보 (학습창 내 표준화 전 원값):")
    for k, v in P.items():
        s = pd.Series(v, index=yms).loc[ds:]
        sb = pd.Series(v, index=yms).loc[SLOW_A:SLOW_B]
        print(f"  {k:8} 평균 {s.mean():+8.3f}  σ {s.std():7.3f}   slow bear 평균 {sb.mean():+8.3f}"
              f"  ({(sb.mean()-s.mean())/ (s.std()+1e-9):+.2f}σ)")

    LAMS = [0.0, 0.4, 0.8, 1.2, 2.0, 4.0, 8.0, 16.0]
    rows, paths = [], []
    base = None
    print(f"\n{'='*96}")
    print(f"  {'신호':9}{'λ':>5}{'slowbear평균':>13}{'최대':>7}{'>0.5':>6}{'Bear월':>8}"
          f"{'리프트':>8}{'분별력':>9}{'FP':>5}{'Bear비율':>9}{'선행':>7}{'Recall':>8}")
    print("=" * 96)
    for sig in ["(none)"] + list(P):
        for lam in LAMS:
            if sig == "(none)" and lam != 0.0:
                continue
            pr = np.zeros(n) if sig == "(none)" else P[sig]
            pb_raw, start = walk_forward(df, yms, n, pr, lam, ds)
            pb = pb_raw.copy()
            for t in range(start + 1, n):
                pb[t] = PBEAR_EMA * pb_raw[t] + (1 - PBEAR_EMA) * pb[t - 1]
            reg = regimes(pb, n, start)
            m = evaluate(pb, reg, yms, n, start, dd6, thr)
            if base is None:
                base = m
            name = "production" if sig == "(none)" else sig
            print(f"  {name:9}{lam:5.1f}{m['sb_mean']:13.3f}{m['sb_max']:7.2f}{m['sb_gt05']:6d}"
                  f"{m['sb_bear']:8d}{m['lift']:8.2f}{m['disc']:+8.1f}%{m['fp']:5d}"
                  f"{m['bear_ratio']:9.0%}{m['lead']:+7.1f}{m['rec']:8.0%}")
            rows.append(dict(signal=name, lam=lam, **{k: v for k, v in m.items()
                                                      if k not in ("evs", "onset")}))
            for t in range(n):
                paths.append(dict(signal=name, lam=lam, ym=yms[t], pbear=pb[t],
                                  regime=reg[t], dd6=dd6[t]))
        if sig != "(none)":
            print("-" * 96)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "hsmm_persist_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paths).to_csv(OUT / "hsmm_persist_path.csv", index=False, encoding="utf-8-sig")

    print(f"\n  기준(production, λ=0): slow bear 평균 {base['sb_mean']:.3f} / Bear월 {base['sb_bear']}"
          f" / 리프트 {base['lift']:.2f} / 분별력 {base['disc']:+.1f}%p / FP {base['fp']}")
    cand = R[(R.lam > 0) & (R.sb_bear > base["sb_bear"]) & (R.lift >= base["lift"] * 0.95)
             & (R.bear_ratio <= base["bear_ratio"] + 0.15) & (R.lead >= 0)]
    print("\n  ★ 1차 통과(slow bear 개선 + 리프트 비악화 + Bear비율 과증가 없음 + 양의 선행):")
    if len(cand):
        print(cand[["signal", "lam", "sb_bear", "sb_mean", "lift", "disc", "fp", "bear_ratio", "lead"]]
              .round(3).to_string(index=False))
        print("\n  → λ 민감도를 확인할 것. 한 λ에서만 좋으면 과적합 의심.")
    else:
        print("    없음 — 지속성 채널로도 개선 없음")
    print(f"\n  → {OUT/'hsmm_persist_compare.csv'}\n  → {OUT/'hsmm_persist_path.csv'}")


if __name__ == "__main__":
    main()
