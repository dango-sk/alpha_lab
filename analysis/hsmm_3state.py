"""
analysis/hsmm_3state.py

장기 패널(2004~)에서 K-state HSMM 비교 — 2-state vs 3-state.

■ 배경
  production(hsmm_final.py, 2-state, 짧은 패널)은 2021~22 slow bear 14개월 중 Bear 판정 0개월.
  피처 교체(ever_newlow_20 원시/Z36)·피처 추가(ma200_slope_60) 모두 실패했고,
  장기 패널에서 정상 covariance로 추정해도 slow bear 0개월이었다.
  → 피처가 아니라 **2-state 구조의 한계** 가설. 상태가 2개면 crash와 slow bear 중
    하나만 표현할 수 있고, 현재는 crash(newlow 스파이크)를 잡고 있다.

■ 이 스크립트가 하는 것
  - 장기 패널(analysis/.cache/hsmm_longrun_features.pkl, 2004-01~)만 사용. production 미수정.
  - covariance는 **정상 초기화**(모든 피처에 적합 분산). 장기 패널은 창 60개월이 차 있어 가능하다.
    ※ production의 앵커-바닥 초기화는 13개월 창 전용 정규화다(hsmm_final.cold_emission 주석 참조).
  - 엔진을 K-state로 일반화(원본은 K=2 하드코딩). K=2 결과가 기존과 같은 성질을 갖는지 함께 확인.
  - **상태 이름을 사전에 박지 않는다.** 학습 후 emission 평균·지속기간·조건부 미래낙폭을
    출력하고, 거기서 위험도 순서를 데이터로 매긴다.

■ 2-state 로직의 K-state 일반화 (원본과 K=2에서 동치)
  원본 hsmm_filter는 세그먼트 종료 시 '상대 상태로 이동'(2개뿐이라 가능)하고,
  스트레스로 exit hazard를 변조한다(Bull 이탈↑, Bear 유지↑).
    he[bull] = sigmoid(logit(h) + KAPPA*z)
    he[bear] = sigmoid(logit(h) - KAPPA*z)
  K개로 늘리려면 두 가지가 필요하다.
    (1) 종료 시 이동할 곳: 대각 0으로 만든 전이행렬 Aexit (xi에서 추정, 행 정규화)
    (2) 스트레스 부호: 상태별 위험도 r_k를 [-1,+1]로 정규화해
        he_k = sigmoid(logit(h_k) - KAPPA*z*r_k)
        K=2에서 r=[-1,+1]이 되어 원본과 정확히 같아진다.
  r_k는 emission 평균에서 계산하므로 하드코딩이 아니다.

■ 평가 (2022를 잡았다는 이유만으로 채택하지 않는다)
  - slow bear(2021-11~2022-12): 판정 개월수 + **선행 시점**
  - crash 유지: 2008/2011/2020 등 기존 탐지가 깨지지 않는지
  - 리프트 / 분별력 / false positive / Bear비율
  - 상태 duration 분포
  - **경제적 반복성**: 각 상태가 전 기간에 걸쳐 몇 개 에피소드로, 어떤 시기에 나타나는지
  - BIC로 K 선택이 지지되는지

■ 사용
  .venv/bin/python analysis/hsmm_3state.py
  옵션: --decide-start 2009-01   --thr -15   --seeds 5
■ 산출
  analysis/results/hsmm_3state_path.csv
  analysis/results/hsmm_3state_states.csv
"""
import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent.parent
A_DIR = Path(__file__).parent
OUT = A_DIR / "results"; OUT.mkdir(exist_ok=True)
LONG_PANEL = A_DIR / ".cache" / "hsmm_longrun_features.pkl"

sys.path.insert(0, str(BASE))
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ModuleNotFoundError:
    pass

import importlib.util
_spec = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_spec)
sys.modules["hsmm_final"] = HF
_spec.loader.exec_module(HF)

EMIS = HF.EMIS_COLS                 # ['breadth', 'newlow', 'trend']
TRAN = HF.TRAN_COLS                 # ['fx3m', 'fflow']
# emission 부호: 값이 클수록 약세이면 +1, 강세이면 -1  (bear_score와 동일 의미)
EMIS_SIGN = np.array([-1.0, +1.0, -1.0])
WINDOW_M, REFIT_EVERY, DMAX = HF.WINDOW_M, HF.REFIT_EVERY, HF.DMAX
HL, KAPPA, EPS = HF.HL, HF.KAPPA, HF.EPS
PBEAR_EMA, T_IN, T_OUT, WIN = HF.PBEAR_EMA, HF.T_IN, HF.T_OUT, HF.WIN
SLOW_A, SLOW_B = "2021-11", "2022-12"


# ─────────────────────────── K-state 엔진 ───────────────────────────
def emis_logB(X, means, covs):
    n, d = X.shape; K = len(means)
    logB = np.zeros((n, K))
    for k in range(K):
        L = np.linalg.cholesky(covs[k])
        sol = np.linalg.solve(L, (X - means[k]).T)
        logB[:, k] = -0.5 * ((sol ** 2).sum(0) + 2 * np.log(np.diag(L)).sum() + d * np.log(2 * np.pi))
    return logB


def m_step_emis(X, gamma, w):
    n, d = X.shape; K = gamma.shape[1]
    means = np.zeros((K, d)); covs = np.zeros((K, d, d))
    for k in range(K):
        r = gamma[:, k] * w; R = r.sum() + EPS
        mu = (r[:, None] * X).sum(0) / R
        Xc = X - mu; C = (r[:, None] * Xc).T @ Xc / R
        hard = gamma[:, k] > 0.5
        delta = float(LedoitWolf().fit(X[hard]).shrinkage_) if hard.sum() >= d + 2 else 0.5
        mt = np.trace(C) / d
        covs[k] = (1 - delta) * C + delta * mt * np.eye(d) + 1e-6 * np.eye(d)
        means[k] = mu
    return means, covs


def cold_emission(X, K, seed):
    """장기 패널용 **정상** 초기화 — 모든 피처에 적합 분산을 준다.
       (production의 앵커-바닥 초기화는 13개월 창 전용 정규화이므로 여기선 쓰지 않는다.)"""
    d = X.shape[1]
    hm = GaussianHMM(K, "diag", n_iter=50, random_state=seed)
    hm.fit(X)
    covs = np.array([np.diag(np.diag(np.asarray(hm.covars_[k]))) + 1e-6 * np.eye(d) for k in range(K)])
    A = np.full((K, K), 0.1 / max(K - 1, 1)); np.fill_diagonal(A, 0.9)
    return dict(means=hm.means_.copy(), covs=covs, Amat=A, pi=np.full(K, 1.0 / K))


def fit_emission(X, w, init, n_iter):
    means, covs, Amat, pi = init["means"], init["covs"], init["Amat"], init["pi"]
    gamma = None
    for _ in range(n_iter):
        logB = emis_logB(X, means, covs)
        gamma, xi = HF.forward_backward(logB, Amat, pi, w)      # 원본은 이미 K-general
        pi = gamma[0] / (gamma[0].sum() + EPS)
        Amat = xi / (xi.sum(axis=1, keepdims=True) + EPS)
        means, covs = m_step_emis(X, gamma, w)
        # 수치 가드: 상태 붕괴 시 NaN/비양정 방지 (장기 패널에선 거의 발동하지 않는다)
        means = np.nan_to_num(means)
        for k in range(len(covs)):
            C = np.nan_to_num(covs[k]); C = (C + C.T) / 2
            ev = np.linalg.eigvalsh(C)
            if not np.all(np.isfinite(ev)) or ev.min() < 1e-6:
                C = C + max(1e-6 - min(ev.min(), 0.0), 1e-6) * np.eye(C.shape[0])
            covs[k] = C
    return dict(means=means, covs=covs, Amat=Amat, pi=pi, gamma=gamma)


def state_durations(path, K):
    durs = {k: [] for k in range(K)}
    s, L = int(path[0]), 1
    for i in range(1, len(path)):
        if int(path[i]) == s:
            L += 1
        else:
            durs[s].append(L); s, L = int(path[i]), 1
    durs[s].append(L)
    return durs


def risk_scores(means):
    """emission 평균 → 상태별 위험도. 하드코딩된 이름이 아니라 피처 부호로 계산."""
    return means @ EMIS_SIGN


def hsmm_filter(logB, stress, haz, pi, Aexit, r_norm):
    """K-state EDHMM forward 필터. 확장상태(state × 경과기간).

    exit hazard를 스트레스로 변조:  he_k = sigmoid(logit(h_k) - KAPPA*z*r_k)
      r_k는 [-1,+1] 정규화 위험도. 저위험 상태(r<0)는 스트레스↑ → 이탈↑,
      고위험 상태(r>0)는 스트레스↑ → 유지↑. K=2에서 r=[-1,+1]이면 원본과 동치.
    종료 시에는 Aexit(대각 0, 행 정규화)에 따라 다른 상태의 d=1로 이동한다.
    """
    T, K = logB.shape; D = haz.shape[1]
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    haz_l = np.log(haz / (1 - haz))
    filt = np.zeros((T, K))
    a = np.zeros((K, D)); a[:, 0] = pi * B[0]; a /= a.sum() + EPS; filt[0] = a.sum(1)
    for t in range(1, T):
        z = stress[t]
        he = 1.0 / (1.0 + np.exp(-(haz_l - KAPPA * z * r_norm[:, None])))
        he = np.clip(he, 1e-6, 1 - 1e-6); he[:, -1] = 1.0
        cont = a * (1 - he)                       # 계속 진행 (경과기간 +1)
        endm = (a * he).sum(1)                    # 각 상태에서 종료된 질량
        nxt = np.zeros((K, D))
        nxt[:, 1:] = cont[:, :-1]
        nxt[:, 0] = endm @ Aexit                  # 종료 → 다른 상태의 d=1
        nxt = nxt * B[t][:, None]
        nxt /= nxt.sum() + EPS
        a = nxt; filt[t] = a.sum(1)
    return filt


def walk_forward(df, yms, n, K, decide_start, seed):
    """production walk_forward와 같은 골격(60월 창·연1회 재적합·웜스타트·시간감쇠)."""
    start = yms.index(decide_start) if decide_start in yms else 12
    TRz = HF.roll_z(df, TRAN).values
    EM = df[EMIS].values
    stress_raw = TRz[:, 0] - TRz[:, 1]
    P = np.full((n, K), np.nan)
    params, sc, last_refit = None, None, -10 ** 9
    means_log = {}
    for t in range(start, n):
        lo = max(0, t + 1 - WINDOW_M); Xr = EM[lo:t + 1]; nw = len(Xr)
        w = 0.5 ** ((nw - 1 - np.arange(nw)) / HL)
        if params is None or (t - last_refit) >= REFIT_EVERY:
            sc = StandardScaler().fit(Xr)
            params = fit_emission(sc.transform(Xr), w,
                                  cold_emission(sc.transform(Xr), K, seed) if params is None else params,
                                  40 if params is None else 10)
            last_refit = t
            means_log[yms[t]] = sc.inverse_transform(params["means"])   # 원 단위 emission 평균
        Xz = sc.transform(Xr)
        logB = emis_logB(Xz, params["means"], params["covs"])
        gamma, _ = HF.forward_backward(logB, params["Amat"], params["pi"], w)
        durs = state_durations(gamma.argmax(1), K)
        haz = np.vstack([HF.to_hazard(HF.dur_pmf(durs[k])) for k in range(K)])
        r = risk_scores(params["means"])
        rng = r.max() - r.min()
        r_norm = (2 * (r - r.min()) / rng - 1) if rng > EPS else np.zeros(K)   # [-1,+1]
        Aex = params["Amat"].copy(); np.fill_diagonal(Aex, 0.0)
        Aex = Aex / (Aex.sum(axis=1, keepdims=True) + EPS)
        sw = stress_raw[lo:t + 1]; sw = (sw - sw.mean()) / (sw.std() + EPS)
        filt = hsmm_filter(logB, sw, haz, params["pi"], Aex, r_norm)
        # 상태 순서는 학습마다 뒤바뀔 수 있으므로 위험도 순위로 재정렬해 기록
        order = np.argsort(r)                      # 낮은 위험 → 높은 위험
        P[t] = filt[-1][order]
        if t % 24 == 0 or t == n - 1:
            print(f"  {yms[t]}  P={np.round(P[t], 2)}", flush=True)
    return P, start, means_log


def bic_for_K(EM, K, seed):
    """전 구간 1회 적합 기준 BIC(작을수록 좋음). K 선택의 참고치."""
    X = StandardScaler().fit_transform(EM)
    w = np.ones(len(X))
    p = fit_emission(X, w, cold_emission(X, K, seed), 60)
    logB = emis_logB(X, p["means"], p["covs"])
    gamma, _ = HF.forward_backward(logB, p["Amat"], p["pi"], w)
    ll = float((gamma * logB).sum())
    d = X.shape[1]
    n_par = K * d + K * d + K * (K - 1) + (K - 1)      # 평균 + 대각공분산 + 전이 + 초기
    return -2 * ll + n_par * np.log(len(X)), ll


# ─────────────────────────── 평가 ───────────────────────────
def to_regime(prisk, n, start):
    reg = ["Benign"] * n; p = "Benign"
    for t in range(start, n):
        p = ("Risk" if prisk[t] >= T_OUT else "Benign") if p == "Risk" else ("Risk" if prisk[t] >= T_IN else "Benign")
        reg[t] = p
    return reg


def evaluate(name, prisk, reg, yms, n, start, dd6, ret, thr):
    idx = list(range(start, n))
    evs = [i for i in idx if dd6[i] <= thr and (i == start or not (dd6[i - 1] <= thr))]
    onset = [t for t in idx if reg[t] == "Risk" and reg[t - 1] != "Risk"]
    hits, leads, matched = [], [], set()
    for ev in evs:
        cand = [s for s in onset if abs(s - ev) <= WIN]
        if cand:
            b = min(cand, key=lambda s: abs(s - ev)); hits.append(ev); leads.append(ev - b); matched.add(b)
        else:
            hits.append(None)
    ok = [t for t in idx if not np.isnan(dd6[t])]
    tp = sum(1 for t in ok if dd6[t] <= thr and reg[t] == "Risk")
    fn = sum(1 for t in ok if dd6[t] <= thr and reg[t] != "Risk")
    fp = sum(1 for t in ok if dd6[t] > thr and reg[t] == "Risk")
    tn = sum(1 for t in ok if dd6[t] > thr and reg[t] != "Risk")
    prec = tp / (tp + fp) if tp + fp else np.nan
    base = (tp + fn) / len(ok) if ok else np.nan
    lift = prec / base if (base and not np.isnan(prec)) else np.nan
    bdd = np.mean([dd6[t] for t in ok if reg[t] == "Risk"]) if (tp + fp) else np.nan
    udd = np.mean([dd6[t] for t in ok if reg[t] != "Risk"]) if (tn + fn) else np.nan
    disc = udd - bdd if not (np.isnan(bdd) or np.isnan(udd)) else np.nan

    sb = [t for t in idx if SLOW_A <= yms[t] <= SLOW_B]
    sb_hit = [t for t in sb if reg[t] == "Risk"]
    sb_first = yms[sb_hit[0]] if sb_hit else None
    # slow bear 선행: 구간 시작(2021-11) 대비 최초 Risk 진입 시점
    sb_lead = (sb.index(sb_hit[0]) * -1) if sb_hit else None

    print(f"\n  [{name}]")
    print(f"    이벤트 {len(evs)}개 중 탐지 {sum(h is not None for h in hits)}개"
          f"   평균선행 {(f'{np.mean(leads):+.1f}개월' if leads else '-')}")
    print(f"    Risk비율 {(tp+fp)/len(ok):.0%} (기저 {base:.0%})   리프트 "
          f"{('nan' if np.isnan(lift) else f'{lift:.2f}배')}   분별력 "
          f"{('nan' if np.isnan(disc) else f'{disc:+.1f}%p')}   FP {fp}")
    print(f"    ★ slow bear({SLOW_A}~{SLOW_B}) {len(sb)}개월 중 Risk {len(sb_hit)}개월"
          f"   최초 진입 {sb_first or '-'}")
    return dict(lift=lift, disc=disc, fp=fp, sb=len(sb_hit), sb_first=sb_first,
                lead=np.mean(leads) if leads else np.nan, evs=evs, hits=hits, onset=onset,
                risk_ratio=(tp + fp) / len(ok), rec=sum(h is not None for h in hits) / len(evs) if evs else np.nan)


def describe_states(name, P, yms, n, start, dd6, ret, means_log, K):
    """상태 해석 — 이름을 박지 않고 emission 평균·지속·조건부 미래로 성격을 읽는다."""
    print(f"\n  ── {name} 상태 해석 (위험도 낮은 순 S0..S{K-1}) ──")
    hard = np.full(n, -1)
    for t in range(start, n):
        hard[t] = int(np.argmax(P[t]))
    last_means = means_log[sorted(means_log)[-1]]
    r = risk_scores((last_means - last_means.mean(0)) / (last_means.std(0) + EPS))
    order = np.argsort(r)
    M = last_means[order]
    print(f"    {'상태':5}{'개월':>6}{'비중':>7}" + "".join(f"{c:>10}" for c in EMIS)
          + f"{'평균지속':>9}{'조건부6M낙폭':>13}{'익월수익':>10}")
    rows = []
    for k in range(K):
        sel = [t for t in range(start, n) if hard[t] == k]
        if not sel:
            print(f"    S{k:<4}{0:>6}"); continue
        segs = []
        cur = 0
        for t in range(start, n):
            if hard[t] == k:
                cur += 1
            elif cur:
                segs.append(cur); cur = 0
        if cur:
            segs.append(cur)
        d6 = np.nanmean([dd6[t] for t in sel])
        rt = np.nanmean([ret[t] for t in sel]) * 100
        print(f"    S{k:<4}{len(sel):>6}{len(sel)/(n-start):>6.0%}"
              + "".join(f"{M[k][j]:>10.3f}" for j in range(len(EMIS)))
              + f"{np.mean(segs):>9.1f}{d6:>13.1f}%{rt:>9.2f}%")
        rows.append(dict(model=name, state=f"S{k}", months=len(sel), share=len(sel) / (n - start),
                         **{c: M[k][j] for j, c in enumerate(EMIS)},
                         mean_duration=np.mean(segs), n_episodes=len(segs),
                         cond_dd6=d6, next_ret=rt))
    # 경제적 반복성: 각 상태의 에피소드가 전 기간에 흩어져 있는가
    print(f"\n    경제적 반복성 (에피소드 시작 시점):")
    for k in range(K):
        eps_start = [yms[t] for t in range(start, n)
                     if hard[t] == k and (t == start or hard[t - 1] != k)]
        print(f"      S{k}  {len(eps_start):2d}회  {', '.join(eps_start[:10])}{' ...' if len(eps_start) > 10 else ''}")
    return hard, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide-start", default="2009-01")
    ap.add_argument("--thr", type=float, default=-15.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    thr = -abs(args.thr)

    if not LONG_PANEL.exists():
        print(f"!! 장기 패널이 없습니다: {LONG_PANEL}\n   먼저 analysis/hsmm_longrun.py 실행")
        return
    df, yms, n, ret, _rv, _dv, dd6, _raw = pd.read_pickle(LONG_PANEL)
    print(f"장기 패널 {yms[0]} ~ {yms[-1]} ({n}개월)   판정 시작 {args.decide_start}")
    print(f"emission {EMIS} (부호 {EMIS_SIGN.tolist()})   transition {TRAN}")
    print("covariance: 정상 초기화(모든 피처 적합 분산) — 장기 패널이라 가능")

    print(f"\n{'='*76}\n  BIC (전 구간 1회 적합, 작을수록 좋음)\n{'='*76}")
    for K in (2, 3, 4):
        try:
            b, ll = bic_for_K(df[EMIS].values, K, args.seed)
            print(f"  K={K}   BIC {b:10.1f}   loglik {ll:10.1f}")
        except Exception as e:
            print(f"  K={K}   실패: {type(e).__name__}: {str(e)[:50]}")

    res, staterows = {}, []
    for K in (2, 3):
        name = f"{K}-state"
        print(f"\n{'='*76}\n  {name}  walk-forward\n{'='*76}")
        P, start, mlog = walk_forward(df, yms, n, K, args.decide_start, args.seed)
        # 위험확률 = 최저위험 상태(S0)가 아닐 확률
        prisk_raw = 1.0 - P[:, 0]
        prisk = prisk_raw.copy()
        for t in range(start + 1, n):
            prisk[t] = PBEAR_EMA * prisk_raw[t] + (1 - PBEAR_EMA) * prisk[t - 1]
        reg = to_regime(prisk, n, start)
        res[name] = (P, prisk, reg, start, mlog)
        hard, rows = describe_states(name, P, yms, n, start, dd6, ret, mlog, K)
        staterows += rows

    print(f"\n{'='*76}\n  탐지기 평가 (이벤트 = 향후6M 낙폭 <= {thr:.0f}%)\n{'='*76}")
    summ = {}
    for name, (P, prisk, reg, start, _m) in res.items():
        summ[name] = evaluate(name, prisk, reg, yms, n, start, dd6, ret, thr)

    print(f"\n{'='*76}\n  2-state vs 3-state\n{'='*76}")
    names = list(res)
    print(f"  {'지표':22}" + "".join(f"{x:>14}" for x in names))
    for k, lab, fmt in [("sb", "★slow bear Risk월", "{:d}"), ("lead", "평균선행(월)", "{:+.1f}"),
                        ("lift", "★리프트", "{:.2f}배"), ("disc", "★분별력", "{:+.1f}%p"),
                        ("fp", "False Positive", "{:d}"), ("risk_ratio", "Risk비율", "{:.0%}"),
                        ("rec", "이벤트Recall", "{:.0%}")]:
        cells = ""
        for x in names:
            v = summ[x][k]
            cells += f"{'  -  ':>14}" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{fmt.format(v):>14}"
        print(f"  {lab:22}{cells}")

    print("\n  이벤트별 상세:")
    NAMES = {"2008-06": "2008 금융위기", "2011-02": "2011 유럽", "2015-02": "2015 차이나",
             "2018-04": "2018 하락", "2020-02": "2020 코로나", "2021-07": "2021 하락",
             "2021-11": "2021~22 slow", "2024-02": "2024 급락", "2025-09": "2025 하락"}
    ev_all = summ[names[0]]["evs"]
    print(f"    {'이벤트':16}{'낙폭':>7}" + "".join(f"{x:>14}" for x in names))
    for i, ev in enumerate(ev_all):
        cells = ""
        for x in names:
            s = summ[x]
            cand = [o for o in s["onset"] if abs(o - ev) <= WIN]
            cells += f"{(f'{ev - min(cand, key=lambda o: abs(o-ev)):+d}개월' if cand else 'X(놓침)'):>14}"
        print(f"    {NAMES.get(yms[ev], yms[ev]):16}{dd6[ev]:>6.0f}%{cells}")

    rows = []
    for name, (P, prisk, reg, start, _m) in res.items():
        for t in range(n):
            d = dict(model=name, ym=yms[t], prisk=prisk[t], regime=reg[t], ret=ret[t], dd6=dd6[t])
            for k in range(P.shape[1]):
                d[f"P_S{k}"] = P[t, k]
            rows.append(d)
    pd.DataFrame(rows).to_csv(OUT / "hsmm_3state_path.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(staterows).to_csv(OUT / "hsmm_3state_states.csv", index=False, encoding="utf-8-sig")
    print(f"\n  경로 → {OUT / 'hsmm_3state_path.csv'}")
    print(f"  상태 → {OUT / 'hsmm_3state_states.csv'}")
    print("\n※ 채택 판단은 2022 탐지 여부만이 아니라 상태의 경제적 반복성·리프트·FP를 함께 볼 것.")
    print("  2004~2016 구간은 생존편향(상폐 누락)이 커서 성과 수치 인용 불가 — 국면 구조 확인용.")


if __name__ == "__main__":
    main()
