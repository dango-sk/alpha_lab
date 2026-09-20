# -*- coding: utf-8 -*-
"""analysis/hsmm_v5_bounded.py — V5: newlow 축 bounded-influence 학습. production 미수정.

사양: analysis/HSMM_V5_SPEC.md

■ 한 줄 요약
  newlow 의 Bear 방향 정보(부호)는 전부 유지하고, **극단적 크기가 상태 평균을 끌고 가는
  영향력만** 사전 고정 상한 c 로 자른다(Huber M-estimator).

■ 왜 z=0 강제가 아닌가 (기각된 대안)
  "newlow↑ & breadth↓ 는 노이즈로 안 보낸다"는 규칙은 COVID(newlow 0.466, breadth 0.117)가
  정확히 그 조건이라, z=0 → 학습 기여 100% 복원 → V4 가 해결한 평균 오염이 그대로 재발한다.
  **0 이냐 1 이냐의 이분법 자체가 원인이다.** V5 는 그 사이를 c 로 연속적으로 다룬다.

■ V4(노이즈) 감사가 준 근거
  z>0.5 격리월 = 전체 newlow 상위 10개월 중 9개 → 노이즈가 지운 건 극단값이 아니라
  **newlow 위험축 자체**였다. Bear newlow 평균 0.158 → 0.048 로 소거되고 2026-05
  스파이크(0.168)에 무반응 → 익월 −22% 미탐지.
  γ(분산)는 원인이 아니었다(분산비 최대 5.09, 2026-01 은 0.44). π_max 미발동.

■ ★ 판정 순서 — 진단 먼저, 성과는 그다음
  성과부터 보면 상시 저노출의 부수효과에 속는다(t·V4 에서 두 번 반복). 아래 4개를 먼저 본다.
    1) COVID 이후 Bear newlow 평균 : 낮아지되 소거되지 않을 것 (목표 0.07~0.12)
    2) 2022 slow bear 탐지         : production 0.129 → ≥ 0.5
    3) 2026-05 신호                : V4 0.021 → ≥ 0.4
    4) Bear 점유율·분별력          : bear_ratio ≤ 45%, 리프트 ≥ 1.2, bear_ret_pos ≤ 0.45

■ 사용
  .venv/bin/python analysis/hsmm_v5_bounded.py                       # 진단만
  .venv/bin/python analysis/hsmm_v5_bounded.py --perf                # 진단 후 성과까지
  .venv/bin/python analysis/hsmm_v5_bounded.py --c 1 1.5 2 3 --axes newlow
"""
import sys
import argparse
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

A_DIR = Path(__file__).parent
OUT = A_DIR / "results"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(A_DIR.parent))
_sp = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_sp); sys.modules["hsmm_final"] = HF; _sp.loader.exec_module(HF)

EPS = HF.EPS
TC = 0.0030
B_ST, B_EN = "2021-10", "2022-09"      # slow bear (production 이 놓친 구간)


def consistency(c):
    """s = sqrt(E[clip(Z,−c,c)²]), Z~N(0,1). winsorize 로 인한 분산 축소편향 보정."""
    if not np.isfinite(c):
        return 1.0
    v = (2 * norm.cdf(c) - 1) - 2 * c * norm.pdf(c) + 2 * c * c * (1 - norm.cdf(c))
    return float(np.sqrt(max(v, 1e-6)))


def pd_guard(C, floor=1e-8):
    """대칭화 + 고유값 바닥. winsorized 편차로 만든 C 가 수치적으로 준-특이해질 수 있어
    production 의 1e-6 ridge 만으로는 cholesky 가 깨지는 경우가 있다(실측 확인)."""
    C = (C + C.T) / 2
    ev, V = np.linalg.eigh(C)
    if ev.min() >= floor:
        return C
    return V @ np.diag(np.maximum(ev, floor)) @ V.T


def fit_bounded(X, w, init, n_iter, c, axes, corr=True):
    """production fit_emission 과 동일. M-step 의 평균·공분산 기여만 유계화."""
    means, covs = init["means"].copy(), init["covs"].copy()
    Amat, pi = init["Amat"].copy(), init["pi"].copy()
    n, d = X.shape
    s = np.ones(d)
    if corr:
        for j in axes:
            s[j] = consistency(c)
    gamma = None; clipped = np.zeros(2)
    aset = set(axes)
    for _ in range(n_iter):
        logB = HF.emis_logB(X, means, covs)              # ★ 밀도·디코딩은 production 그대로
        if not np.isfinite(logB).all():
            break
        gamma, xi = HF.forward_backward(logB, Amat, pi, w)
        pi = gamma[0] / (gamma[0].sum() + EPS)
        Amat = xi / (xi.sum(axis=1, keepdims=True) + EPS)
        for k in range(2):
            r = gamma[:, k] * w; R = r.sum() + EPS
            D = X - means[k]
            # ── 평균: Huber W-estimator (IRLS 가중평균 형태) ──
            #   ★ 덧셈형 스텝 μ ← μ + Σr·clip(D)/Σr 은 **발산한다**: 상태 평균이 데이터에서
            #     멀면 전 관측치가 ±c 로 잘려 매 반복 c 씩 같은 방향으로 행진한다(실측: LinAlgError).
            #     가중평균 형태는 갱신값이 데이터 볼록껍질 안에 갇혀 구조적으로 발산할 수 없다.
            #     영향함수는 동일하게 c 로 유계이고, 부호(방향)는 그대로 보존된다.
            mu = np.empty(d)
            for j in range(d):
                if j in aset:
                    u = np.minimum(1.0, c / np.maximum(np.abs(D[:, j]), 1e-12))   # Huber 가중
                    rr = r * u
                    mu[j] = (rr * X[:, j]).sum() / (rr.sum() + EPS)
                else:
                    mu[j] = (r * X[:, j]).sum() / R
            clipped[k] = float((np.abs(D[:, list(axes)]) > c).any(1).sum())
            # ── 공분산: 같은 상한을 적용한 편차 + 축소편향 일관성 보정 ──
            Dc = X - mu
            for j in axes:
                Dc[:, j] = np.clip(Dc[:, j], -c, c)
            Dc = Dc / s
            C = (r[:, None] * Dc).T @ Dc / R
            hard = gamma[:, k] > 0.5                      # production 과 동일한 축소 표본 선택
            delta = float(LedoitWolf().fit(X[hard]).shrinkage_) if hard.sum() >= d + 2 else 0.5
            Ck = (1 - delta) * C + delta * (np.trace(C) / d) * np.eye(d) + 1e-6 * np.eye(d)
            if not (np.isfinite(Ck).all() and np.isfinite(mu).all()):
                # c 가 너무 작으면(예: c=1.0, 2024-01) 제한 축 분산이 붕괴해 비유한 값이 난다.
                # 그 반복을 버리고 직전 유효 파라미터를 유지 → 상위에서 REFIT_FAILED 로 집계.
                return dict(means=means, covs=covs, Amat=Amat, pi=pi, gamma=gamma,
                            clipped=clipped, ok=False)
            covs[k] = pd_guard(Ck)
            means[k] = mu
    return dict(means=means, covs=covs, Amat=Amat, pi=pi, gamma=gamma, clipped=clipped,
                ok=(gamma is not None))


def walk_forward(df, yms, n, ret, c, axes, seed, label):
    start = yms.index(HF.DECIDE_START)
    TRz = HF.roll_z(df, HF.TRAN_COLS).values
    EM = df[HF.EMIS_COLS].values
    stress_raw = TRz[:, 0] - TRz[:, 1]
    pbear = np.full(n, np.nan); diag = []; n_failed = 0
    o_seed = HF.SEED
    try:
        HF.SEED = seed
        params, sc, last = None, None, -10 ** 9
        for t in range(start, n):
            lo = max(0, t + 1 - HF.WINDOW_M); Xr = EM[lo:t + 1]; nw = len(Xr)
            w = 0.5 ** ((nw - 1 - np.arange(nw)) / HF.HL)
            if params is None or (t - last) >= HF.REFIT_EVERY:
                sc = StandardScaler().fit(Xr); Z = sc.transform(Xr)
                init = HF.cold_emission(Z) if params is None else params
                cand = fit_bounded(Z, w, init, 40 if params is None else 10, c, axes)
                if not cand.get("ok") or cand["gamma"] is None:
                    n_failed += 1                        # 직전 파라미터 유지(웜스타트 계속)
                    if params is None:
                        raise RuntimeError(f"{label}: 최초 적합 실패 ({yms[t]}) — c 가 너무 작다")
                    last = t
                else:
                    params = cand; last = t
                g = params["gamma"]; b = int(np.argmax(HF.bear_score(params["means"])))
                hard = g.argmax(1)
                nl = df.newlow.values[lo:t + 1][hard == b]
                rw = np.array([ret[i] for i in range(lo, t)]); mr = hard[:len(rw)] == b
                mu_raw = sc.inverse_transform(params["means"])
                diag.append(dict(model=label, c=c, refit_ym=yms[t], bear_months=int((hard == b).sum()),
                                 bear_share=float((hard == b).mean()),
                                 bear_newlow=float(nl.mean()) if len(nl) else np.nan,
                                 bear_ret_pos=float((rw[mr] > 0).mean()) if mr.any() else np.nan,
                                 mu_bear_newlow=float(mu_raw[b, 1]),
                                 mu_bull_newlow=float(mu_raw[1 - b, 1]),
                                 clipped=float(params["clipped"][b])))
            Xz = sc.transform(Xr)
            logB = HF.emis_logB(Xz, params["means"], params["covs"])
            bear = int(np.argmax(HF.bear_score(params["means"])))
            gamma, _ = HF.forward_backward(logB, params["Amat"], params["pi"], w)
            durs = HF.state_durations(gamma.argmax(1))
            haz = np.vstack([HF.to_hazard(HF.dur_pmf(durs[0])), HF.to_hazard(HF.dur_pmf(durs[1]))])
            sw = stress_raw[lo:t + 1]; sw = (sw - sw.mean()) / (sw.std() + EPS)
            pbear[t] = HF.hsmm_filter(logB, sw, haz, bear, params["pi"])[-1, bear]
    finally:
        HF.SEED = o_seed
    sm = pbear.copy()
    for t in range(start + 1, n):
        sm[t] = HF.PBEAR_EMA * pbear[t] + (1 - HF.PBEAR_EMA) * sm[t - 1]
    if n_failed:
        print(f"  ⚠ {label}: 재적합 실패 {n_failed}회 — 직전 파라미터 유지. c 가 너무 작다는 신호.")
    return sm, start, pd.DataFrame(diag)


def exposure_from_p(p, dvol, n, start):
    cur = np.maximum(dvol, HF.VOL_FLOOR)
    tgt = np.full(n, HF.TARGET_VOL, dtype=float)
    tgt[start:] = np.cumsum(dvol[start:]) / np.arange(1, n - start + 1)
    cut = 1.0 - np.minimum(1.0, tgt / cur)
    raw = np.clip((1 - p) * (1.0 - p * cut), HF.EXP_FLOOR, 1.0)
    e = raw.copy(); held = None
    for t in range(start, n):
        if held is None or abs(raw[t] - held) >= HF.REBAL_BAND:
            held = round(raw[t] / 0.05) * 0.05
        e[t] = min(max(held, HF.EXP_FLOOR), 1.0)
    return e


def perf(r, e=None):
    r = np.asarray(r, dtype=float); ok = ~np.isnan(r); r = r[ok]
    if e is not None:
        e = np.asarray(e, dtype=float)[ok]
        r = r - np.abs(np.diff(np.concatenate([[e[0]], e]))) * TC
    cq = np.cumprod(1 + r); y = len(r) / 12
    v = r.std() * np.sqrt(12); mdd = float((cq / np.maximum.accumulate(cq) - 1).min())
    cagr = cq[-1] ** (1 / y) - 1
    return dict(cagr=cagr, sharpe=(r.mean() * 12) / (v + 1e-12), mdd=mdd,
                calmar=cagr / abs(mdd) if mdd else np.nan)


def detect(pb, start, n, dd6):
    idx = list(range(start, n)); reg = ["Bull"] * n; p = "Bull"
    for t in idx:
        p = ("Bear" if pb[t] >= HF.T_OUT else "Bull") if p == "Bear" else ("Bear" if pb[t] >= HF.T_IN else "Bull")
        reg[t] = p
    ok = [t for t in idx if not np.isnan(dd6[t])]
    tp = sum(1 for t in ok if dd6[t] <= -15 and reg[t] == "Bear")
    fn = sum(1 for t in ok if dd6[t] <= -15 and reg[t] != "Bear")
    fp = sum(1 for t in ok if dd6[t] > -15 and reg[t] == "Bear")
    prec = tp / (tp + fp) if tp + fp else np.nan
    base = (tp + fn) / len(ok) if ok else np.nan
    return dict(lift=prec / base if base else np.nan, fp=fp,
                bear_ratio=(tp + fp) / len(ok) if ok else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--axes", default="newlow", choices=["newlow", "all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perf", action="store_true", help="진단 통과 여부와 무관하게 성과까지 출력")
    args = ap.parse_args()
    axes = ([HF.EMIS_COLS.index("newlow")] if args.axes == "newlow" else list(range(len(HF.EMIS_COLS))))

    df, yms, n, ret, rvol, dvol, dd6 = pd.read_pickle(A_DIR / ".cache" / "hsmm_features.pkl")
    F = pd.read_csv(A_DIR / "fcf_overlay_series.csv", encoding="utf-8-sig").set_index("ym")
    bench = F["bench"]

    runs, diags = {}, []
    for nm, c in [("production", np.inf)] + [(f"V5 c={c:g}", c) for c in args.c]:
        pb, st, dg = walk_forward(df, yms, n, ret, c, axes, args.seed, nm)
        runs[nm] = (pb, st); diags.append(dg)
    D = pd.concat(diags, ignore_index=True)
    S = {nm: pd.Series(pb, index=yms) for nm, (pb, st) in runs.items()}

    print("=" * 104)
    print(f"  HSMM V5 — newlow 축 bounded-influence (제한 축: {args.axes})")
    print("  ★ 진단 먼저. 4개 통과한 c 만 성과 평가로 넘긴다.")
    print("=" * 104)

    print("\n  [진단 1] COVID 이후 Bear 상태 newlow 평균 — 낮아지되 소거되지 않을 것 (목표 0.07~0.12)")
    print(f"  {'모델':14}" + "".join(f"{y:>10}" for y in ["2020-01", "2022-01", "2023-01", "2025-01", "2026-01"]))
    for nm in runs:
        row = D[D.model == nm].set_index("refit_ym")
        print(f"  {nm:14}" + "".join(f"{row.bear_newlow.get(y, np.nan):10.3f}"
                                     for y in ["2020-01", "2022-01", "2023-01", "2025-01", "2026-01"]))
    print("  참고: production 0.037/0.158/0.193/0.234/0.159   V4 노이즈 0.037/0.048/0.053/0.051/0.046(소거)")

    print("\n  [진단 2] 2022 slow bear 탐지 — 2021-10~2022-09 평균 P_bear (production 0.129 → ≥0.50)")
    print("  [진단 3] 2026-05 신호 — newlow 0.168 스파이크 (V4 0.021 → ≥0.40, production 0.503)")
    print(f"\n  {'모델':14}{'진단2 P':>10}{'Bear월':>9}{'진단3 2026-05':>14}{'2026-06':>10}"
          f"{'2020-02':>10}{'2022-06':>10}")
    for nm, s in S.items():
        b = s.loc[B_ST:B_EN]
        print(f"  {nm:14}{b.mean():10.3f}{int((b >= 0.6).sum()):>5}/{len(b):<3}"
              f"{s.get('2026-05', np.nan):14.3f}{s.get('2026-06', np.nan):10.3f}"
              f"{s.get('2020-02', np.nan):10.3f}{s.get('2022-06', np.nan):10.3f}")

    print("\n  [진단 4] Bear 점유율·분별력 (bear_ratio ≤45%, 리프트 ≥1.2, bear_ret_pos ≤0.45)")
    print(f"  {'모델':14}{'bear_share':>12}{'bear_ret_pos':>14}{'bear_ratio':>12}{'리프트':>9}"
          f"{'FP':>5}{'clip월/창':>11}")
    verdict = {}
    for nm, (pb, st) in runs.items():
        row = D[D.model == nm]
        dt = detect(pb, st, n, dd6)
        bs, brp, cl = row.bear_share.mean(), row.bear_ret_pos.mean(), row.clipped.mean()
        print(f"  {nm:14}{bs:12.2f}{brp:14.2f}{dt['bear_ratio']:12.0%}{dt['lift']:9.2f}"
              f"{dt['fp']:5.0f}{cl:11.1f}")
        nlm = D[(D.model == nm) & (D.refit_ym.isin(["2022-01", "2023-01"]))].bear_newlow.mean()
        d2 = S[nm].loc[B_ST:B_EN].mean(); d3 = S[nm].get("2026-05", np.nan)
        verdict[nm] = dict(d1=bool(0.07 <= nlm <= 0.12), d2=bool(d2 >= 0.50), d3=bool(d3 >= 0.40),
                           d4=bool(dt["bear_ratio"] <= 0.45 and dt["lift"] >= 1.2 and brp <= 0.45),
                           nlm=nlm, p2=d2, p3=d3, **dt, bear_ret_pos=brp)

    print(f"\n{'='*104}\n  진단 판정\n{'='*104}")
    print(f"  {'모델':14}{'1 newlow평균':>14}{'2 slowbear':>12}{'3 2026-05':>11}{'4 분별력':>10}{'통과':>8}")
    passed = []
    for nm, v in verdict.items():
        k = sum(v[f"d{i}"] for i in (1, 2, 3, 4))
        mark = lambda b: "  ○" if b else "  ×"
        print(f"  {nm:14}{mark(v['d1']):>14}{mark(v['d2']):>12}{mark(v['d3']):>11}{mark(v['d4']):>10}{k:>6}/4")
        if k == 4 and nm != "production":
            passed.append(nm)
    D.to_csv(OUT / "hsmm_v5_diag.csv", index=False, encoding="utf-8-sig")

    if not passed and not args.perf:
        print("\n  → 4개 전부 통과한 c 가 없다. **성과는 보지 않는다** (상시 저노출 부수효과에 속지 않기 위해).")
        print("     전체 성과를 굳이 보려면 --perf 를 붙일 것.")
    else:
        print(f"\n{'='*104}\n  성과 (거래비용 30bp) — null 대조군은 같은 평균노출의 상수노출\n{'='*104}")
        print(f"  {'전략':22}{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'Calmar':>9}{'평균노출':>9}{'리프트':>8}")
        m = perf(bench)
        print(f"  {'FCF 단독':22}{m['cagr']:8.1%}{m['sharpe']:9.2f}{m['mdd']:9.1%}{m['calmar']:9.2f}{1.0:9.2f}")
        rows = []
        for nm, (pb, st) in runs.items():
            e = exposure_from_p(np.nan_to_num(pb, nan=0.0), dvol, n, st)
            # ★ 인과 정렬: 당월 수익 × **전월말** 노출. shift(1) 을 빠뜨리면 lookahead 다
            #   (fcf_overlay_series.csv 의 expB 가 이미 전월말 노출이라 그것과 대조하면 바로 드러난다)
            E = pd.Series(e, index=yms).shift(1).reindex(F.index)
            mm = perf(bench * E, E.values); mn = perf(bench * E.mean(), np.full(len(bench), E.mean()))
            dt = detect(pb, st, n, dd6)
            print(f"  {('  [null] 상수 %.2f' % E.mean()):22}{mn['cagr']:8.1%}{mn['sharpe']:9.2f}"
                  f"{mn['mdd']:9.1%}{mn['calmar']:9.2f}{E.mean():9.2f}")
            print(f"  {nm:22}{mm['cagr']:8.1%}{mm['sharpe']:9.2f}{mm['mdd']:9.1%}{mm['calmar']:9.2f}"
                  f"{E.mean():9.2f}{dt['lift']:8.2f}")
            rows.append(dict(model=nm, **mm, exp=float(E.mean()), null_sharpe=mn["sharpe"],
                             null_calmar=mn["calmar"], **dt))
        pd.DataFrame(rows).to_csv(OUT / "hsmm_v5_metrics.csv", index=False, encoding="utf-8-sig")

    P = pd.DataFrame({"ym": yms})
    for nm, s in S.items():
        P[f"pbear::{nm}"] = s.values
    P.to_csv(OUT / "hsmm_v5_path.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT/'hsmm_v5_diag.csv'}  {OUT/'hsmm_v5_path.csv'}")


if __name__ == "__main__":
    main()
