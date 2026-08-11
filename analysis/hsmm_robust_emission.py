"""
analysis/hsmm_robust_emission.py

강건 emission 추정 실험 — 피처·상태수 그대로, **추정량만** 교체. production 미수정.

■ 진단 (이번에 새로 밝혀진 근본 원인)
  같은 조건인데 판정이 정반대였다.
    2018-05~2020-03 분산형 하락  breadth 0.265 / newlow 0.059 / trend -0.044 → P_bear 0.868, Bear 22/23
    2021-10~2022-09 분산형 하락  breadth 0.237 / newlow 0.063 / trend -0.108 → P_bear 0.129, Bear  0/12
  후자가 오히려 더 약세인데 놓쳤다. 피처 문제가 아니다.

  원인은 **Bear 상태 정의의 이동**:
    2020-01 재학습  Bear 배정 24개월, newlow 평균 0.037 (>0.15는 1/24)  ← 완만한 약세
    2022-01 재학습  Bear 배정  8개월, newlow 평균 0.158 (>0.15는 3/8)   ← 급락
    8개월 구성: 2017-02,03,04,07 / 2018-06 / 2020-01,02 / 2021-11
    → 2018-06~2019-03이 Bear에서 빠졌다. COVID(2020-02, newlow 0.466)가 들어와 평균을 끌어올림.
      최대값 1개만 빼도 0.158 → 0.114.

  가우시안 혼합의 평균은 극단 관측치에 취약하다. 60개월 창에 -20% 급락이 하나 들어오면
  그것이 상태 정의를 지배한다. **피처가 아니라 추정량의 문제.**

■ 처방 — Student-t emission (표준 강건 혼합)
  가우시안 대신 다변량 t를 쓴다. EM의 M-step에서 관측치별 가중이 자동으로 붙는다.
      u_i = (ν + d) / (ν + maha_i)
  maha가 큰 관측치(COVID)는 가중이 줄어 평균을 덜 흔든다. ν→∞면 가우시안과 동일.
  ν=4면 maha 30인 관측치 가중이 0.21로 떨어진다(전형적 관측치는 1.0).

  ★ 앞서 실패한 시도와의 차이
    ever_newlow_20 / pct60 : **피처를 바꿔** crash 신호까지 잃음
    강건 추정              : **피처는 그대로**, 추정량만 강건화 → crash는 여전히 crash

■ 사전 예측 (이 가설의 시험대)
  두 구간의 피처가 사실상 동일하므로, 추정이 강건해지면 **2018-20과 2021-22가 같은 판정**을
  받아야 한다. 안 되면 이 가설도 기각이다.

■ ★ 판정에는 반드시 null 대조군을 쓴다
  Sharpe는 노출 스케일에 불변이라 고정노출은 언제나 0.73. 즉
      Sharpe > 0.73 → 타이밍이 가치를 더함 /  < 0.73 → 가치를 파괴함
  K=3 실험이 2022만 보고 채택될 뻔한 것을 이 검정이 막았다(K=3 Sharpe 0.53).

■ 사용 / 산출
  .venv/bin/python analysis/hsmm_robust_emission.py
  analysis/results/hsmm_robust_emission.csv
"""
import sys
import argparse
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent.parent
A_DIR = Path(__file__).parent
OUT = A_DIR / "results"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE))
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ModuleNotFoundError:
    pass

_sp = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_sp); sys.modules["hsmm_final"] = HF; _sp.loader.exec_module(HF)

EPS = HF.EPS
SEEDS = [0, 1, 7, 42, 123]
A_ST, A_EN = "2018-05", "2020-03"      # 2018-20 분산형 (production이 잡은 구간)
B_ST, B_EN = "2021-10", "2022-09"      # 2021-22 분산형 (놓친 구간)
_U = {"w": None}                        # emis_logB가 계산한 강건 가중을 m_step에 전달


def make_t_emis(nu):
    def logB(X, means, covs):
        n, d = X.shape; K = len(means)
        out = np.zeros((n, K)); U = np.zeros((n, K))
        c = gammaln((nu + d) / 2) - gammaln(nu / 2) - (d / 2) * np.log(nu * np.pi)
        for k in range(K):
            L = np.linalg.cholesky(covs[k])
            sol = np.linalg.solve(L, (X - means[k]).T)
            maha = (sol ** 2).sum(0)
            out[:, k] = c - np.log(np.diag(L)).sum() - ((nu + d) / 2) * np.log1p(maha / nu)
            U[:, k] = (nu + d) / (nu + maha)          # 강건 가중
        _U["w"] = U
        return out
    return logB


def make_t_mstep(nu):
    def m_step(X, gamma, w):
        n, d = X.shape; K = gamma.shape[1]
        U = _U["w"] if (_U["w"] is not None and _U["w"].shape == gamma.shape) else np.ones_like(gamma)
        means = np.zeros((K, d)); covs = np.zeros((K, d, d))
        for k in range(K):
            r = gamma[:, k] * w
            ru = r * U[:, k]                          # ★ 극단 관측치 가중 축소
            R = r.sum() + EPS; RU = ru.sum() + EPS
            mu = (ru[:, None] * X).sum(0) / RU
            Xc = X - mu
            C = (ru[:, None] * Xc).T @ Xc / R
            hard = gamma[:, k] > 0.5
            delta = float(LedoitWolf().fit(X[hard]).shrinkage_) if hard.sum() >= d + 2 else 0.5
            mt = np.trace(C) / d
            covs[k] = (1 - delta) * C + delta * mt * np.eye(d) + 1e-6 * np.eye(d)
            means[k] = mu
        return means, covs
    return m_step


def walk_forward(df, yms, n, mode, nu, seed):
    """production walk_forward와 동일 골격. emission 추정량만 교체."""
    start = yms.index(HF.DECIDE_START)
    TRz = HF.roll_z(df, HF.TRAN_COLS).values
    EM = df[HF.EMIS_COLS].values
    stress_raw = TRz[:, 0] - TRz[:, 1]
    pbear = np.full(n, np.nan); bear_info = {}
    o_logB, o_mstep, o_seed = HF.emis_logB, HF.m_step_emis, HF.SEED
    try:
        HF.SEED = seed
        if mode == "t":
            HF.emis_logB = make_t_emis(nu); HF.m_step_emis = make_t_mstep(nu)
        params, sc, last = None, None, -10 ** 9
        for t in range(start, n):
            lo = max(0, t + 1 - HF.WINDOW_M); Xr = EM[lo:t + 1]; nw = len(Xr)
            w = 0.5 ** ((nw - 1 - np.arange(nw)) / HF.HL)
            if params is None or (t - last) >= HF.REFIT_EVERY:
                sc = StandardScaler().fit(Xr)
                params = HF.fit_emission(sc.transform(Xr), w,
                                         HF.cold_emission(sc.transform(Xr)) if params is None else params,
                                         40 if params is None else 10)
                last = t
                lb = HF.emis_logB(sc.transform(Xr), params["means"], params["covs"])
                g, _ = HF.forward_backward(lb, params["Amat"], params["pi"], w)
                b = int(np.argmax(HF.bear_score(params["means"])))
                hard = g.argmax(1)
                nl = df.newlow.values[lo:t + 1][hard == b]
                bear_info[yms[t]] = (int((hard == b).sum()), float(nl.mean()) if len(nl) else np.nan)
            Xz = sc.transform(Xr)
            logB = HF.emis_logB(Xz, params["means"], params["covs"])
            bear = int(np.argmax(HF.bear_score(params["means"])))
            gamma, _ = HF.forward_backward(logB, params["Amat"], params["pi"], w)
            durs = HF.state_durations(gamma.argmax(1))
            haz = np.vstack([HF.to_hazard(HF.dur_pmf(durs[0])), HF.to_hazard(HF.dur_pmf(durs[1]))])
            sw = stress_raw[lo:t + 1]; sw = (sw - sw.mean()) / (sw.std() + EPS)
            filt = HF.hsmm_filter(logB, sw, haz, bear, params["pi"])
            pbear[t] = filt[-1, bear]
    finally:
        HF.emis_logB, HF.m_step_emis, HF.SEED = o_logB, o_mstep, o_seed
        _U["w"] = None
    # ★ production과 동일한 EMA 스무딩 (hsmm_final.main). 빠뜨리면 raw로 평가돼 기준선이 어긋난다.
    sm = pbear.copy()
    for t in range(start + 1, n):
        sm[t] = HF.PBEAR_EMA * pbear[t] + (1 - HF.PBEAR_EMA) * sm[t - 1]
    return sm, start, bear_info


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


def perf(r):
    r = np.asarray(r, dtype=float); r = r[~np.isnan(r)]
    c = np.cumprod(1 + r); y = len(r) / 12
    v = r.std() * np.sqrt(12)
    return dict(cagr=c[-1] ** (1 / y) - 1, sharpe=(r.mean() * 12) / (v + 1e-12),
                mdd=float((c / np.maximum.accumulate(c) - 1).min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, nargs="+", default=[4.0, 8.0])
    args = ap.parse_args()

    df, yms, n, ret, rvol, dvol, dd6 = pd.read_pickle(A_DIR / ".cache" / "hsmm_features.pkl")
    F = pd.read_csv(A_DIR / "fcf_overlay_series.csv", encoding="utf-8-sig").set_index("ym")
    bench = F["bench"]; prod_exp = F["expB"]
    D6 = pd.Series(dd6, index=yms)

    def blk(pb, a, b):
        x = pd.Series(pb, index=yms).loc[a:b]
        return x.mean(), int((x >= 0.6).sum()), len(x)

    def metrics(pb, start):
        idx = list(range(start, n))
        reg = ["Bull"] * n; p = "Bull"
        for t in idx:
            p = ("Bear" if pb[t] >= HF.T_OUT else "Bull") if p == "Bear" else ("Bear" if pb[t] >= HF.T_IN else "Bull")
            reg[t] = p
        ok = [t for t in idx if not np.isnan(dd6[t])]
        tp = sum(1 for t in ok if dd6[t] <= -15 and reg[t] == "Bear")
        fn = sum(1 for t in ok if dd6[t] <= -15 and reg[t] != "Bear")
        fp = sum(1 for t in ok if dd6[t] > -15 and reg[t] == "Bear")
        prec = tp / (tp + fp) if tp + fp else np.nan
        base = (tp + fn) / len(ok)
        return dict(lift=prec / base if base else np.nan, fp=fp,
                    bear_ratio=(tp + fp) / len(ok))

    print("=" * 96)
    print("  ★ 사전 예측: 강건화하면 2018-20과 2021-22가 같은 판정을 받아야 한다")
    print("=" * 96)
    print(f"  {'모델':16}{'2018-20 P':>12}{'Bear월':>8}{'2021-22 P':>12}{'Bear월':>8}"
          f"{'격차':>8}   Bear상태 newlow평균(2022-01 재적합)")
    rows = []
    pb0, start, bi0 = walk_forward(df, yms, n, "gauss", None, 42)
    a0 = blk(pb0, A_ST, A_EN); b0 = blk(pb0, B_ST, B_EN)
    print(f"  {'production':16}{a0[0]:12.3f}{a0[1]:>4}/{a0[2]:<3}{b0[0]:12.3f}{b0[1]:>4}/{b0[2]:<3}"
          f"{a0[0]-b0[0]:+8.3f}   {bi0.get('2022-01',(0,np.nan))[1]:.3f} ({bi0.get('2022-01',(0,0))[0]}개월)")
    store = {"production": (pb0, start)}
    for nu in args.nu:
        pb, st, bi = walk_forward(df, yms, n, "t", nu, 42)
        a = blk(pb, A_ST, A_EN); b = blk(pb, B_ST, B_EN)
        nm = f"t-emission ν={nu:g}"
        print(f"  {nm:16}{a[0]:12.3f}{a[1]:>4}/{a[2]:<3}{b[0]:12.3f}{b[1]:>4}/{b[2]:<3}"
              f"{a[0]-b[0]:+8.3f}   {bi.get('2022-01',(0,np.nan))[1]:.3f} ({bi.get('2022-01',(0,0))[0]}개월)")
        store[nm] = (pb, st)

    print(f"\n{'='*96}\n  ★ null 대조군 검정 (고정노출 Sharpe = 0.73. 넘어야 타이밍 가치 있음)\n{'='*96}")
    m_bm = perf(bench)
    print(f"  {'전략':22}{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'평균노출':>9}{'리프트':>8}{'Bear비율':>9}")
    print(f"  {'FCF불 단독':22}{m_bm['cagr']:8.1%}{m_bm['sharpe']:9.2f}{m_bm['mdd']:9.1%}{1.0:9.2f}")
    mp = perf(bench * prod_exp)
    print(f"  {'고정노출 0.66':22}{perf(bench*0.66)['cagr']:8.1%}{perf(bench*0.66)['sharpe']:9.2f}"
          f"{perf(bench*0.66)['mdd']:9.1%}{0.66:9.2f}")
    for nm, (pb, st) in store.items():
        e = exposure_from_p(np.nan_to_num(pb, nan=0.0), dvol, n, st)
        E = pd.Series(e, index=yms).reindex(F.index)
        m = perf(bench * E); mt = metrics(pb, st)
        print(f"  {nm:22}{m['cagr']:8.1%}{m['sharpe']:9.2f}{m['mdd']:9.1%}{E.mean():9.2f}"
              f"{mt['lift']:8.2f}{mt['bear_ratio']:9.0%}")
        rows.append(dict(model=nm, seed=42, **m, exp=float(E.mean()), **mt))

    print(f"\n{'='*96}\n  시드 안정성 (t-emission)\n{'='*96}")
    print(f"  {'모델':16}{'시드':>5}{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'2021-22 P':>12}")
    for nu in args.nu:
        for sd in SEEDS:
            pb, st, _ = walk_forward(df, yms, n, "t", nu, sd)
            e = exposure_from_p(np.nan_to_num(pb, nan=0.0), dvol, n, st)
            E = pd.Series(e, index=yms).reindex(F.index)
            m = perf(bench * E)
            print(f"  {'ν=%g'%nu:16}{sd:>5}{m['cagr']:8.1%}{m['sharpe']:9.2f}{m['mdd']:9.1%}"
                  f"{blk(pb,B_ST,B_EN)[0]:12.3f}")
            rows.append(dict(model=f"t ν={nu:g}", seed=sd, **m, exp=float(E.mean())))
    pd.DataFrame(rows).to_csv(OUT / "hsmm_robust_emission.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT/'hsmm_robust_emission.csv'}")
    print("\n※ production 미수정. 판정은 반드시 Sharpe>0.73(null) 과 함께 볼 것.")


if __name__ == "__main__":
    main()
