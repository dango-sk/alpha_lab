"""
analysis/hsmm_newlow_veto.py

newlow veto 완화 실험 — 극단적 Gaussian 거리만 제한해 breadth/trend와 대등하게 경쟁시킨다.

■ 진단 (analysis/hsmm_3state.py 결과)
  2022-01 실측 breadth 0.134 / newlow 0.015 / trend -0.152 를 3-state의 crash 상태(S2:
  breadth 0.122 / newlow 0.295 / trend -0.150)와 비교하면

      |z| breadth 0.07   |z| trend 0.01   |z| newlow 5.59

  breadth·trend는 crash와 사실상 동일한데 **newlow 하나가 5.59σ로 진입을 거부**한다.
  가우시안 우도는 z²에 비례하므로 newlow가 단독 veto를 행사한다.
  → ever_newlow 교체(veto 제거)는 신호까지 소실시켜 포화, ma200_slope 추가는 veto가 남아 실패,
    3-state는 BIC상 지지되지만 세 번째 상태가 slow bear가 아니라 짧은 crash로 형성됐다.

■ 이번 목적
  newlow를 **제거하지 않고** 꼬리만 압축해 veto 강도를 2σ 안팎으로 제한한다.
  스파이크의 '순서'는 보존하므로 crash 탐지력은 유지되어야 한다.

■ 변형 (모두 causal — 현재 시점 이전 데이터만)
  1) baseline   : raw newlow
  2) zclip2     : rolling 36M z-score → ±2 clip        (우선순위 1)
  3) pct60      : rolling 60M percentile rank          (우선순위 2)
  4) log1p      : log1p(newlow*100)                     (보조)

■ 상태 위험도 정렬 (★ lookahead 금지)
  emission 평균이 아니라 **학습창 내부에서 그 상태에 실제로 연결됐던 향후 6M 낙폭**으로 정렬한다.
  단 dd6[i]는 i+1..i+6을 쓰므로, 재학습 시점 t에서 알 수 있는 것은 i <= t-6뿐이다.
  따라서 정렬에는 [lo, t-6] 구간만 사용한다. (표본 부족 상태는 emission 평균으로 폴백)

■ 채택 기준 (2022를 잡았다는 것만으로는 채택하지 않는다)
  1 slow bear P_risk가 baseline 대비 유의하게 상승   2 양(+)의 lead
  3 전체 lift 비악화                                  4 FP/Risk비율 과증가 없음
  5 crash 탐지 유지                                   6 새 상태가 다른 시기에도 반복

■ 사용 / 산출
  .venv/bin/python analysis/hsmm_newlow_veto.py
  analysis/results/hsmm_veto_path.csv / _states.csv / _slowbear.csv
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
LONG_PANEL = A_DIR / ".cache" / "hsmm_longrun_features.pkl"
sys.path.insert(0, str(BASE))
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ModuleNotFoundError:
    pass

import importlib.util


def _load(mod, path):
    sp = importlib.util.spec_from_file_location(mod, A_DIR / path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[mod] = m
    sp.loader.exec_module(m)
    return m


HF = _load("hsmm_final", "hsmm_final.py")
H3 = _load("hsmm_3state", "hsmm_3state.py")      # K-state 엔진 재사용(production 미수정)

EMIS, TRAN = HF.EMIS_COLS, HF.TRAN_COLS
WINDOW_M, REFIT_EVERY, HL, EPS = HF.WINDOW_M, HF.REFIT_EVERY, HF.HL, HF.EPS
PBEAR_EMA, T_IN, T_OUT, WIN = HF.PBEAR_EMA, HF.T_IN, HF.T_OUT, HF.WIN
SLOW_A, SLOW_B = "2021-11", "2022-12"
DETAIL_A, DETAIL_B = "2021-07", "2023-03"
DIAG_MONTHS = ["2022-01", "2022-08"]
K = 3


# ─────────────────── newlow 변형 (전부 causal: trailing window) ───────────────────
def t_baseline(s):
    return s.copy()


def t_zclip2(s, win=36, lo=12, cap=2.0):
    m = s.rolling(win, min_periods=lo).mean()
    sd = s.rolling(win, min_periods=lo).std().replace(0, np.nan)
    return ((s - m) / sd).fillna(0.0).clip(-cap, cap)


def t_pct60(s, win=60, lo=12):
    """과거 win개월(현재 포함) 안에서의 percentile rank. 미래 미사용."""
    def _r(x):
        return float((x[:-1] < x[-1]).mean()) if len(x) > 1 else 0.5
    return s.rolling(win, min_periods=lo).apply(_r, raw=True).fillna(0.5)


def t_log1p(s):
    return np.log1p(s * 100.0)


VARIANTS = [("baseline", t_baseline), ("zclip2", t_zclip2), ("pct60", t_pct60), ("log1p", t_log1p)]


# ─────────────────── walk-forward (인과적 상태 위험도 정렬) ───────────────────
def walk_forward(df, yms, n, dd6, decide_start, seed):
    start = yms.index(decide_start) if decide_start in yms else 12
    TRz = HF.roll_z(df, TRAN).values
    EM = df[EMIS].values
    stress_raw = TRz[:, 0] - TRz[:, 1]
    P = np.full((n, K), np.nan)
    params, sc, last_refit, order = None, None, -10 ** 9, np.arange(K)
    refits = {}          # ym -> 진단용 스냅샷
    for t in range(start, n):
        lo = max(0, t + 1 - WINDOW_M); Xr = EM[lo:t + 1]; nw = len(Xr)
        w = 0.5 ** ((nw - 1 - np.arange(nw)) / HL)
        if params is None or (t - last_refit) >= REFIT_EVERY:
            sc = StandardScaler().fit(Xr)
            params = H3.fit_emission(sc.transform(Xr), w,
                                     H3.cold_emission(sc.transform(Xr), K, seed) if params is None else params,
                                     40 if params is None else 10)
            last_refit = t
            # ── ★ 상태 위험도: 학습창 내 '이미 확정된' 향후 6M 낙폭으로만 정렬 ──
            logB_f = H3.emis_logB(sc.transform(Xr), params["means"], params["covs"])
            g_f, _ = HF.forward_backward(logB_f, params["Amat"], params["pi"], w)
            hard_f = g_f.argmax(1)
            usable = [(i, hard_f[i - lo]) for i in range(lo, t - 5) if not np.isnan(dd6[i])]
            risk_dd = np.full(K, np.nan)
            for k in range(K):
                v = [dd6[i] for i, s_ in usable if s_ == k]
                if len(v) >= 3:
                    risk_dd[k] = float(np.mean(v))
            if np.isnan(risk_dd).any():          # 표본 부족 → emission 평균 폴백
                em_r = H3.risk_scores(params["means"])
                fill = -em_r * 10.0              # 부호만 맞추면 됨(정렬용)
                risk_dd = np.where(np.isnan(risk_dd), fill, risk_dd)
            order = np.argsort(-risk_dd)         # dd6 큰(=덜 위험) 순 → S0가 최저위험
            refits[yms[t]] = dict(means=params["means"].copy(), covs=params["covs"].copy(),
                                  scaler=sc, order=order.copy(), risk_dd=risk_dd.copy(),
                                  means_orig=sc.inverse_transform(params["means"]))
        Xz = sc.transform(Xr)
        logB = H3.emis_logB(Xz, params["means"], params["covs"])
        gamma, _ = HF.forward_backward(logB, params["Amat"], params["pi"], w)
        durs = H3.state_durations(gamma.argmax(1), K)
        haz = np.vstack([HF.to_hazard(HF.dur_pmf(durs[k])) for k in range(K)])
        # 스트레스 부호도 인과적 위험도 순서를 따른다 (S0 최저위험 → -1, 최고위험 → +1)
        r_norm = np.zeros(K)
        for rank, k in enumerate(order):
            r_norm[k] = -1.0 + 2.0 * rank / (K - 1)
        Aex = params["Amat"].copy(); np.fill_diagonal(Aex, 0.0)
        Aex = Aex / (Aex.sum(axis=1, keepdims=True) + EPS)
        sw = stress_raw[lo:t + 1]; sw = (sw - sw.mean()) / (sw.std() + EPS)
        filt = H3.hsmm_filter(logB, sw, haz, params["pi"], Aex, r_norm)
        row = filt[-1][order]                    # 위험도 오름차순으로 재배열
        if not np.all(np.isfinite(row)) or row.sum() <= 0:
            row = P[t - 1] if t > start and np.all(np.isfinite(P[t - 1])) else np.full(K, 1.0 / K)
        P[t] = row / row.sum()
    return P, start, refits


# 위험도 가중 집계: w = [0, 0.5, 1] (위험도 순위 기반).
#   K=2면 w=[0,1] → production의 P_bear와 동일한 정의가 된다.
#   1-P(S0)은 K=3에서 '가장 안전한 상태가 아님'이 되어 Risk비율이 60~98%로 무의미해진다.
RISK_W = np.arange(K) / (K - 1)


def aggregate_risk(P, n, start):
    pr = np.full(n, np.nan)
    for t in range(start, n):
        pr[t] = float(P[t] @ RISK_W)
    return pr


def to_regime(pr, n, start):
    reg = ["Benign"] * n; p = "Benign"
    for t in range(start, n):
        p = ("Risk" if pr[t] >= T_OUT else "Benign") if p == "Risk" else ("Risk" if pr[t] >= T_IN else "Benign")
        reg[t] = p
    return reg


# ─────────────────── 진단: feature별 거리 기여 ───────────────────
def contribution(refits, ym, x_vec, yms):
    """해당 월에 유효한 마지막 재학습 파라미터로 state별 feature 거리 기여를 분해."""
    keys = [k for k in sorted(refits) if k <= ym]
    if not keys:
        return None
    R = refits[keys[-1]]
    z = R["scaler"].transform(x_vec.reshape(1, -1))[0]
    rows = []
    for rank, k in enumerate(R["order"]):
        mu, C = R["means"][k], R["covs"][k]
        sd = np.sqrt(np.diag(C))
        per = (z - mu) / sd                       # 대각 기준 표준화 거리(해석용)
        Ci = np.linalg.inv(C)
        maha = float((z - mu) @ Ci @ (z - mu))
        rows.append(dict(state=f"S{rank}", per=np.abs(per), maha=maha,
                         mu_orig=R["means_orig"][k]))
    return keys[-1], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide-start", default="2009-01")
    ap.add_argument("--thr", type=float, default=-15.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    thr = -abs(args.thr)

    if not LONG_PANEL.exists():
        print(f"!! 장기 패널 없음: {LONG_PANEL}\n   먼저 analysis/hsmm_longrun.py 실행")
        return
    df0, yms, n, ret, _rv, _dv, dd6, _raw = pd.read_pickle(LONG_PANEL)
    print(f"장기 패널 {yms[0]} ~ {yms[-1]} ({n}개월)  판정 {args.decide_start}  K={K}")
    print("covariance 정상 초기화 / 상태 위험도 = 학습창 내 확정 dd6 (i<=t-6) 기준\n")

    raw_newlow = df0["newlow"].copy()
    print(f"{'변형':10}{'평균':>9}{'σ':>9}{'왜도':>8}{'첨도':>9}   설명")
    panels = {}
    for name, fn in VARIANTS:
        s = fn(raw_newlow)
        d = df0.copy(); d["newlow"] = s
        panels[name] = d
        print(f"{name:10}{s.mean():9.3f}{s.std():9.3f}{s.skew():8.2f}{s.kurt():9.2f}")

    res, staterows = {}, []
    for name, _fn in VARIANTS:
        print(f"\n{'='*78}\n  {name}   walk-forward (K={K})\n{'='*78}")
        P, start, refits = walk_forward(panels[name], yms, n, dd6, args.decide_start, args.seed)
        pr_raw = aggregate_risk(P, n, start)              # 위험도 가중
        top_raw = P[:, K - 1].copy()                      # 최고위험 상태 확률(보조)
        pr, top = pr_raw.copy(), top_raw.copy()
        for t in range(start + 1, n):
            pr[t] = PBEAR_EMA * pr_raw[t] + (1 - PBEAR_EMA) * pr[t - 1]
            top[t] = PBEAR_EMA * top_raw[t] + (1 - PBEAR_EMA) * top[t - 1]
        reg = to_regime(pr, n, start)
        res[name] = dict(P=P, pr=pr, top=top, reg=reg, start=start, refits=refits,
                         newlow_t=panels[name]["newlow"].values)

        # 상태 성격 (전 구간 hard assignment 기준)
        hard = np.array([int(np.argmax(P[t])) if t >= start else -1 for t in range(n)])
        print(f"  {'상태':5}{'개월':>6}{'비중':>7}{'breadth':>9}{'newlowT':>9}{'trend':>8}"
              f"{'지속평균':>9}{'지속중앙':>9}{'dd6평균':>9}{'dd6중앙':>9}{'익월수익':>9}{'에피소드':>8}")
        for k in range(K):
            sel = [t for t in range(start, n) if hard[t] == k]
            if not sel:
                continue
            segs, cur = [], 0
            for t in range(start, n):
                if hard[t] == k:
                    cur += 1
                elif cur:
                    segs.append(cur); cur = 0
            if cur:
                segs.append(cur)
            b = np.mean([panels[name]["breadth"].iloc[t] for t in sel])
            nlt = np.mean([panels[name]["newlow"].iloc[t] for t in sel])
            tr = np.mean([panels[name]["trend"].iloc[t] for t in sel])
            d6 = [dd6[t] for t in sel if not np.isnan(dd6[t])]
            rt = np.nanmean([ret[t] for t in sel]) * 100
            print(f"  S{k:<4}{len(sel):>6}{len(sel)/(n-start):>6.0%}{b:>9.3f}{nlt:>9.3f}{tr:>8.3f}"
                  f"{np.mean(segs):>9.1f}{np.median(segs):>9.1f}{np.mean(d6):>8.1f}%{np.median(d6):>8.1f}%"
                  f"{rt:>8.2f}%{len(segs):>8}")
            staterows.append(dict(variant=name, state=f"S{k}", months=len(sel),
                                  breadth=b, newlow_t=nlt, trend=tr,
                                  dur_mean=np.mean(segs), dur_med=np.median(segs),
                                  dd6_mean=np.mean(d6), dd6_med=np.median(d6),
                                  next_ret=rt, episodes=len(segs)))
        eps = {k: [yms[t] for t in range(start, n) if hard[t] == k and (t == start or hard[t-1] != k)]
               for k in range(K)}
        for k in range(K):
            print(f"    S{k} 에피소드 {len(eps[k]):2d}회: {', '.join(eps[k][:8])}{' ...' if len(eps[k])>8 else ''}")

    # ── slow bear 요약 ──
    print(f"\n{'='*78}\n  ★ slow bear {SLOW_A}~{SLOW_B} (14개월) — 변형별 P_risk\n{'='*78}")
    sb_idx = [i for i, y in enumerate(yms) if SLOW_A <= y <= SLOW_B]
    base_mean = None
    sbrows = []
    print(f"  {'변형':10}{'평균':>8}{'중앙':>8}{'최대':>8}{'>0.3':>7}{'>0.5':>7}{'Risk월':>8}"
          f"{'Δ평균':>9}{'P_top평균':>10}")
    for name, _ in VARIANTS:
        pr = res[name]["pr"]; reg = res[name]["reg"]; tp_ = res[name]["top"]
        v = pr[sb_idx]
        m = float(np.nanmean(v))
        if base_mean is None:
            base_mean = m
        rm = sum(1 for i in sb_idx if reg[i] == "Risk")
        print(f"  {name:10}{m:8.3f}{np.nanmedian(v):8.3f}{np.nanmax(v):8.3f}"
              f"{int((v>0.3).sum()):>7}{int((v>0.5).sum()):>7}{rm:>8}{m-base_mean:+9.3f}"
              f"{np.nanmean(tp_[sb_idx]):>10.3f}")
        sbrows.append(dict(variant=name, mean=m, median=float(np.nanmedian(v)), max=float(np.nanmax(v)),
                           gt03=int((v > 0.3).sum()), gt05=int((v > 0.5).sum()), risk_months=rm,
                           delta_mean=m - base_mean, p_top_mean=float(np.nanmean(tp_[sb_idx]))))

    # ── 전체 탐지기 지표 ──
    print(f"\n{'='*78}\n  전체 기간 탐지기 지표 (이벤트 = 향후6M 낙폭 <= {thr:.0f}%)\n{'='*78}")
    summ = {}
    for name, _ in VARIANTS:
        r = res[name]
        summ[name] = H3.evaluate(name, r["pr"], r["reg"], yms, n, r["start"], dd6, ret, thr)

    print(f"\n  {'지표':20}" + "".join(f"{v[0]:>12}" for v in VARIANTS))
    for k, lab, fmt in [("sb", "slow bear Risk월", "{:d}"), ("lead", "평균선행(월)", "{:+.1f}"),
                        ("lift", "★리프트", "{:.2f}배"), ("disc", "분별력", "{:+.1f}%p"),
                        ("fp", "False Positive", "{:d}"), ("risk_ratio", "Risk비율", "{:.0%}"),
                        ("rec", "이벤트Recall", "{:.0%}")]:
        cells = ""
        for name, _ in VARIANTS:
            v = summ[name][k]
            cells += f"{'  -  ':>12}" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{fmt.format(v):>12}"
        print(f"  {lab:20}{cells}")

    print("\n  이벤트별 lead (양수=선행):")
    ev_all = summ["baseline"]["evs"]
    print(f"    {'이벤트':14}{'낙폭':>7}" + "".join(f"{v[0]:>12}" for v in VARIANTS))
    for ev in ev_all:
        cells = ""
        for name, _ in VARIANTS:
            cand = [o for o in summ[name]["onset"] if abs(o - ev) <= WIN]
            cells += f"{(f'{ev-min(cand,key=lambda o:abs(o-ev)):+d}' if cand else 'X'):>12}"
        print(f"    {yms[ev]:14}{dd6[ev]:>6.0f}%{cells}")

    # ── 구간 상세 ──
    for name, _ in VARIANTS:
        r = res[name]
        print(f"\n{'='*78}\n  [{name}] {DETAIL_A}~{DETAIL_B} 월별 상세\n{'='*78}")
        print(f"  {'ym':9}{'P(S0)':>7}{'P(S1)':>7}{'P(S2)':>7}{'P_risk':>8}"
              f"{'P_top':>7}{'breadth':>9}{'nl_raw':>8}{'nl_T':>8}{'trend':>8}  판정")
        for i, y in enumerate(yms):
            if not (DETAIL_A <= y <= DETAIL_B):
                continue
            print(f"  {y:9}{r['P'][i,0]:7.2f}{r['P'][i,1]:7.2f}{r['P'][i,2]:7.2f}{r['pr'][i]:8.2f}"
                  f"{r['top'][i]:7.2f}"
                  f"{df0['breadth'].iloc[i]:9.3f}{raw_newlow.iloc[i]:8.3f}{r['newlow_t'][i]:8.3f}"
                  f"{df0['trend'].iloc[i]:8.3f}  {r['reg'][i]}")

    # ── ★ feature 거리 기여 attribution ──
    print(f"\n{'='*78}\n  ★ feature별 표준화 거리 기여 (veto 완화 여부 확인)\n{'='*78}")
    for ym in DIAG_MONTHS:
        i = yms.index(ym)
        print(f"\n  ── {ym}  (실측 breadth {df0['breadth'].iloc[i]:.3f}  "
              f"newlow_raw {raw_newlow.iloc[i]:.3f}  trend {df0['trend'].iloc[i]:+.3f}) ──")
        for name, _ in VARIANTS:
            x = panels[name][EMIS].iloc[i].values
            got = contribution(res[name]["refits"], ym, x, yms)
            if got is None:
                continue
            rk, rows = got
            print(f"    [{name}]  (재학습 {rk})")
            print(f"      {'상태':5}{'|z|breadth':>12}{'|z|newlow':>11}{'|z|trend':>10}"
                  f"{'Maha':>9}   newlow 지배율")
            for rr in rows:
                per = rr["per"]; dom = per[1] ** 2 / max((per ** 2).sum(), 1e-9)
                print(f"      {rr['state']:5}{per[0]:12.2f}{per[1]:11.2f}{per[2]:10.2f}"
                      f"{rr['maha']:9.1f}{dom:>14.0%}")

    rows = []
    for name, _ in VARIANTS:
        r = res[name]
        for t in range(n):
            d = dict(variant=name, ym=yms[t], p_risk=r["pr"][t], regime=r["reg"][t],
                     ret=ret[t], dd6=dd6[t], breadth=df0["breadth"].iloc[t],
                     newlow_raw=raw_newlow.iloc[t], newlow_t=r["newlow_t"][t],
                     trend=df0["trend"].iloc[t])
            for k in range(K):
                d[f"P_S{k}"] = r["P"][t, k]
            rows.append(d)
    pd.DataFrame(rows).to_csv(OUT / "hsmm_veto_path.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(staterows).to_csv(OUT / "hsmm_veto_states.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sbrows).to_csv(OUT / "hsmm_veto_slowbear.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT/'hsmm_veto_path.csv'}\n  → {OUT/'hsmm_veto_states.csv'}\n  → {OUT/'hsmm_veto_slowbear.csv'}")
    print("\n※ 2004~2016은 생존편향(상폐 누락)으로 성과 인용 불가. 국면 구조 확인용.")


if __name__ == "__main__":
    main()
