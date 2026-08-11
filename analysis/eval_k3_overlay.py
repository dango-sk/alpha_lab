"""
analysis/eval_k3_overlay.py

K=3 + 인과적 dd6 상태정렬이 **실전에서 쓸 만한가**를 검증. production 미수정.

■ 검증 대상 (옵션 B)
  · K=3 HSMM (정상 covariance)
  · 상태 위험도 = 학습창 내 확정 dd6(i<=t-6) 평균으로 정렬 — lookahead 없음
  · P_risk = Σ P_k·w_k,  w=[0, 0.5, 1]   (K=2면 production P_bear과 동일 정의)
  · 노출 = **production과 동일한 공식** (소프트 비대칭 vol-target, 하한 0.20, 리밸밴드 0.15, 0.05스텝)
  · 오버레이 대상 = FCF불 전략 실제 월수익 (analysis/fcf_overlay_series.csv의 bench)

■ 왜 이 검증이 필요한가
  앞선 -19.3% → -10.0% 수치는 **장기패널 + (1-p) 단순환산**이었다. 실전 판단에는 부족하다.
    ① production 패널(2017~)에서도 추정되는가 (13개월 첫 창에서 K=3이 붕괴할 수 있다)
    ② 2022뿐 아니라 **전 기간** CAGR/Sharpe/MDD가 개선되는가
    ③ Risk비율 45%가 만드는 상시 저노출의 **비용**은 얼마인가
    ④ **시드 안정성** — [[project_regime_macro_features]]에 "분별력 +1.75p는 전부 seed42 운빨"
       이었던 이력이 있다. 5개 시드로 확인한다.

■ 판정 기준
  · 5개 시드 전부에서 MDD 개선 + Sharpe 비악화 → 쓸 만함
  · 시드별 부호가 갈리면 → 운빨. 채택 불가
  · 2022만 좋고 전 기간이 나빠지면 → 채택 불가

■ 사용 / 산출
  .venv/bin/python analysis/eval_k3_overlay.py
  analysis/results/eval_k3_overlay.csv
"""
import sys
import argparse
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

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


def _load(mod, path):
    sp = importlib.util.spec_from_file_location(mod, A_DIR / path)
    m = importlib.util.module_from_spec(sp); sys.modules[mod] = m; sp.loader.exec_module(m)
    return m


HF = _load("hsmm_final", "hsmm_final.py")
H3 = _load("hsmm_3state", "hsmm_3state.py")
VETO = _load("hsmm_newlow_veto", "hsmm_newlow_veto.py")   # walk_forward(인과 정렬) 재사용

SEEDS = [0, 1, 7, 42, 123]
SLOW_A, SLOW_B = "2021-11", "2022-12"


def exposure_from_p(p, dvol, n, start):
    """production(hsmm_final.main)과 동일한 노출 산출."""
    cur = np.maximum(dvol, HF.VOL_FLOOR)
    tgt = np.full(n, HF.TARGET_VOL, dtype=float)
    tgt[start:] = np.cumsum(dvol[start:]) / np.arange(1, n - start + 1)
    cut = 1.0 - np.minimum(1.0, tgt / cur)
    raw = np.clip((1 - p) * (1.0 - p * cut), HF.EXP_FLOOR, 1.0)
    exp = raw.copy(); held = None
    for t in range(start, n):
        if held is None or abs(raw[t] - held) >= HF.REBAL_BAND:
            held = round(raw[t] / 0.05) * 0.05
        exp[t] = min(max(held, HF.EXP_FLOOR), 1.0)
    return exp


def perf(r):
    r = np.asarray(r, dtype=float); r = r[~np.isnan(r)]
    if len(r) < 6:
        return dict(cagr=np.nan, sharpe=np.nan, mdd=np.nan, vol=np.nan)
    c = np.cumprod(1 + r)
    yrs = len(r) / 12
    cagr = c[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    return dict(cagr=cagr, sharpe=(r.mean() * 12) / (vol + 1e-12), vol=vol,
                mdd=float((c / np.maximum.accumulate(c) - 1).min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true", help="장기패널로도 확인")
    args = ap.parse_args()

    P_SHORT = A_DIR / ".cache" / "hsmm_features.pkl"
    df, yms, n, ret, rvol, dvol, dd6 = pd.read_pickle(P_SHORT)
    ds = HF.DECIDE_START

    # FCF 전략 실제 월수익 (production 오버레이 대상)
    F = pd.read_csv(A_DIR / "fcf_overlay_series.csv", encoding="utf-8-sig").set_index("ym")
    bench = F["bench"].reindex(yms)
    prod_exp = F["expB"].reindex(yms)          # production 오버레이(60일 하방) 노출
    print(f"패널 {yms[0]}~{yms[-1]}  판정 {ds}   오버레이 대상 = FCF불 {F.index[0]}~{F.index[-1]} ({len(F)}개월)")

    # ── 기준: production 2-state ──
    prod_p = pd.read_csv(A_DIR / "hsmm_final_path.csv", encoding="utf-8-sig").set_index("ym")["pbear"].reindex(yms)
    rows = []
    m_bm = perf(bench)
    m_pr = perf(bench * prod_exp)
    print(f"\n{'='*92}\n  기준선\n{'='*92}")
    print(f"  {'전략':22}{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'평균노출':>9}{'2022 손실':>11}")
    sb = [y for y in F.index if SLOW_A <= y <= SLOW_B]

    def sb_loss(r):
        x = r.reindex(sb).dropna()
        return (np.prod(1 + x) - 1) * 100 if len(x) else np.nan

    print(f"  {'FCF불 단독':22}{m_bm['cagr']:8.1%}{m_bm['sharpe']:9.2f}{m_bm['mdd']:9.1%}"
          f"{1.0:9.2f}{sb_loss(bench):10.1f}%")
    print(f"  {'production 오버레이':22}{m_pr['cagr']:8.1%}{m_pr['sharpe']:9.2f}{m_pr['mdd']:9.1%}"
          f"{prod_exp.mean():9.2f}{sb_loss(bench*prod_exp):10.1f}%")
    rows += [dict(model="FCF불 단독", seed=None, **m_bm, exp=1.0, sb=sb_loss(bench)),
             dict(model="production 2-state", seed=None, **m_pr, exp=float(prod_exp.mean()),
                  sb=sb_loss(bench * prod_exp))]

    # ── K=3 + 인과 정렬, 5 시드 ──
    print(f"\n{'='*92}\n  K=3 + 인과 dd6 정렬  (시드 {SEEDS})\n{'='*92}")
    print(f"  {'시드':>5}{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'평균노출':>9}{'2022 손실':>11}"
          f"{'Risk비율':>9}{'상태붕괴':>9}")
    keep = {}
    for sd in SEEDS:
        try:
            P, start, _rf = VETO.walk_forward(df, yms, n, dd6, ds, sd)
        except Exception as e:
            print(f"  {sd:>5}   실패: {type(e).__name__} {str(e)[:40]}")
            continue
        pr_raw = VETO.aggregate_risk(P, n, start)
        pr = pr_raw.copy()
        for t in range(start + 1, n):
            pr[t] = HF.PBEAR_EMA * pr_raw[t] + (1 - HF.PBEAR_EMA) * pr[t - 1]
        collapse = int(np.sum(np.nanstd(P[start:], axis=1) < 1e-6))   # 상태 구분 사라진 달
        e = exposure_from_p(np.nan_to_num(pr, nan=0.0), dvol, n, start)
        E = pd.Series(e, index=yms).reindex(F.index)
        r = bench * E
        m = perf(r)
        rr = float(np.nanmean(pr[start:] >= 0.5))
        print(f"  {sd:>5}{m['cagr']:8.1%}{m['sharpe']:9.2f}{m['mdd']:9.1%}{E.mean():9.2f}"
              f"{sb_loss(r):10.1f}%{rr:9.0%}{collapse:9d}")
        rows.append(dict(model="K3 인과정렬", seed=sd, **m, exp=float(E.mean()), sb=sb_loss(r),
                         risk_ratio=rr, collapse=collapse))
        keep[sd] = (pr, E, r)

    K = pd.DataFrame([r for r in rows if r["model"] == "K3 인과정렬"])
    if len(K):
        print(f"\n  {'집계':>5}{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}")
        print(f"  {'평균':>5}{K.cagr.mean():8.1%}{K.sharpe.mean():9.2f}{K.mdd.mean():9.1%}")
        print(f"  {'σ':>5}{K.cagr.std():8.1%}{K.sharpe.std():9.2f}{K.mdd.std():9.1%}")
        print(f"  {'최악':>5}{K.cagr.min():8.1%}{K.sharpe.min():9.2f}{K.mdd.min():9.1%}")

        print(f"\n{'='*92}\n  판정\n{'='*92}")
        better_mdd = (K.mdd > m_pr["mdd"]).sum()      # mdd는 음수 → 큰 값이 개선
        better_shp = (K.sharpe >= m_pr["sharpe"]).sum()
        better_sb = (K.sb > sb_loss(bench * prod_exp)).sum()
        print(f"  production 대비  MDD 개선 {better_mdd}/{len(K)} 시드"
              f"   Sharpe 비악화 {better_shp}/{len(K)}   2022 손실 축소 {better_sb}/{len(K)}")
        if K.collapse.max() > 0:
            print(f"  ★ 상태 붕괴 발생 시드 있음(최대 {K.collapse.max()}개월) — 짧은 패널에서 K=3 추정 불안정")
        if better_mdd == len(K) and better_shp >= len(K) - 1:
            print("  ▶ 쓸 만함 — 전 시드에서 MDD 개선, Sharpe 비악화")
        elif better_mdd >= len(K) * 0.6:
            print("  ▶ 조건부 — 다수 시드에서 개선되나 전부는 아님. 시드 다수결 운용 필요")
        else:
            print("  ▶ 쓸 만하지 않음 — 시드에 따라 부호가 갈림(운빨)")

    pd.DataFrame(rows).to_csv(OUT / "eval_k3_overlay.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT/'eval_k3_overlay.csv'}")
    print("\n※ production 미수정. 노출 공식·오버레이 대상은 production과 동일하게 맞춤.")


if __name__ == "__main__":
    main()
