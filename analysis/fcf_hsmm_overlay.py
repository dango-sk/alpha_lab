"""
analysis/fcf_hsmm_overlay.py  (실험 스크립트, production 미수정)

HSMM 레짐의 '연속 익스포저'를 FCF불 전략 월수익에 곱해서(overlay) 비교.
  benchmark = FCF불 전략 단독 (FCF_YIELD추가전략, 익스포저=1)
  Overlay A = cur_vol 분모 = 20일 실현변동성(rvol)   ← 박사님 스펙
  Overlay B = cur_vol 분모 = 60일 하방변동성(dvol)   ← 현재 hsmm_final 코드
  Overlay P = pbear만 (vol-타게팅 미적용, 참고용)

목표변동률: 각 버전 '그 측정치의 인과적 확장평균'(역사적 평균 변동률) → 분자·분모 스케일 자동 일치.
타이밍(인과): FCF의 M월 수익 = M-1월말에 확정된 익스포저 × 그 달 수익.
사용: DATABASE_URL 세팅 후  .venv/bin/python analysis/fcf_hsmm_overlay.py
"""
import os, sys
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
import numpy as np
import hsmm_final as H
from lib.data import load_strategy

BULL = "FCF_YIELD추가전략"


def prior_ym(ym):
    y, m = int(ym[:4]), int(ym[5:7]); m -= 1
    if m == 0: y -= 1; m = 12
    return f"{y:04d}-{m:02d}"


def main():
    # ── 1. FCF불 전략 단독(월수익/리밸일) ──
    bd = load_strategy(BULL, rebal_type="monthly", universe="KOSPI")
    r = bd["results"]
    fret, frd = r["monthly_returns"], r["rebalance_dates"]
    fmonths = [frd[i][:7] for i in range(len(fret))]        # M월 수익의 '그 달' ym
    fcf = {fmonths[i]: fret[i] for i in range(len(fret))}
    CUTOFF = "2026-07"                                       # 오늘 7/24: 7월 rebal까지만, 8월 예정 rebal 무시
    fmonths = [ym for ym in fmonths if ym <= CUTOFF]
    fcf = {ym: fcf[ym] for ym in fmonths}
    print(f"[FCF불 단독] {fmonths[0]}~{fmonths[-1]} {len(fmonths)}개월  (컷오프 {CUTOFF}, 8월 예정 rebal 제외)  "
          f"CAGR {r['cagr']*100:.1f}% MDD {r['mdd']*100:.1f}% Sharpe {r['sharpe']:.2f}", flush=True)

    # ── 2. HSMM 레짐 엔진 ──
    df, yms, n, _ret, rvol, dvol, _dd6 = H.build_features()
    pbear_raw, start = H.walk_forward(df, yms, n)
    idx = list(range(start, n))
    pbear = pbear_raw.copy()
    for t in idx[1:]:
        pbear[t] = H.PBEAR_EMA * pbear_raw[t] + (1 - H.PBEAR_EMA) * pbear[t - 1]

    # ── 3. 익스포저 경로 (분모 스위치) ──
    def build_exposure(volmeas):
        """volmeas=None → vol-타게팅 미적용(pbear만). 아니면 그 측정치로 소프트 비대칭 vol-타겟."""
        if volmeas is None:
            cut = np.zeros(n); tgt = np.full(n, np.nan)
        else:
            cur = np.maximum(volmeas, H.VOL_FLOOR)
            tgt = np.full(n, np.nan); acc = []
            for t in idx:                                   # 인과적 확장평균 = 역사적 평균 변동률
                acc.append(cur[t]); tgt[t] = float(np.mean(acc))
            cut = 1.0 - np.minimum(1.0, np.nan_to_num(tgt, nan=1.0) / cur)
        vf = 1.0 - pbear * cut
        raw = np.clip((1 - pbear) * vf, H.EXP_FLOOR, 1.0)
        exp = raw.copy(); held = None                       # Z단계 리밸밴드 (hsmm_final과 동일)
        for t in idx:
            if held is None or abs(raw[t] - held) >= H.REBAL_BAND:
                held = round(raw[t] / 0.05) * 0.05
            exp[t] = min(max(held, H.EXP_FLOOR), 1.0)
        return exp, cut

    expA, cutA = build_exposure(rvol)
    expB, cutB = build_exposure(dvol)
    expP, _ = build_exposure(None)

    def edict(exp): return {yms[t]: exp[t] for t in idx}
    eA, eB, eP = edict(expA), edict(expB), edict(expP)

    # ── 4. overlay 수익 (인과: 전월말 익스포저 × 당월 FCF수익) ──
    def overlay(edic):
        return np.array([edic.get(prior_ym(ym), 1.0) * fcf[ym] for ym in fmonths])
    bench = np.array([fcf[ym] for ym in fmonths])
    ovA, ovB, ovP = overlay(eA), overlay(eB), overlay(eP)

    import pandas as pd                                       # 유의성 검정용 시계열 저장
    pd.DataFrame({"ym": fmonths, "bench": bench, "ovA": ovA, "ovB": ovB, "ovP": ovP,
                  "expA": [eA.get(prior_ym(y), 1.0) for y in fmonths],
                  "expB": [eB.get(prior_ym(y), 1.0) for y in fmonths],
                  "expP": [eP.get(prior_ym(y), 1.0) for y in fmonths],
                  }).to_csv(REPO / "analysis" / "fcf_overlay_series.csv", index=False, encoding="utf-8-sig")

    def avg_exp(edic):
        return float(np.mean([edic.get(prior_ym(ym), 1.0) for ym in fmonths]))

    # ── 5. 리포트 ──
    print(f"\n{'='*82}\n  FCF불 전략에 HSMM 연속 익스포저 오버레이 ({fmonths[0]}~{fmonths[-1]}, {len(fret)}개월)\n{'='*82}")
    print(f"  {'전략':22} {'CAGR':>7} {'Sharpe':>7} {'MDD':>8} {'Calmar':>7} {'Vol':>7}  {'평균exp':>7}")
    for nm, rr, ed in [("FCF불 단독(BM)", bench, None),
                       ("A: 20일 실현변동성", ovA, eA),
                       ("B: 60일 하방변동성", ovB, eB),
                       ("(참고) pbear만", ovP, eP)]:
        ae = f"{avg_exp(ed):>7.2f}" if ed else f"{'1.00':>7}"
        print("  " + H.line(nm, H.perf(rr)).strip().ljust(0) + f"  {ae}")

    labs = fmonths
    for nm, rr in [("FCF불", bench), ("A", ovA), ("B", ovB)]:
        mw = H.mdd_window(rr, labs)
        if mw: print(f"  MDD구간[{nm}] {mw[0]} → {mw[1]}  ({mw[2]:.1f}%)")

    cA = int(sum(1 for t in idx if cutA[t] > 1e-6)); cB = int(sum(1 for t in idx if cutB[t] > 1e-6))
    print(f"\n  vol-타게팅 실제 작동(cut>0) 개월수:  A(20일실현) {cA}/{len(idx)}   B(60일하방) {cB}/{len(idx)}")
    print(f"  * 목표변동률 = 각 측정치의 인과적 확장평균(역사적 평균). A/B는 분모만 다름.")
    print(f"  * 타이밍: 당월 FCF수익 × 전월말 확정 익스포저 (lookahead 없음).")


if __name__ == "__main__":
    main()
