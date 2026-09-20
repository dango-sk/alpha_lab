# -*- coding: utf-8 -*-
"""analysis/hsmm_current_report.py — 현재 hsmm_final.py 최종본의 성과 리포트 (단일 버전).

비교/실험 없이 **지금 production 에 들어있는 설정 그대로** 돌린 결과만 낸다.
설정은 hsmm_final.py 의 상수를 그대로 읽으므로, 코드가 바뀌면 이 리포트도 따라 바뀐다.
★ 인과 정렬: 당월 수익 × 전월말 노출 (shift(1)). 거래비용 30bp.
사용: .venv/bin/python analysis/hsmm_current_report.py
"""
import sys, warnings, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
A = Path(__file__).parent; OUT = A / "results"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(A.parent))
_sp = importlib.util.spec_from_file_location("hsmm_final", A / "hsmm_final.py")
HF = importlib.util.module_from_spec(_sp); sys.modules["hsmm_final"] = HF; _sp.loader.exec_module(HF)
TC = 0.0030


def metrics(r):
    r = pd.Series(r).dropna()
    cq = (1 + r).cumprod(); y = len(r) / 12
    v = r.std() * np.sqrt(12); mdd = float((cq / cq.cummax() - 1).min())
    cg = cq.iloc[-1] ** (1 / y) - 1
    return dict(cum=cq.iloc[-1] - 1, cagr=cg, sharpe=(r.mean() * 12) / (v + 1e-12),
                mdd=mdd, calmar=cg / abs(mdd) if mdd else np.nan, vol=v)


def row(nm, m, ex=None):
    e = f"{ex:9.2f}" if ex is not None else " " * 9
    return (f"  {nm:24}{m['cum']:9.0%}{m['cagr']:9.1%}{m['sharpe']:9.2f}"
            f"{m['mdd']:9.1%}{m['calmar']:9.2f}{m['vol']:8.1%}{e}")


def main():
    df, yms, n, ret, rvol, dvol, dd6 = HF.build_features(use_cache=True)
    pb_raw, start = HF.walk_forward(df, yms, n)
    pbear = pb_raw.copy()
    for t in range(start + 1, n):
        pbear[t] = HF.PBEAR_EMA * pb_raw[t] + (1 - HF.PBEAR_EMA) * pbear[t - 1]

    cur = np.maximum(dvol, HF.VOL_FLOOR)
    tgt = np.full(n, HF.TARGET_VOL); tgt[start:] = np.cumsum(dvol[start:]) / np.arange(1, n - start + 1)
    raw = np.clip((1 - pbear) * (1 - pbear * (1 - np.minimum(1, tgt / cur))), HF.EXP_FLOOR, 1.0)
    exp = raw.copy(); held = None
    for t in range(start, n):
        if held is None or abs(raw[t] - held) >= HF.REBAL_BAND:
            held = round(raw[t] / 0.05) * 0.05
        exp[t] = min(max(held, HF.EXP_FLOOR), 1.0)
    P = pd.Series(pbear, index=yms); E = pd.Series(exp, index=yms)

    F = pd.read_csv(A / "fcf_overlay_series.csv", encoding="utf-8-sig").set_index("ym")
    bench = F["bench"]; El = E.shift(1).reindex(F.index)          # 전월말 노출
    r_ov = bench * El - np.abs(El.diff().fillna(0)) * TC
    D = pd.DataFrame({"bench": bench, "exp": El, "r_ov": r_ov}).dropna()

    print("=" * 104)
    print(f"  현행 HSMM 레짐 오버레이 성과  —  hsmm_final.py 최종본")
    print(f"  설정: BOUNDED_COLS={HF.BOUNDED_COLS} BOUNDED_C={HF.BOUNDED_C} / 2-state HSMM(Dmax {HF.DMAX}) /"
          f" 창 {HF.WINDOW_M}M·연1회 재적합 / 노출하한 {HF.EXP_FLOOR:.0%}")
    print(f"  기간: {D.index[0]}~{D.index[-1]} ({len(D)}개월) · 거래비용 {TC*10000:.0f}bp · 당월수익×전월말노출")
    print("=" * 104)

    print(f"\n  [1] FCF 강세전략에 오버레이 적용")
    print(f"  {'전략':24}{'누적':>9}{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'Calmar':>9}{'Vol':>8}{'평균노출':>9}")
    print(row("FCF 단독 (노출 100%)", metrics(D.bench), 1.0))
    print(row("★ HSMM 오버레이", metrics(D.r_ov), D.exp.mean()))
    e0 = D.exp.mean()
    print(row(f"[null] 상수노출 {e0:.2f}", metrics(D.bench * e0), e0))
    print("  ※ null = 타이밍 없이 비중만 같은 수준으로 낮춘 대조군. 이걸 넘어야 모델이 기여한 것.")

    m0, mv, mc = metrics(D.bench), metrics(D.r_ov), metrics(D.bench * e0)
    print(f"\n  [2] MDD 개선의 출처 분해")
    tot = mv["mdd"] - m0["mdd"]; expo = mc["mdd"] - m0["mdd"]; tim = mv["mdd"] - mc["mdd"]
    print(f"      FCF 단독 {m0['mdd']:.1%}  →  상수노출 {mc['mdd']:.1%}  →  오버레이 {mv['mdd']:.1%}")
    print(f"      총 개선 {tot:+.1%}p   ├ 익스포저 감소 {expo:+.1%}p ({expo/tot:.0%})"
          f"   └ 타이밍 기여 {tim:+.1%}p ({tim/tot:.0%})")
    print(f"      수익 쪽: 상수노출은 CAGR {mc['cagr']:.1%} 로 떨어지지만 오버레이는 {mv['cagr']:.1%} 유지"
          f" (FCF 단독 {m0['cagr']:.1%})")

    print(f"\n  [3] 위기 국면 탐지 (Recall / 리드타임)")
    yr = pd.Series([x[:4] for x in D.index], index=D.index)
    cq = (1 + D.bench).cumprod(); dd = cq / cq.cummax() - 1
    EPI = [("2018-06~2019-01", "2018 분산형 하락"), ("2020-01~2020-03", "COVID 급락"),
           ("2021-10~2022-09", "2021-22 slow bear"), ("2024-07~2024-08", "2024 조정"),
           ("2025-07~2025-08", "2025 조정"), ("2026-05~2026-07", "2026 급락")]
    print(f"  {'구간':>18}{'유형':>20}{'FCF낙폭':>9}{'최대P_bear':>11}{'첫탐지':>9}{'리드':>7}"
          f"{'구간평균노출':>12}{'오버레이수익':>12}")
    hit = 0
    for rng, lab in EPI:
        s0, s1 = rng.split("~")
        g = D.loc[s0:s1]
        if not len(g): continue
        pk = P.loc[s0:s1]
        mx = pk.max()
        first = pk[pk >= 0.5].index[0] if (pk >= 0.5).any() else None
        lead = (int(s0[:4]) * 12 + int(s0[5:])) - (int(first[:4]) * 12 + int(first[5:])) if first else None
        det = "✓" if mx >= 0.5 else "✗"
        if mx >= 0.5: hit += 1
        dmin = float(((1 + g.bench).cumprod() / (1 + g.bench).cumprod().cummax() - 1).min())
        print(f"  {rng:>18}{lab:>20}{dmin:9.1%}{mx:11.2f}{(first or '-'):>9}"
              f"{(f'{lead:+d}M' if lead is not None else '-'):>7}{g.exp.mean():12.2f}"
              f"{((1+g.r_ov).prod()-1):12.1%}   {det}")
    print(f"      → Recall {hit}/{len(EPI)}   (탐지 = 구간 내 P_bear ≥ 0.50 도달, 리드 + 는 낙폭 시작 전)")

    print(f"\n  [4] 연도별")
    print(f"  {'연도':>6}{'FCF단독':>10}{'오버레이':>10}{'차이':>9}{'평균노출':>10}")
    for y, g in D.groupby(yr):
        b = (1 + g.bench).prod() - 1; o = (1 + g.r_ov).prod() - 1
        print(f"  {y:>6}{b:10.1%}{o:10.1%}{o-b:+9.1%}{g.exp.mean():10.2f}")

    print(f"\n  [5] 현재 상태")
    last = yms[-1]
    print(f"      최종 패널월 {last}  ·  P_bear {P[last]:.3f}  →  익월 적용 노출 {E[last]:.0%}"
          f" (현금 {1-E[last]:.0%})")
    print(f"      최근 6개월 P_bear: " + "  ".join(f"{k} {P[k]:.2f}" for k in yms[-6:]))
    print(f"      최근 6개월 노출  : " + "  ".join(f"{k} {E[k]:.2f}" for k in yms[-6:]))

    D.to_csv(OUT / "hsmm_current_report.csv", encoding="utf-8-sig")
    print(f"\n  → {OUT/'hsmm_current_report.csv'}")


if __name__ == "__main__":
    main()
