# -*- coding: utf-8 -*-
"""analysis/hsmm_v5_periods.py — V5(c=2) vs production 을 **모든 기간에서** 대조.

전체 요약 하나로 채택을 판단하면 특정 1~2년이 만든 우위를 전 구간 우위로 오독한다
(DIET-4 에서 개선분 96.6%가 3종목이었던 것과 같은 함정). 그래서 아래를 전부 본다.
  1) 연도별  2) 3분할 소구간  3) 롤링 12개월 초과수익 분포  4) 낙폭 에피소드별
  5) 월별 승률·초과수익 t검정  6) 상승장/하락장 조건부  7) 최악 월 5개

★ 인과 정렬: 당월 수익 × **전월말** 노출 (shift(1)). 빠뜨리면 lookahead.
사용: .venv/bin/python analysis/hsmm_v5_periods.py [--c 2.0]
"""
import sys, argparse, warnings, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
A_DIR = Path(__file__).parent; OUT = A_DIR / "results"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(A_DIR.parent))
_sp = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_sp); sys.modules["hsmm_final"] = HF; _sp.loader.exec_module(HF)
_s2 = importlib.util.spec_from_file_location("v5", A_DIR / "hsmm_v5_bounded.py")
V5 = importlib.util.module_from_spec(_s2); sys.modules["v5"] = V5; _s2.loader.exec_module(V5)
TC = V5.TC


def series(df, yms, n, ret, dvol, F, c, axes, seed):
    pb, st, _ = V5.walk_forward(df, yms, n, ret, c, axes, seed, f"c={c}")
    e = V5.exposure_from_p(np.nan_to_num(pb, nan=0.0), dvol, n, st)
    E = pd.Series(e, index=yms).shift(1).reindex(F.index)      # ★ 전월말 노출
    r = F["bench"] * E - np.abs(E.diff().fillna(0)) * TC
    return pd.Series(pb, index=yms), E, r


def m(r):
    r = r.dropna()
    if len(r) < 6: return dict(cagr=np.nan, sharpe=np.nan, mdd=np.nan, calmar=np.nan)
    cq = (1 + r).cumprod(); y = len(r) / 12
    v = r.std() * np.sqrt(12); mdd = float((cq / cq.cummax() - 1).min())
    cg = cq.iloc[-1] ** (1 / y) - 1
    return dict(cagr=cg, sharpe=(r.mean() * 12) / (v + 1e-12), mdd=mdd,
                calmar=cg / abs(mdd) if mdd else np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--c", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
    axes = [HF.EMIS_COLS.index("newlow")]
    df, yms, n, ret, rvol, dvol, dd6 = pd.read_pickle(A_DIR / ".cache" / "hsmm_features.pkl")
    F = pd.read_csv(A_DIR / "fcf_overlay_series.csv", encoding="utf-8-sig").set_index("ym")
    bench = F["bench"]

    pbP, EP, rP = series(df, yms, n, ret, dvol, F, np.inf, axes, a.seed)
    pbV, EV, rV = series(df, yms, n, ret, dvol, F, a.c, axes, a.seed)
    D = pd.DataFrame({"bench": bench, "exp_prod": EP, "exp_v5": EV,
                      "r_prod": rP, "r_v5": rV}).dropna()
    D["diff"] = D.r_v5 - D.r_prod
    yr = pd.Series([x[:4] for x in D.index], index=D.index)

    print("=" * 100)
    print(f"  V5 c={a.c:g} vs production — 전 기간 대조  (당월수익 × 전월말 노출, 30bp)")
    print("=" * 100)
    print(f"\n  [1] 연도별\n  {'연도':>6}{'FCF원본':>10}{'prod':>9}{'V5':>9}{'차이':>9}"
          f"{'노출prod':>10}{'노출V5':>9}{'승자':>7}")
    winsY = []
    for y, g in D.groupby(yr):
        b = (1 + g.bench).prod() - 1; p = (1 + g.r_prod).prod() - 1; v = (1 + g.r_v5).prod() - 1
        w = "V5" if v > p else ("prod" if p > v else "=")
        winsY.append(w)
        print(f"  {y:>6}{b:10.1%}{p:9.1%}{v:9.1%}{v-p:+9.1%}"
              f"{g.exp_prod.mean():10.2f}{g.exp_v5.mean():9.2f}{w:>7}")
    print(f"  → 연도 승패: V5 {winsY.count('V5')}승 / prod {winsY.count('prod')}승 ({len(winsY)}년)")

    print(f"\n  [2] 소구간 3분할\n  {'구간':>14}{'전략':>10}{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}"
          f"{'Calmar':>9}{'평균노출':>9}")
    segs = {"2018-2020": ("2018", "2020"), "2021-2023": ("2021", "2023"), "2024-2026": ("2024", "2026")}
    for nm, (s0, s1) in segs.items():
        g = D[(yr >= s0) & (yr <= s1)]
        for lab, col, ex in [("FCF 원본", "bench", None), ("production", "r_prod", "exp_prod"),
                             (f"V5 c={a.c:g}", "r_v5", "exp_v5")]:
            mm = m(g[col])
            e = g[ex].mean() if ex else 1.0
            print(f"  {nm if lab=='FCF 원본' else '':>14}{lab:>10}{mm['cagr']:9.1%}{mm['sharpe']:9.2f}"
                  f"{mm['mdd']:9.1%}{mm['calmar']:9.2f}{e:9.2f}")

    print(f"\n  [3] 롤링 12개월 초과수익 (V5 − prod)")
    roll = (1 + D.r_v5).rolling(12).apply(np.prod, raw=True) - (1 + D.r_prod).rolling(12).apply(np.prod, raw=True)
    rr = roll.dropna()
    print(f"      창 {len(rr)}개 중 V5 우세 {int((rr > 0).sum())}개 ({(rr > 0).mean():.0%})"
          f"   중앙값 {rr.median():+.1%}   최악 {rr.min():+.1%} ({rr.idxmin()})   최고 {rr.max():+.1%} ({rr.idxmax()})")
    worst = rr.nsmallest(5)
    print("      V5 가 가장 뒤진 12M 창:  " + "  ".join(f"{i}:{v:+.1%}" for i, v in worst.items()))

    print(f"\n  [4] 낙폭 에피소드 (FCF 원본 기준 −10% 이상 구간)")
    cq = (1 + D.bench).cumprod(); dd = cq / cq.cummax() - 1
    inep = dd < -0.05; eps = []; st0 = None
    for i, (k, v) in enumerate(inep.items()):
        if v and st0 is None: st0 = k
        if (not v or i == len(inep) - 1) and st0 is not None:
            seg = dd.loc[st0:k]
            if seg.min() <= -0.10: eps.append((st0, k, float(seg.min())))
            st0 = None
    print(f"  {'구간':>18}{'FCF낙폭':>9}{'prod':>9}{'V5':>9}{'차이':>9}{'노출prod':>10}{'노출V5':>9}")
    for s0, s1, mn in eps:
        g = D.loc[s0:s1]
        p = (1 + g.r_prod).prod() - 1; v = (1 + g.r_v5).prod() - 1
        print(f"  {s0+'~'+s1:>18}{mn:9.1%}{p:9.1%}{v:9.1%}{v-p:+9.1%}"
              f"{g.exp_prod.mean():10.2f}{g.exp_v5.mean():9.2f}")

    print(f"\n  [5] 월별 승률·유의성")
    d = D["diff"].dropna()
    t, pv = stats.ttest_1samp(d, 0)
    print(f"      월 승률 {(d > 0).mean():.0%} ({int((d>0).sum())}/{len(d)})   평균 초과 {d.mean()*12:+.2%}/년"
          f"   t={t:+.2f}  p={pv:.3f}")
    print(f"      ※ p 는 참고치. 같은 데이터로 c 를 고른 뒤의 검정이라 과대평가된다.")

    print(f"\n  [6] 시장국면 조건부 (FCF 원본 월수익 기준)")
    print(f"  {'국면':>12}{'월수':>6}{'FCF원본':>10}{'prod':>9}{'V5':>9}{'차이':>9}{'노출prod':>10}{'노출V5':>9}")
    for nm, msk in [("하락 <−5%", D.bench < -0.05), ("약보합 −5~0%", (D.bench >= -0.05) & (D.bench < 0)),
                    ("강보합 0~5%", (D.bench >= 0) & (D.bench < 0.05)), ("상승 >+5%", D.bench >= 0.05)]:
        g = D[msk]
        if not len(g): continue
        print(f"  {nm:>12}{len(g):6}{g.bench.mean():10.2%}{g.r_prod.mean():9.2%}{g.r_v5.mean():9.2%}"
              f"{g.r_v5.mean()-g.r_prod.mean():+9.2%}{g.exp_prod.mean():10.2f}{g.exp_v5.mean():9.2f}")

    print(f"\n  [7] FCF 원본 최악 월 6개 — 방어했는가")
    print(f"  {'월':>9}{'FCF원본':>10}{'prod':>9}{'V5':>9}{'노출prod':>10}{'노출V5':>9}")
    for k, v in D.bench.nsmallest(6).items():
        r = D.loc[k]
        print(f"  {k:>9}{v:10.1%}{r.r_prod:9.1%}{r.r_v5:9.1%}{r.exp_prod:10.2f}{r.exp_v5:9.2f}")

    D.to_csv(OUT / "hsmm_v5_periods.csv", encoding="utf-8-sig")
    print(f"\n  → {OUT/'hsmm_v5_periods.csv'}")


if __name__ == "__main__":
    main()
