"""
analysis/compare_overlay_cash.py

오버레이 비교표 — KOSPI200 벤치마크 추가 + **현금수익 연 2.5% 반영**. production 미수정.

■ 기존 표와 달라지는 점
  기존(fcf_hsmm_overlay.py)은 미투자분 수익을 **0%**로 가정했다.
  실제로는 현금이 이자를 벌므로 노출이 낮은 전략이 과소평가된다.
      오버레이 수익 = exp × 전략수익 + (1 − exp) × 현금수익
  현금 연 2.5% → 월 (1.025)^(1/12) − 1 = 0.2060%

■ 비교 대상
  KOSPI200(KODEX 200, 069500)   시장 벤치마크
  FCF불 단독                     전략 단독(노출 1.00)
  A: 20일 실현변동성 / B: 60일 하방변동성 / pbear만   기존 3종 (fcf_overlay_series.csv)
  production(hsmm_final 자체 노출)
  t-emission ν=4                강건 emission (5시드)

■ 사용 / 산출
  .venv/bin/python analysis/compare_overlay_cash.py [--cash 2.5]
  analysis/results/compare_overlay_cash.csv
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
RB = _load("hsmm_robust_emission", "hsmm_robust_emission.py")

BM_CODE = "069500"          # KODEX 200
SEEDS = [0, 1, 7, 42, 123]


def perf(r, rf_m):
    """CAGR / Sharpe(무위험 차감) / MDD / Calmar"""
    r = np.asarray(r, dtype=float); r = r[~np.isnan(r)]
    c = np.cumprod(1 + r); yrs = len(r) / 12
    cagr = c[-1] ** (1 / yrs) - 1
    ex = r - rf_m
    vol = r.std() * np.sqrt(12)
    mdd = float((c / np.maximum.accumulate(c) - 1).min())
    return dict(cagr=cagr, sharpe=(ex.mean() * 12) / (vol + 1e-12), mdd=mdd,
                calmar=cagr / abs(mdd) if mdd else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cash", type=float, default=2.5, help="현금 연수익률 %%")
    args = ap.parse_args()
    rf_m = (1 + args.cash / 100) ** (1 / 12) - 1

    F = pd.read_csv(A_DIR / "fcf_overlay_series.csv", encoding="utf-8-sig").set_index("ym")
    bench = F["bench"]
    idx = list(F.index)

    # KODEX 200 월수익
    conn = HF._connect()
    bm = pd.read_sql(f"SELECT trade_date::date dt, adj_close::float p FROM alpha_lab.daily_price "
                     f"WHERE stock_code='{BM_CODE}' AND adj_close>0 ORDER BY 1", conn)
    conn.close()
    bm["dt"] = pd.to_datetime(bm.dt)
    s = bm.set_index("dt")["p"].sort_index()
    mend = s.groupby(s.index.to_period("M")).last()
    mend.index = mend.index.strftime("%Y-%m")
    k200 = mend.pct_change().reindex(idx)

    # production / t-emission 노출
    df, yms, n, ret, rvol, dvol, dd6 = pd.read_pickle(A_DIR / ".cache" / "hsmm_features.pkl")
    P = pd.read_csv(A_DIR / "hsmm_final_path.csv", encoding="utf-8-sig").set_index("ym")
    # ★ 시차: bench는 '당월' 수익이므로 '전월말' 노출과 짝지어야 한다
    #   (fcf_hsmm_overlay.py:82 overlay()가 prior_ym으로 하는 것과 동일. 안 맞추면 lookahead)
    #   CSV의 expA/expB/expP는 이미 시차가 반영된 값이라 shift하지 않는다.
    exp_prod = P["exposure"].shift(1).reindex(idx)

    t_exp = {}
    for sd in SEEDS:
        pb, st, _ = RB.walk_forward(df, yms, n, "t", 4.0, sd)
        e = RB.exposure_from_p(np.nan_to_num(pb, nan=0.0), dvol, n, st)
        t_exp[sd] = pd.Series(e, index=yms).shift(1).reindex(idx)   # ★ 전월말 노출

    def ov(e):
        """오버레이 수익 = exp×전략 + (1−exp)×현금"""
        return bench * e + (1 - e) * rf_m

    print(f"기간 {idx[0]} ~ {idx[-1]} ({len(idx)}개월)   현금 연 {args.cash:.1f}% (월 {rf_m*100:.4f}%)")
    print(f"Sharpe는 무위험 {args.cash:.1f}% 차감 기준\n")
    print("=" * 92)
    print(f"  {'전략':30}{'CAGR':>9}{'Sharpe':>9}{'MDD':>10}{'Calmar':>9}{'평균exp':>9}")
    print("=" * 92)

    rows = []

    def line(nm, r, e):
        m = perf(r, rf_m)
        em = float(np.nanmean(e)) if e is not None else 1.00
        print(f"  {nm:30}{m['cagr']:8.1%}{m['sharpe']:9.2f}{m['mdd']:10.1%}{m['calmar']:9.2f}{em:9.2f}")
        rows.append(dict(strategy=nm, **m, avg_exp=em))

    line("KOSPI200 (KODEX 200)", k200, None)
    line("FCF불 단독(BM)", bench, None)
    print("  " + "-" * 88)
    line("A: 20일 실현변동성", ov(F["expA"]), F["expA"])
    line("B: 60일 하방변동성", ov(F["expB"]), F["expB"])
    line("pbear만 (vol 無)", ov(F["expP"]), F["expP"])
    line("production (hsmm_final)", ov(exp_prod), exp_prod)
    print("  " + "-" * 88)
    for sd in SEEDS:
        line(f"t-emission ν=4 (seed {sd})", ov(t_exp[sd]), t_exp[sd])
    T = pd.DataFrame([r for r in rows if r["strategy"].startswith("t-emission")])
    print(f"  {'t-emission ν=4 (5시드 평균)':30}{T.cagr.mean():8.1%}{T.sharpe.mean():9.2f}"
          f"{T.mdd.mean():10.1%}{T.calmar.mean():9.2f}{T.avg_exp.mean():9.2f}")

    # 노출 맞춤 비교
    print("\n" + "=" * 92)
    print("  노출 맞춤 비교 (production을 t-emission 평균노출로 축소 — 같은 자금 투입)")
    print("=" * 92)
    print(f"  {'전략':30}{'CAGR':>9}{'Sharpe':>9}{'MDD':>10}{'Calmar':>9}{'평균exp':>9}")
    for sd in [0, 42]:
        k = t_exp[sd].mean() / exp_prod.mean()
        line(f"production × {k:.2f}  (vs seed {sd})", ov(exp_prod * k), exp_prod * k)
        line(f"t-emission ν=4 seed {sd}", ov(t_exp[sd]), t_exp[sd])

    pd.DataFrame(rows).to_csv(OUT / "compare_overlay_cash.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT/'compare_overlay_cash.csv'}")
    print(f"\n※ 현금 {args.cash:.1f}% 가정. 0% 가정이던 기존 표보다 저노출 전략이 유리해진다.")
    print("※ production 미수정. t-emission은 실험 단계.")


if __name__ == "__main__":
    main()
