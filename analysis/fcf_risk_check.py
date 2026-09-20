"""
analysis/fcf_risk_check.py   (실험 스크립트, production 미수정)

[1단계] FCF15를 제거하기 전 마지막 확인 — "수익은 안 늘었어도 위험은 줄여줬나?"

  (a) MDD / 하방변동성(Sortino) / 최악 월 / 하위 5% CVaR   — FCF15 vs BASE_noFCF
  (b) 레짐(Bull/Bear) 조건부 성과·위험
  (c) FCF 때문에 새로 들어온 종목(IN) vs 빠진 종목(OUT)의 변동성·베타
  (d) MDD 차이의 통계적 유의성 (stationary bootstrap)

백테스트를 다시 돌리지 않는다. fcf_base_compare.py가 저장한 CSV만 읽는다.
  analysis/results/fcf_base_monthly.csv   (월별 수익, tx0/tx30)
  analysis/results/fcf_base_names.csv     (IN/OUT 종목별 월수익·비중)

실행:
    .venv/bin/python analysis/fcf_risk_check.py
    .venv/bin/python analysis/fcf_risk_check.py --regime analysis/results/hsmm_veto_path.csv
산출:
    analysis/results/fcf_risk_check.csv    (요약표)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / "analysis" / "results"

RNG = np.random.default_rng(20260818)


# ══════════════════════════════════════════════════════════════
# 위험 지표
# ══════════════════════════════════════════════════════════════
def mdd(x) -> float:
    """월수익 배열의 최대낙폭 (음수)."""
    cum = np.cumprod(1 + np.asarray(x, float))
    return float((cum / np.maximum.accumulate(cum) - 1).min())


def risk_stats(x, mar: float = 0.0) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    g = float(np.prod(1 + x)) ** (12 / n) - 1
    vol = float(x.std(ddof=1)) * np.sqrt(12)
    down = x[x < mar] - mar
    dvol = float(np.sqrt((down**2).sum() / n)) * np.sqrt(12) if len(down) else 0.0
    k = max(1, int(np.ceil(n * 0.05)))
    cvar5 = float(np.sort(x)[:k].mean())
    return dict(
        n=n, cagr=g, vol=vol, sharpe=(g / vol if vol else np.nan),
        dvol=dvol, sortino=(g / dvol if dvol else np.nan),
        mdd=mdd(x), worst=float(x.min()), best=float(x.max()),
        cvar5=cvar5, down_months=int((x < 0).sum()) / n,
    )


def show(tag, s):
    print(f"  {tag:14s} CAGR {s['cagr']*100:6.2f}%  vol {s['vol']*100:5.2f}%  "
          f"하방vol {s['dvol']*100:5.2f}%  Sharpe {s['sharpe']:5.2f}  Sortino {s['sortino']:5.2f}  "
          f"MDD {s['mdd']*100:6.1f}%  최악월 {s['worst']*100:6.1f}%  CVaR5% {s['cvar5']*100:6.1f}%  "
          f"하락월 {s['down_months']*100:4.0f}%  n={s['n']}")


# ══════════════════════════════════════════════════════════════
# (d) stationary bootstrap — MDD·하방변동성 차이의 유의성
# ══════════════════════════════════════════════════════════════
def stationary_bootstrap_idx(n, mean_block=6, size=None):
    """Politis-Romano stationary bootstrap 인덱스 (원형)."""
    size = size or n
    p = 1.0 / mean_block
    idx = np.empty(size, dtype=int)
    i = RNG.integers(n)
    for t in range(size):
        idx[t] = i
        i = RNG.integers(n) if RNG.random() < p else (i + 1) % n
    return idx


def boot_diff(a, b, stat, n_boot=2000, mean_block=6):
    """같은 시점 인덱스로 두 시계열을 동시 리샘플 → stat(FCF)-stat(BASE) 분포."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = stat(a) - stat(b)
    d = np.empty(n_boot)
    for k in range(n_boot):
        i = stationary_bootstrap_idx(len(a), mean_block)
        d[k] = stat(a[i]) - stat(b[i])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_improve = float((d > 0).mean())      # FCF가 더 나은(=지표가 큰) 비율
    return obs, lo, hi, p_improve


# ══════════════════════════════════════════════════════════════
# (c) IN / OUT 바스켓
# ══════════════════════════════════════════════════════════════
def basket_series(names: pd.DataFrame, side: str, weighted: bool) -> pd.Series:
    d = names[(names["side"] == side) & names["ret"].notna()]
    if weighted:
        g = d.groupby("ym").apply(
            lambda x: np.average(x["ret"], weights=x["weight"]) if x["weight"].sum() > 0 else np.nan)
    else:
        g = d.groupby("ym")["ret"].mean()
    return g.sort_index()


def beta_alpha(y: pd.Series, mkt: pd.Series):
    df = pd.concat([y.rename("y"), mkt.rename("m")], axis=1).dropna()
    if len(df) < 12:
        return np.nan, np.nan, len(df)
    b = np.cov(df["y"], df["m"], ddof=1)[0, 1] / np.var(df["m"], ddof=1)
    a = df["y"].mean() - b * df["m"].mean()
    return float(b), float(a) * 12, len(df)


# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly", default=str(OUT / "fcf_base_monthly.csv"))
    ap.add_argument("--names", default=str(OUT / "fcf_base_names.csv"))
    ap.add_argument("--regime", default=str(OUT / "hsmm_longrun_path.csv"),
                    help="ym,regime(Bull/Bear)[,ret] 컬럼을 가진 레짐 경로 CSV")
    ap.add_argument("--tx", type=int, default=30, choices=[0, 30])
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_csv(args.monthly, encoding="utf-8-sig")
    f = df[f"fcf_tx{args.tx}"].values
    b = df[f"base_tx{args.tx}"].values
    months = df["ym"].astype(str).values
    rows = []

    print("═" * 100)
    print(f"  (a) 전체 구간 위험 지표  {months[0]}~{months[-1]}  (tx {args.tx}bp)")
    print("═" * 100)
    sf, sb = risk_stats(f), risk_stats(b)
    show("FCF15", sf); show("BASE_noFCF", sb)
    print(f"  → 차이(FCF-BASE): MDD {(sf['mdd']-sb['mdd'])*100:+.1f}%p  "
          f"하방vol {(sf['dvol']-sb['dvol'])*100:+.2f}%p  "
          f"최악월 {(sf['worst']-sb['worst'])*100:+.1f}%p  CVaR5% {(sf['cvar5']-sb['cvar5'])*100:+.1f}%p")
    rows += [dict(scope="ALL", strat="FCF15", **sf), dict(scope="ALL", strat="BASE", **sb)]

    # ── (d) 유의성 ──
    print("\n" + "═" * 100)
    print(f"  (d) 위험 개선의 유의성 — stationary bootstrap (mean block 6M, B={args.boot})")
    print("═" * 100)
    for label, stat, better in [
        ("MDD (클수록 얕음)", mdd, "FCF의 MDD가 더 얕을"),
        ("하방변동성(부호반전)", lambda x: -risk_stats(x)["dvol"], "FCF의 하방vol이 더 낮을"),
        ("최악 월", lambda x: float(np.min(x)), "FCF의 최악월이 덜 나쁠"),
    ]:
        obs, lo, hi, p = boot_diff(f, b, stat, n_boot=args.boot)
        print(f"  {label:22s} 관측차 {obs*100:+6.2f}%p   95%CI [{lo*100:+.2f}, {hi*100:+.2f}]   "
              f"P({better} 확률) = {p:.2f}")

    # ── (b) 레짐 조건부 ──
    reg_path = Path(args.regime)
    if reg_path.exists():
        rg = pd.read_csv(reg_path, encoding="utf-8-sig")
        rg["ym"] = rg["ym"].astype(str)
        rmap = dict(zip(rg["ym"], rg["regime"]))
        mkt = pd.Series(rg.set_index("ym")["ret"]) if "ret" in rg.columns else None
        lab = np.array([rmap.get(m, "NA") for m in months])
        print("\n" + "═" * 100)
        print(f"  (b) 레짐 조건부 ({reg_path.name}; 매칭 {int((lab!='NA').sum())}/{len(lab)}개월)")
        print("═" * 100)
        for r in ["Bull", "Bear"]:
            m = lab == r
            if m.sum() < 6:
                print(f"  {r}: 표본 {int(m.sum())}개월 — 생략")
                continue
            srf, srb = risk_stats(f[m]), risk_stats(b[m])
            print(f"  ── {r} ({int(m.sum())}개월) ──")
            show("FCF15", srf); show("BASE_noFCF", srb)
            print(f"     증분 월평균 {(f[m]-b[m]).mean()*100:+.3f}%p, "
                  f"MDD차 {(srf['mdd']-srb['mdd'])*100:+.1f}%p")
            rows += [dict(scope=r, strat="FCF15", **srf), dict(scope=r, strat="BASE", **srb)]
    else:
        mkt = None
        print(f"\n  [경고] 레짐 파일 없음: {reg_path} — (b) 생략")

    # ── (c) IN / OUT ──
    npath = Path(args.names)
    if npath.exists():
        names = pd.read_csv(npath, encoding="utf-8-sig", dtype={"code": str})
        print("\n" + "═" * 100)
        print("  (c) FCF 때문에 진입(IN) / 제외(OUT)된 종목 바스켓의 위험")
        print("═" * 100)
        for wtd, tag in [(False, "동일가중"), (True, "비중가중")]:
            bi, bo = basket_series(names, "IN", wtd), basket_series(names, "OUT", wtd)
            print(f"  ── {tag} ──")
            for nm, s in [("IN", bi), ("OUT", bo)]:
                st = risk_stats(s.dropna().values)
                line = (f"  {nm:14s} vol {st['vol']*100:5.2f}%  하방vol {st['dvol']*100:5.2f}%  "
                        f"MDD {st['mdd']*100:6.1f}%  최악월 {st['worst']*100:6.1f}%  월평균 "
                        f"{s.mean()*100:+.2f}%  n={st['n']}")
                if mkt is not None:
                    be, al, nn = beta_alpha(s, mkt)
                    line += f"  β={be:5.2f}  α(연) {al*100:+5.1f}%"
                print(line)
            spread = (bi - bo).dropna()
            print(f"  {'IN-OUT':14s} 월평균 {spread.mean()*100:+.2f}%  vol {spread.std(ddof=1)*np.sqrt(12)*100:5.2f}%"
                  f"  하방기여차 {(risk_stats(bi.dropna().values)['dvol']-risk_stats(bo.dropna().values)['dvol'])*100:+.2f}%p")
    else:
        print(f"\n  [경고] names 파일 없음: {npath} — (c) 생략")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "fcf_risk_check.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT}/fcf_risk_check.csv")
    print("\n판단 기준: MDD·하방vol·최악월 어느 것도 유의하게(P≥0.9) 개선되지 않으면 FCF15 제거.")


if __name__ == "__main__":
    main()
