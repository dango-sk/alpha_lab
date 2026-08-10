"""analysis/factor_kospi_regime_quick.py

빠른 두 가지 분석:

1) 월별 KOSPI 수익률 vs 월별 cross-sectional F_EPS_M 평균값 상관계수
2) Bull/Bear 구간별 cross-sectional F_EPS_M 분산 vs ATT_PER 분산 비교

데이터:
- 리밸 날짜: alpha_lab.universe (monthly)
- 팩터: lib.factor_engine.load_factor_data → run_regressions (fper_epsg만)
- KOSPI: 069500 (KODEX 200) adj_close
- 레짐: KOSPI 200 vs MA(200), 그리고 _CYCLE_BEAR_PERIODS 두 가지 모두 보여줌

실행:
    python analysis/factor_kospi_regime_quick.py
    python analysis/factor_kospi_regime_quick.py --large-only
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
import pandas as pd

from lib.db import get_conn
from lib.factor_engine import (
    prefetch_all_data,
    load_factor_data,
    run_regressions,
    clear_prefetch_cache,
    clear_factor_cache,
)
from lib.data import _CYCLE_BEAR_PERIODS, _get_regime_by_cycle


REGRESSION_MODELS = [
    ("fper_epsg", "f_epsg", "f_per", "ratio"),
]
OUTLIER_FILTERS = {
    "fper_epsg": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60},
}


def get_rebal_dates(conn, start="2018-01-01") -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT rebal_date FROM universe "
        "WHERE rebal_type='monthly' AND rebal_date >= ? "
        "ORDER BY rebal_date",
        (start,),
    ).fetchall()
    return [r[0] for r in rows]


def get_kospi_series(conn, dates: list[str]) -> dict[str, float]:
    """각 리밸 날짜의 069500 adj_close (해당 날짜 이전 가장 가까운 거래일)."""
    out = {}
    for d in dates:
        row = conn.execute(
            "SELECT adj_close FROM daily_price "
            "WHERE stock_code='069500' AND trade_date<=? "
            "ORDER BY trade_date DESC LIMIT 1",
            (d,),
        ).fetchone()
        if row and row[0]:
            out[d] = float(row[0])
    return out


def get_ma_regime(conn, calc_date: str, ma_window: int = 200) -> str:
    """KOSPI 200 (069500) 종가 vs MA(ma_window) → Bull/Bear."""
    rows = conn.execute(
        "SELECT close FROM daily_price WHERE stock_code='069500' "
        f"AND trade_date <= ? ORDER BY trade_date DESC LIMIT {ma_window + 1}",
        (calc_date,),
    ).fetchall()
    prices = [r[0] for r in rows if r[0]]
    if len(prices) < ma_window + 1:
        return "Bull"
    current = prices[0]
    ma_val = float(np.mean(prices[1:ma_window + 1]))
    return "Bull" if current >= ma_val else "Bear"


def collect_factor_stats(conn, calc_date: str, large_only: bool) -> dict | None:
    df = load_factor_data(conn, calc_date)
    if df is None or df.empty:
        return None

    if large_only:
        df = df[df["size_group"] == "large"].copy()

    if len(df) < 30:
        return None

    df, _ = run_regressions(df, REGRESSION_MODELS, OUTLIER_FILTERS)

    f_eps_m = df["f_eps_m"].dropna().to_numpy() if "f_eps_m" in df.columns else np.array([])
    att_per = (
        df["fper_epsg_attractiveness"].dropna().to_numpy()
        if "fper_epsg_attractiveness" in df.columns else np.array([])
    )

    def _stats(arr):
        if arr.size < 5:
            return {"n": int(arr.size), "mean": None, "median": None, "std": None, "var": None}
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)),
            "var": float(np.var(arr, ddof=1)),
        }

    return {
        "date": calc_date,
        "n_universe": int(len(df)),
        "f_eps_m": _stats(f_eps_m),
        "att_per": _stats(att_per),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--large-only", action="store_true",
                    help="대형주(size_group='large')만 사용. 미지정시 전체 유니버스.")
    ap.add_argument("--ma-window", type=int, default=200)
    ap.add_argument("--out", default=str(Path(__file__).parent / "factor_kospi_regime_quick.json"))
    args = ap.parse_args()

    conn = get_conn()
    print(f"[load] prefetching factor data…", flush=True)
    clear_prefetch_cache()
    clear_factor_cache()
    prefetch_all_data(conn, use_local_cache=True)

    dates = get_rebal_dates(conn, start=args.start)
    print(f"[load] {len(dates)} monthly rebal dates from {args.start}", flush=True)

    kospi_px = get_kospi_series(conn, dates)
    rows = []
    for i, d in enumerate(dates):
        s = collect_factor_stats(conn, d, large_only=args.large_only)
        if s is None:
            continue
        s["regime_ma"] = get_ma_regime(conn, d, ma_window=args.ma_window)
        s["regime_cycle"] = _get_regime_by_cycle(d)
        s["kospi_close"] = kospi_px.get(d)
        rows.append(s)
        if (i + 1) % 12 == 0:
            print(f"  ..{i+1}/{len(dates)} ({d})", flush=True)

    # ── 월별 데이터프레임 구성 ──
    df = pd.DataFrame([{
        "date": r["date"],
        "regime_ma": r["regime_ma"],
        "regime_cycle": r["regime_cycle"],
        "n": r["n_universe"],
        "f_eps_m_n": r["f_eps_m"]["n"],
        "f_eps_m_mean": r["f_eps_m"]["mean"],
        "f_eps_m_var": r["f_eps_m"]["var"],
        "f_eps_m_std": r["f_eps_m"]["std"],
        "att_per_n": r["att_per"]["n"],
        "att_per_mean": r["att_per"]["mean"],
        "att_per_var": r["att_per"]["var"],
        "att_per_std": r["att_per"]["std"],
        "kospi_close": r["kospi_close"],
    } for r in rows]).sort_values("date").reset_index(drop=True)

    # ── KOSPI 월별 수익률 ──
    df["kospi_ret"] = df["kospi_close"].pct_change()

    # ────────────────────────────────────────
    # 1) 상관계수: KOSPI 월별수익률 vs 월별 mean(F_EPS_M)
    # ────────────────────────────────────────
    # 케이스 A: 같은 시점 (concurrent)
    # 케이스 B: F_EPS_M(t) → KOSPI(t→t+1) 선행 신호
    sub = df.dropna(subset=["kospi_ret", "f_eps_m_mean"])
    pearson_concurrent = sub[["kospi_ret", "f_eps_m_mean"]].corr().iloc[0, 1]
    spearman_concurrent = sub[["kospi_ret", "f_eps_m_mean"]].corr(method="spearman").iloc[0, 1]

    df["kospi_ret_next"] = df["kospi_ret"].shift(-1)
    sub2 = df.dropna(subset=["kospi_ret_next", "f_eps_m_mean"])
    pearson_lead = sub2[["kospi_ret_next", "f_eps_m_mean"]].corr().iloc[0, 1]
    spearman_lead = sub2[["kospi_ret_next", "f_eps_m_mean"]].corr(method="spearman").iloc[0, 1]

    print("\n" + "=" * 72)
    print("[1] 월별 KOSPI 수익률 vs 월별 cross-sectional mean(F_EPS_M)")
    print("=" * 72)
    print(f"  표본수: {len(sub)} months ({sub['date'].min()} ~ {sub['date'].max()})")
    print(f"  Concurrent  Pearson r = {pearson_concurrent:+.4f}   Spearman ρ = {spearman_concurrent:+.4f}")
    print(f"  Lead (F→ret_next)  Pearson r = {pearson_lead:+.4f}   Spearman ρ = {spearman_lead:+.4f}")

    # ────────────────────────────────────────
    # 2) 레짐별 F_EPS_M 분산 vs ATT_PER 분산
    # ────────────────────────────────────────
    def _summary(group: pd.DataFrame, name: str) -> dict:
        return {
            "regime": name,
            "n_months": int(len(group)),
            "f_eps_m_var_mean": float(group["f_eps_m_var"].mean()),
            "f_eps_m_var_median": float(group["f_eps_m_var"].median()),
            "f_eps_m_std_mean": float(group["f_eps_m_std"].mean()),
            "att_per_var_mean": float(group["att_per_var"].mean()),
            "att_per_var_median": float(group["att_per_var"].median()),
            "att_per_std_mean": float(group["att_per_std"].mean()),
        }

    print("\n" + "=" * 72)
    print("[2-A] 레짐별 분산 비교 — MA(200) 기준 (KOSPI 200 vs MA200)")
    print("=" * 72)
    print(f"{'regime':<6} {'n_mo':>5}  {'fEPS_M var':>12} {'fEPS_M std':>12}  {'ATT_PER var':>12} {'ATT_PER std':>12}")
    ma_rows = []
    for r in ["Bull", "Bear"]:
        g = df[df["regime_ma"] == r]
        if len(g) == 0:
            continue
        s = _summary(g, r)
        ma_rows.append(s)
        print(f"{r:<6} {s['n_months']:>5}  {s['f_eps_m_var_mean']:>12.4f} {s['f_eps_m_std_mean']:>12.4f}  "
              f"{s['att_per_var_mean']:>12.4f} {s['att_per_std_mean']:>12.4f}")

    print("\n" + "=" * 72)
    print("[2-B] 레짐별 분산 비교 — Cycle 기준 (하드코딩된 약세장 기간)")
    print("=" * 72)
    print(f"  cycle bear periods: {_CYCLE_BEAR_PERIODS}")
    print(f"{'regime':<6} {'n_mo':>5}  {'fEPS_M var':>12} {'fEPS_M std':>12}  {'ATT_PER var':>12} {'ATT_PER std':>12}")
    cycle_rows = []
    for r in ["Bull", "Bear"]:
        g = df[df["regime_cycle"] == r]
        if len(g) == 0:
            continue
        s = _summary(g, r)
        cycle_rows.append(s)
        print(f"{r:<6} {s['n_months']:>5}  {s['f_eps_m_var_mean']:>12.4f} {s['f_eps_m_std_mean']:>12.4f}  "
              f"{s['att_per_var_mean']:>12.4f} {s['att_per_std_mean']:>12.4f}")

    # ── 저장 ──
    out_path = Path(args.out)
    payload = {
        "params": {
            "start": args.start,
            "large_only": args.large_only,
            "ma_window": args.ma_window,
            "regression": "fper_epsg (x=f_epsg, y=f_per, ratio)",
        },
        "monthly": df.to_dict(orient="records"),
        "corr_kospi_f_eps_m": {
            "concurrent_pearson": float(pearson_concurrent),
            "concurrent_spearman": float(spearman_concurrent),
            "lead_pearson": float(pearson_lead),
            "lead_spearman": float(spearman_lead),
            "n_months": int(len(sub)),
        },
        "regime_variance_ma": ma_rows,
        "regime_variance_cycle": cycle_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
