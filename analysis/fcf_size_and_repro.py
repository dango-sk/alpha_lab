"""
analysis/fcf_size_and_repro.py  (실험 스크립트, production 미수정)

[보완] (1) BASE 재현 검증: 저장된 production 월수익 vs holdings 재구성 수익 상관/차이
       (2) 시총그룹(Large/Mid/Small) 집계 비중·수익기여·그룹내 종목선택효과

실행: .venv/bin/python analysis/fcf_size_and_repro.py
산출: output/fcf_monthly_contrib.csv, output/fcf_size_attribution.csv(덮어씀),
      output/fcf_base_reproduction.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
load_dotenv(REPO / ".env")
sys.path.insert(0, str(REPO / "analysis"))
import importlib
fa = importlib.import_module("fcf_performance_attribution")
from lib.data import load_strategy                     # noqa: E402
from lib.db import get_conn, read_sql                  # noqa: E402

OUT = REPO / "output"


def main():
    sd = load_strategy(fa.STRATEGY, "monthly", "KOSPI")
    res = sd["results"]; rd = res["rebalance_dates"]; mr = res["monthly_returns"]
    saved = pd.Series({rd[i][:7]: mr[i] for i in range(len(mr))}).sort_index()
    h = fa.add_counterfactual_weights(fa.load_holdings(sd))
    conn = get_conn()
    dates = sorted(h["date"].unique())
    contrib = fa.monthly_contrib(conn, h, dates)
    contrib["size_group"] = contrib["market_cap"].map(fa.size_bucket)
    contrib["period"] = contrib["ym"].map(fa.period_label)
    contrib.to_csv(OUT / "fcf_monthly_contrib.csv", index=False, encoding="utf-8-sig")

    rec_gross = fa.portfolio_returns_from_weights(contrib, "w_production")
    to = fa.turnover_from_holdings(h, "w_production")
    rec_net30 = rec_gross - to.reindex(rec_gross.index).fillna(to.mean()) * 0.003 * 2
    idx = saved.index.intersection(rec_gross.index)
    rows = []
    for lbl, s in [("recon_gross", rec_gross), ("recon_net30bp", rec_net30)]:
        d = (s.reindex(idx) - saved.reindex(idx))
        rows.append({"spec": lbl, "months": len(idx),
                     "corr_with_saved": float(np.corrcoef(s.reindex(idx), saved.reindex(idx))[0, 1]),
                     "mean_diff_monthly": float(d.mean()), "mae_monthly": float(d.abs().mean()),
                     "max_abs_diff": float(d.abs().max()),
                     "cagr_recon": fa.perf_stats(s.reindex(idx))["cagr"],
                     "cagr_saved": fa.perf_stats(saved.reindex(idx))["cagr"]})
    pd.DataFrame(rows).to_csv(OUT / "fcf_base_reproduction.csv", index=False, encoding="utf-8-sig")

    # 시총그룹 집계 (월별 그룹비중 → 기간평균) + 그룹내 종목선택효과
    g = contrib.dropna(subset=["ret"]).copy()
    monthly_grp = g.groupby(["ym", "period", "size_group"]).apply(
        lambda x: pd.Series({"w": x["w_production"].sum(),
                             "r": np.average(x["ret"], weights=x["w_production"]) if x["w_production"].sum() > 0 else np.nan,
                             "contrib": x["contrib"].sum()}), include_groups=False).reset_index()
    # 벤치마크(유니버스) 그룹 비중·수익
    bm_rows = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        uni = read_sql("SELECT stock_code, market_cap FROM universe WHERE rebal_date=? AND rebal_type=?",
                       conn, params=(d0, "monthly"))
        if uni.empty: continue
        uni["code"] = uni["stock_code"].map(fa.strip_a)
        uni["market_cap"] = pd.to_numeric(uni["market_cap"], errors="coerce")
        uni = uni.dropna(subset=["market_cap"])
        rmap = fa.period_return_maps(conn, uni["code"].tolist(), d0, d1)
        uni["ret"] = uni["code"].map(rmap); uni = uni.dropna(subset=["ret"])
        if uni.empty: continue
        uni["size_group"] = uni["market_cap"].map(fa.size_bucket)
        tot = uni["market_cap"].sum()
        for sg, x in uni.groupby("size_group"):
            bm_rows.append({"ym": d0[:7], "size_group": sg,
                            "bm_w": x["market_cap"].sum() / tot,
                            "bm_r": float(np.average(x["ret"], weights=x["market_cap"]))})
    bm = pd.DataFrame(bm_rows)
    m = monthly_grp.merge(bm, on=["ym", "size_group"], how="outer")
    m["w"] = m["w"].fillna(0.0); m["contrib"] = m["contrib"].fillna(0.0)
    m["period"] = m["ym"].map(fa.period_label)
    m["selection_effect"] = m["bm_w"] * (m["r"] - m["bm_r"])
    m["allocation_effect"] = (m["w"] - m["bm_w"]) * m["bm_r"]
    out = m.groupby(["period", "size_group"]).agg(
        months=("ym", "nunique"), portfolio_weight=("w", "mean"), benchmark_weight=("bm_w", "mean"),
        portfolio_return=("r", "mean"), benchmark_return=("bm_r", "mean"),
        return_contribution=("contrib", "sum"),
        selection_effect_mean=("selection_effect", "mean"),
        allocation_effect_gross=("allocation_effect", "mean")).reset_index()
    out["active_weight"] = out.portfolio_weight - out.benchmark_weight
    for i, r in out.iterrows():
        sub = m[(m.period == r.period) & (m.size_group == r.size_group)]["selection_effect"]
        t, lag = fa.nw_tstat(sub); out.loc[i, "selection_nw_t"] = t; out.loc[i, "nw_lag"] = lag
    out.to_csv(OUT / "fcf_size_attribution.csv", index=False, encoding="utf-8-sig")
    print(out.round(4).to_string(index=False))
    print("\nWrote fcf_monthly_contrib.csv, fcf_size_attribution.csv, fcf_base_reproduction.csv")


if __name__ == "__main__":
    main()
