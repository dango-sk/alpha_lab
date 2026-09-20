"""
analysis/fcf_diet_counterfactual.py   (실험 스크립트, production 미수정)

[post-OOS diagnostic] DIET 성과 변화를 '종목 구성 효과'와 '비중(cap) 효과'로 분리.

  production selector(score_stocks_from_strategy)로 BASE / DIET-3 의 월별 Top30 을 재현하고,
  production 가중 함수(_apply_mcap_cap)로 cap 30% / 20% 비중을 각각 계산해
  2x2 반사실 포트폴리오를 만든다.

    BASE종목 × cap30  (= BASE)
    BASE종목 × cap20
    DIET3종목 × cap30 (= DIET-3)
    DIET3종목 × cap20 (= DIET-3-CAP20)

실행: .venv/bin/python analysis/fcf_diet_counterfactual.py
산출: output/diet_counterfactual.csv, output/diet_stock_contrib.csv
"""
import math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
load_dotenv(REPO / ".env")
sys.path.insert(0, str(REPO / "analysis"))

from lib.data import load_strategy                                        # noqa: E402
from lib.factor_engine import code_to_module, score_stocks_from_strategy  # noqa: E402
from step7_backtest import (                                              # noqa: E402
    get_universe_stocks, get_db, get_rebalance_dates, _apply_mcap_cap, _calc_slippage,
)
import fcf_diet_compare as dc                                             # noqa: E402

OUT = REPO / "output"
IS_END, OOS_START = "2024-06", "2024-07"
OOS_LABEL = "OOS_post-OOS diagnostic"


def holdings_for(conn, module, dates, cap_pct):
    rows = []
    for d in dates:
        uni = set(get_universe_stocks(conn, d, "monthly"))
        if not uni: continue
        sel = [(c, s) for c, s in score_stocks_from_strategy(conn, d, module) if c in uni][:30]
        if not sel: continue
        codes = [c for c, _ in sel]
        ph = ",".join(["?"] * len(codes))
        mc = dict(conn.execute(f"""
            SELECT dp.stock_code, dp.market_cap FROM daily_price dp
            JOIN (SELECT stock_code, MIN(trade_date) d FROM daily_price
                  WHERE stock_code IN ({ph}) AND trade_date >= ? GROUP BY stock_code) t
              ON dp.stock_code=t.stock_code AND dp.trade_date=t.d
        """, (*codes, d)).fetchall())
        raw = [mc.get(c, 0) or 0 for c in codes]
        w = _apply_mcap_cap(raw, cap=cap_pct / 100)
        for j, c in enumerate(codes):
            rows.append({"date": d, "ym": d[:7], "code": c, "weight": w[j], "market_cap": raw[j]})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    conn = get_db()
    sd = load_strategy(dc.STRATEGY, "monthly", "KOSPI")
    base_mod = code_to_module(sd["code"])
    diet3_mod = code_to_module(dc.drop_factors(sd["code"], dc.DROP3)[0])
    diet4_mod = code_to_module(dc.drop_factors(sd["code"], dc.DROP4)[0])
    dates = get_rebalance_dates(conn, "monthly")

    specs = {}
    for label, mod in [("BASE종목", base_mod), ("DIET3종목", diet3_mod), ("DIET4종목", diet4_mod)]:
        for cap in [30, 20]:
            specs[f"{label}×cap{cap}"] = holdings_for(conn, mod, dates, cap)
            print(f"  {label}×cap{cap} holdings 완료 ({time.time()-t0:.0f}s)", flush=True)

    # 종목 수익률 (종목 집합은 cap과 무관 → 한 번만 계산)
    all_hold = pd.concat(specs.values(), ignore_index=True)
    ret_rows = {}
    for d0, d1 in zip(dates[:-1], dates[1:]):
        codes = all_hold[all_hold["date"] == d0]["code"].unique().tolist()
        if not codes: continue
        p0 = dc.price_map(conn, codes, d0, "first"); p1 = dc.price_map(conn, codes, d1, "last")
        for c in codes:
            a, b = p0.get(c), p1.get(c)
            ret_rows[(d0[:7], c)] = (b / a - 1.0) if (a and b and a > 0) else np.nan
    print(f"  종목 수익률 완료 ({time.time()-t0:.0f}s)", flush=True)

    bm = {}
    for d0, d1 in zip(dates[:-1], dates[1:]):
        a = dc.price_map(conn, [dc.BM], d0, "first").get(dc.BM)
        b = dc.price_map(conn, [dc.BM], d1, "last").get(dc.BM)
        bm[d0[:7]] = (b / a - 1.0) if (a and b) else np.nan
    bm = pd.Series(bm).sort_index()

    rows, contrib_rows = [], []
    series = {}
    for name, h in specs.items():
        h = h.copy()
        h["ret"] = [ret_rows.get((ym, c), np.nan) for ym, c in zip(h["ym"], h["code"])]
        h["contrib"] = h["weight"] * h["ret"]
        h["slip"] = h["market_cap"].map(_calc_slippage)
        gross = h.groupby("ym")["contrib"].sum().sort_index()
        gross = gross[gross.index.isin(bm.index)]
        # turnover (첫 달 1.0, production 규칙)
        to, prev = {}, None
        for ym, g in h.groupby("ym", sort=True):
            cur = dict(zip(g["code"], g["weight"]))
            to[ym] = 1.0 if prev is None else 0.5 * sum(
                abs(cur.get(k, 0) - prev.get(k, 0)) for k in set(cur) | set(prev))
            prev = cur
        to = pd.Series(to).reindex(gross.index)
        slip = h.groupby("ym")["slip"].mean().reindex(gross.index)
        net30 = gross - to * (0.003 + slip) * 2
        series[name] = net30
        for period, ix in dc.period_slices(gross.index).items():
            r = net30.reindex(ix).dropna()
            ex = (r - bm.reindex(r.index)).dropna()
            t, lag = dc.nw_t(ex)
            rows.append({"spec": name, "period": period, **dc.perf(r),
                         "market_excess_mean": float(ex.mean()), "market_excess_nw_t": t, "nw_lag": lag,
                         "turnover_mean": float(to.reindex(r.index).mean()),
                         "max_weight_mean": float(h[h.ym.isin(r.index)].groupby("ym")["weight"].max().mean()),
                         "top5_weight_mean": float(h[h.ym.isin(r.index)].groupby("ym")["weight"]
                                                   .apply(lambda s: s.nlargest(5).sum()).mean())})
        agg = h.groupby("code")["contrib"].sum().sort_values(ascending=False)
        nm = dc.name_map(conn, agg.index.tolist())
        for kind, v in [("top10", agg.head(10)), ("bottom10", agg.tail(10))]:
            for c, val in v.items():
                contrib_rows.append({"spec": name, "kind": kind, "code": c,
                                     "name": nm.get(c, ""), "cum_contrib": float(val)})

    df = pd.DataFrame(rows)
    # 효과 분해 (Full/IS/OOS, net30 기준 CAGR·월평균)
    dec = []
    for period in ["Full", "IS", OOS_LABEL]:
        g = df[df.period == period].set_index("spec")
        row = {"period": period}
        for d in ["DIET3", "DIET4"]:
            row[f"종목효과_{d}_cap30"] = g.loc[f"{d}종목×cap30", "cagr"] - g.loc["BASE종목×cap30", "cagr"]
            row[f"종목효과_{d}_cap20"] = g.loc[f"{d}종목×cap20", "cagr"] - g.loc["BASE종목×cap20", "cagr"]
            row[f"비중효과_{d}종목(cap20-cap30)"] = g.loc[f"{d}종목×cap20", "cagr"] - g.loc[f"{d}종목×cap30", "cagr"]
            row[f"총효과_{d}cap20-BASEcap30"] = g.loc[f"{d}종목×cap20", "cagr"] - g.loc["BASE종목×cap30", "cagr"]
        row["비중효과_BASE종목(cap20-cap30)"] = g.loc["BASE종목×cap20", "cagr"] - g.loc["BASE종목×cap30", "cagr"]
        dec.append(row)
    df.to_csv(OUT / "diet_counterfactual.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dec).to_csv(OUT / "diet_counterfactual_decomposition.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(contrib_rows).to_csv(OUT / "diet_stock_contrib.csv", index=False, encoding="utf-8-sig")
    print(df.round(4).to_string(index=False))
    print("\n== 효과 분해 (CAGR, net30) ==")
    print(pd.DataFrame(dec).round(4).to_string(index=False))
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
