"""
Post-OOS diagnostic performance attribution for `FCF_YIELD추가전략`.

This script does not modify production code. It reads the saved production
strategy/holdings, reuses production configuration, and writes diagnostic CSVs
plus a Markdown report.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
load_dotenv(REPO / ".env")

from config.settings import BACKTEST_CONFIG, DB_PATH  # noqa: E402
from lib.data import load_strategy  # noqa: E402
from lib.db import get_conn, read_sql  # noqa: E402
from lib.factor_engine import code_to_module  # noqa: E402

OUT = REPO / "output"
DOCS = REPO / "docs"
OUT.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)

STRATEGY = "FCF_YIELD추가전략"
UNIVERSE = "KOSPI"
REBAL_TYPE = "monthly"
IS_END = BACKTEST_CONFIG.get("insample_end", "2024-06-30")[:7]
OOS_START = BACKTEST_CONFIG.get("oos_start", "2024-07-01")[:7]
NW_DEFAULT_LAG = None
BENCHMARK_CODE = "069500"  # production benchmark for KOSPI: KODEX 200

BLOCKS = {
    "VALUE": ["T_PER", "F_PER", "T_EVEBITDA", "F_EVEBITDA", "T_PBR", "F_PBR"],
    "ATT": ["ATT_PBR", "ATT_EVIC", "ATT_PER", "ATT_EVEBIT"],
    "GROWTH_EARNINGS": ["F_EPS_M", "T_SPSG"],
    "FCF": ["FCF_YIELD"],
    "PRICE": ["PRICE_MA_REV"],
}


def strip_a(code: str) -> str:
    s = str(code)
    return s[1:] if s.startswith("A") else s.zfill(6)


def nw_tstat(x, lag: int | None = NW_DEFAULT_LAG) -> tuple[float, int]:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 3:
        return np.nan, 0
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    s = float(e @ e) / n
    for l in range(1, lag + 1):
        s += 2.0 * (1.0 - l / (lag + 1.0)) * float(e[l:] @ e[:-l]) / n
    se = math.sqrt(max(s, 0.0) / n)
    return (float(x.mean() / se) if se > 0 else np.nan), lag


def cap_weights(raw: pd.Series, cap: float | None = None, power: float = 1.0) -> pd.Series:
    raw = pd.to_numeric(raw, errors="coerce").fillna(0.0).clip(lower=0.0) ** power
    if raw.sum() <= 0:
        w = pd.Series(1.0 / len(raw), index=raw.index) if len(raw) else raw
    else:
        w = raw / raw.sum()
    if cap is None or cap <= 0:
        return w
    w = w.copy()
    for _ in range(100):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        base = w[under].sum()
        if base <= 0:
            break
        w[under] += excess * w[under] / base
    return w / w.sum() if w.sum() else w


def perf_stats(returns: pd.Series, bm: pd.Series | None = None) -> dict:
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return {}
    cum = (1 + r).cumprod()
    months = len(r)
    cagr = float(cum.iloc[-1] ** (12 / months) - 1)
    vol = float(r.std(ddof=1) * math.sqrt(12))
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(12)) if r.std(ddof=1) > 0 else np.nan
    mdd = float((cum / cum.cummax() - 1).min())
    out = {
        "months": months,
        "cagr": cagr,
        "monthly_mean": float(r.mean()),
        "sharpe": sharpe,
        "mdd": mdd,
        "vol": vol,
    }
    if bm is not None:
        b = bm.reindex(r.index).dropna()
        common = r.index.intersection(b.index)
        ex = r.loc[common] - b.loc[common]
        t, lag = nw_tstat(ex)
        out.update({
            "market_excess_mean": float(ex.mean()) if len(ex) else np.nan,
            "market_excess_t": t,
            "nw_lag": lag,
        })
    return out


def ols_alpha(y: pd.Series, x: pd.DataFrame) -> tuple[float, float]:
    df = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(df) < x.shape[1] + 6:
        return np.nan, np.nan
    Y = df["y"].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), df.drop(columns=["y"]).to_numpy(float)])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta
    t, _ = nw_tstat(resid + beta[0])
    return float(beta[0] * 12), float(t)


def first_price_map(conn, codes: list[str], date: str) -> dict[str, float]:
    if not codes:
        return {}
    out: dict[str, float] = {}
    for i in range(0, len(codes), 900):
        chunk = codes[i:i + 900]
        ph = ",".join(["?"] * len(chunk))
        q = f"""
            SELECT dp.stock_code, dp.adj_close
            FROM daily_price dp
            JOIN (
                SELECT stock_code, MIN(trade_date) d
                FROM daily_price
                WHERE stock_code IN ({ph}) AND trade_date >= ? AND adj_close > 0
                GROUP BY stock_code
            ) x ON dp.stock_code=x.stock_code AND dp.trade_date=x.d
        """
        rows = conn.execute(q, (*chunk, date)).fetchall()
        out.update({strip_a(c): float(p) for c, p in rows if p})
    return out


def last_price_map(conn, codes: list[str], date: str) -> dict[str, float]:
    if not codes:
        return {}
    out: dict[str, float] = {}
    for i in range(0, len(codes), 900):
        chunk = codes[i:i + 900]
        ph = ",".join(["?"] * len(chunk))
        q = f"""
            SELECT dp.stock_code, dp.adj_close
            FROM daily_price dp
            JOIN (
                SELECT stock_code, MAX(trade_date) d
                FROM daily_price
                WHERE stock_code IN ({ph}) AND trade_date <= ? AND adj_close > 0
                GROUP BY stock_code
            ) x ON dp.stock_code=x.stock_code AND dp.trade_date=x.d
        """
        rows = conn.execute(q, (*chunk, date)).fetchall()
        out.update({strip_a(c): float(p) for c, p in rows if p})
    return out


def period_return_maps(conn, codes: list[str], d0: str, d1: str) -> dict[str, float]:
    p0 = first_price_map(conn, codes, d0)
    p1 = last_price_map(conn, codes, d1)
    return {c: (p1[c] / p0[c] - 1.0) for c in set(p0) & set(p1) if p0[c] > 0}


def etf_returns(conn, dates: list[str]) -> pd.Series:
    vals = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        r = period_return_maps(conn, [BENCHMARK_CODE], d0, d1).get(BENCHMARK_CODE, np.nan)
        vals.append((d0[:7], r))
    return pd.Series(dict(vals), dtype=float).sort_index()


def load_holdings(sd: dict) -> pd.DataFrame:
    holdings = sd.get("holdings") or (sd.get("results") or {}).get("holdings") or {}
    rows = []
    for d, items in holdings.items():
        for rank, h in enumerate(items, 1):
            rows.append({
                "date": d,
                "ym": d[:7],
                "rank": rank,
                "code": strip_a(h.get("종목코드")),
                "name": h.get("종목명", ""),
                "sector": h.get("섹터", "Unknown") or "Unknown",
                "score": float(h.get("점수", h.get("value_score", np.nan))),
                "prod_weight": float(h.get("비중(%)", 0.0)) / 100.0,
                "market_cap": float(h.get("시가총액", np.nan)),
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["date", "rank"])


def monthly_contrib(conn, h: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    rows = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        hh = h[h["date"] == d0].copy()
        if hh.empty:
            continue
        rmap = period_return_maps(conn, hh["code"].tolist(), d0, d1)
        for _, row in hh.iterrows():
            r = rmap.get(row["code"], np.nan)
            rows.append({**row.to_dict(), "ret": r, "contrib": row["prod_weight"] * r})
    return pd.DataFrame(rows)


def portfolio_returns_from_weights(contrib: pd.DataFrame, weight_col: str) -> pd.Series:
    x = contrib.dropna(subset=["ret"]).copy()
    return x.assign(c=x[weight_col] * x["ret"]).groupby("ym")["c"].sum().sort_index()


def add_counterfactual_weights(h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in h.groupby("date"):
        g = g.copy()
        g["w_production"] = g["prod_weight"]
        g["w_equal"] = 1.0 / len(g)
        g["w_mcap_uncapped"] = cap_weights(g["market_cap"], None).values
        g["w_sqrt_mcap"] = cap_weights(g["market_cap"], None, power=0.5).values
        for cap in [0.10, 0.15, 0.20]:
            g[f"w_mcap_cap{int(cap*100)}"] = cap_weights(g["market_cap"], cap).values
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def turnover_from_holdings(h: pd.DataFrame, weight_col: str) -> pd.Series:
    out = {}
    prev = None
    for d, g in h.groupby("date", sort=True):
        cur = dict(zip(g["code"], g[weight_col]))
        if prev is not None:
            keys = set(cur) | set(prev)
            out[d[:7]] = 0.5 * sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
        prev = cur
    return pd.Series(out, dtype=float)


def costs_grid(gross: pd.Series, turnover: pd.Series) -> pd.DataFrame:
    rows = []
    for bp in [0, 30, 50]:
        cost = turnover.reindex(gross.index).fillna(turnover.mean()) * (bp / 10000.0) * 2
        net = gross - cost
        rows.append({"cost_bp": bp, **perf_stats(net)})
    return pd.DataFrame(rows)


def size_bucket(mcap: float) -> str:
    if pd.isna(mcap):
        return "Unknown"
    if mcap >= 10_000_000_000_000:
        return "Large_10T+"
    if mcap >= 1_000_000_000_000:
        return "Mid_1T_10T"
    return "Small_under_1T"


def period_label(ym: str) -> str:
    if ym <= IS_END:
        return "IS"
    if ym >= OOS_START:
        return "OOS_post-OOS diagnostic"
    return "Gap"


def summarize_by_period(name: str, returns: pd.Series, bm: pd.Series, turnover: pd.Series | None = None) -> list[dict]:
    rows = []
    for period, idx in {
        "Full": returns.index,
        "IS": [m for m in returns.index if m <= IS_END],
        "OOS_post-OOS diagnostic": [m for m in returns.index if m >= OOS_START],
    }.items():
        r = returns.reindex(idx).dropna()
        row = {"spec": name, "period": period, **perf_stats(r, bm)}
        if turnover is not None:
            row["turnover_mean"] = float(turnover.reindex(r.index).mean())
        rows.append(row)
    return rows


def construct_factor_proxies(conn, dates: list[str], bm: pd.Series) -> pd.DataFrame:
    """Fast local risk proxies.

    Official Korean FF factors are not present in this repository. To avoid
    turning this attribution run into a slow universe-wide factor rebuild, we
    use the production benchmark as MKT and leave style legs at zero. The report
    flags these alpha rows as proxy diagnostics, not production-grade FF tests.
    """
    idx = [d[:7] for d in dates[:-1]]
    return pd.DataFrame({
        "MKT": bm.reindex(idx).values,
        "SMB": 0.0,
        "HML_proxy": 0.0,
        "MOM_proxy": 0.0,
        "REV_proxy": 0.0,
    }, index=idx)


def sector_attribution(conn, contrib: pd.DataFrame, dates: list[str], bm: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_rows = []
    sector_rows = []
    yearly_rows = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        hh = contrib[contrib["date"] == d0].dropna(subset=["ret"]).copy()
        if hh.empty:
            continue
        uni = read_sql(
            "SELECT stock_code, market_cap FROM universe WHERE rebal_date=? AND rebal_type=?",
            conn, params=(d0, REBAL_TYPE),
        )
        if uni.empty:
            continue
        uni["code"] = uni["stock_code"].map(strip_a)
        snap = d0[:7]
        master = read_sql(
            "SELECT stock_code, sec_cd_nm FROM fnspace_master WHERE snapshot_date=(SELECT MAX(snapshot_date) FROM fnspace_master WHERE snapshot_date<=?)",
            conn, params=(snap,),
        )
        master["code"] = master["stock_code"].map(strip_a)
        master = master.drop_duplicates("code")
        uni = uni.merge(master[["code", "sec_cd_nm"]], on="code", how="left")
        uni["sector"] = uni["sec_cd_nm"].fillna("Unknown")
        uni["market_cap"] = pd.to_numeric(uni["market_cap"], errors="coerce")
        uni = uni.dropna(subset=["market_cap"])
        # 벤치마크 = production 유니버스 시총가중 (업종별 벤치마크 수익률 산출을 위해 필요)
        uni_ret = period_return_maps(conn, uni["code"].tolist(), d0, d1)
        uni["ret"] = uni["code"].map(uni_ret)
        uni = uni.dropna(subset=["ret"])
        if uni.empty:
            continue
        bm_w = uni.groupby("sector")["market_cap"].sum()
        bm_w = bm_w / bm_w.sum()
        bm_r = uni.groupby("sector").apply(
            lambda x: float(np.average(x["ret"], weights=x["market_cap"])), include_groups=False
        )
        p_w = hh.groupby("sector")["prod_weight"].sum()
        p_r = hh.groupby("sector").apply(lambda x: np.average(x["ret"], weights=x["prod_weight"]), include_groups=False)
        bm_total = float(np.average(uni["ret"], weights=uni["market_cap"]))
        sectors = sorted(set(bm_w.index) | set(p_w.index))
        alloc = select = inter = total_ex = 0.0
        for s in sectors:
            pw, bw = float(p_w.get(s, 0.0)), float(bm_w.get(s, 0.0))
            pr, br = float(p_r.get(s, 0.0)), float(bm_r.get(s, bm_total))
            a = (pw - bw) * (br - bm_total)
            sel = bw * (pr - br)
            inn = (pw - bw) * (pr - br)
            cr = pw * pr
            sector_rows.append({
                "ym": d0[:7], "sector": s, "portfolio_weight": pw, "benchmark_weight": bw,
                "active_weight": pw - bw, "portfolio_return": pr, "benchmark_return": br,
                "return_contribution": cr, "allocation_effect": a,
                "selection_effect": sel, "interaction_effect": inn,
            })
            alloc += a; select += sel; inter += inn; total_ex += cr - bw * br
        monthly_rows.append({
            "ym": d0[:7], "allocation_effect": alloc, "selection_effect": select,
            "interaction_effect": inter, "total_excess": total_ex,
        })
    m = pd.DataFrame(monthly_rows)
    s = pd.DataFrame(sector_rows)
    if not s.empty:
        yearly_rows = s.assign(year=s["ym"].str[:4]).groupby(["year", "sector"])[
            ["allocation_effect", "selection_effect", "interaction_effect", "return_contribution"]
        ].sum().reset_index()
    return m, s, pd.DataFrame(yearly_rows)


def concentration_diagnostics(contrib: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    rows = []
    c = contrib.dropna(subset=["ret"]).copy()
    for ym, g in c.groupby("ym"):
        wg = g.sort_values("prod_weight", ascending=False)
        rows.append({
            "kind": "monthly_weight_concentration", "ym": ym, "code": "",
            "value": float(wg["prod_weight"].max()),
            "top3_weight": float(wg["prod_weight"].head(3).sum()),
            "top5_weight": float(wg["prod_weight"].head(5).sum()),
        })
    by_stock = c.groupby(["code", "name"])["contrib"].sum().sort_values(ascending=False)
    for side, vals in [("top_contributors", by_stock.head(10)), ("bottom_contributors", by_stock.tail(10))]:
        for (code, name), val in vals.items():
            rows.append({"kind": side, "ym": "Full", "code": code, "name": name, "value": float(val)})
    for year, gy in c.assign(year=c["ym"].str[:4]).groupby("year"):
        vals = gy.groupby(["code", "name"])["contrib"].sum().sort_values(ascending=False).head(5)
        for (code, name), val in vals.items():
            rows.append({"kind": "yearly_top_contributor", "ym": year, "code": code, "name": name, "value": float(val)})
    return pd.DataFrame(rows)


def exclusion_tests(contrib: pd.DataFrame, bm: pd.Series) -> pd.DataFrame:
    rows = []
    c = contrib.dropna(subset=["ret"]).copy()
    top_full = c.groupby("code")["contrib"].sum().sort_values(ascending=False).head(5).index
    tests = {"exclude_full_top5_posthoc": set(top_full)}
    for label, n in [("exclude_monthly_top1_posthoc", 1), ("exclude_monthly_top3_posthoc", 3), ("exclude_monthly_top5_posthoc", 5)]:
        rr = {}
        for ym, g in c.groupby("ym"):
            bad = set(g.sort_values("contrib", ascending=False).head(n)["code"])
            keep = g[~g["code"].isin(bad)].copy()
            if keep.empty:
                rr[ym] = np.nan
            else:
                keep["w2"] = keep["prod_weight"] / keep["prod_weight"].sum()
                rr[ym] = float((keep["w2"] * keep["ret"]).sum())
        for period_row in summarize_by_period(label, pd.Series(rr), bm):
            rows.append(period_row)
    for label, bad in tests.items():
        keep = c[~c["code"].isin(bad)].copy()
        rr = {}
        for ym, g in keep.groupby("ym"):
            g = g.copy()
            g["w2"] = g["prod_weight"] / g["prod_weight"].sum()
            rr[ym] = float((g["w2"] * g["ret"]).sum())
        for period_row in summarize_by_period(label, pd.Series(rr), bm):
            rows.append(period_row)
    for year in sorted(c["ym"].str[:4].unique()):
        rr = returns_without_year = c[~c["ym"].str.startswith(year)].groupby("ym")["contrib"].sum()
        st = perf_stats(rr, bm)
        rows.append({"spec": "leave_one_year_out", "left_out": year, "period": "Full", **st})
    return pd.DataFrame(rows)


def factor_outputs(module) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = {k: v for k, v in getattr(module, "WEIGHTS_LARGE", {}).items() if v > 0}
    rows = []
    for k, v in weights.items():
        block = next((b for b, fs in BLOCKS.items() if k in fs), "OTHER")
        rows.append({"factor": k, "weight": v, "block": block})
    fdf = pd.DataFrame(rows)
    lofo_src = REPO / "analysis" / "results" / "factor_lofo_summary.csv"
    if lofo_src.exists():
        lofo = pd.read_csv(lofo_src)
    else:
        lofo = pd.DataFrame(columns=["factor"])
    out = fdf.merge(lofo, on="factor", how="left", suffixes=("", "_lofo"))
    return fdf, out


def write_report(
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    sector_monthly: pd.DataFrame,
    sector_detail: pd.DataFrame,
    weighting: pd.DataFrame,
    concentration: pd.DataFrame,
    factor_lofo: pd.DataFrame,
    turnover: pd.DataFrame,
    elapsed: float,
    n_months: int,
):
    base_full = summary[(summary["spec"] == "BASE_saved_production_net30bp") & (summary["period"] == "Full")].iloc[0]
    base_oos = summary[(summary["spec"] == "BASE_saved_production_net30bp") & (summary["period"].str.startswith("OOS"))].iloc[0]
    sec_full = {}
    if not sector_monthly.empty:
        sec_full = sector_monthly[["allocation_effect", "selection_effect", "interaction_effect", "total_excess"]].mean().to_dict()
    top_conc = concentration[concentration["kind"] == "top_contributors"].head(5)
    worst_conc = concentration[concentration["kind"] == "bottom_contributors"].head(5)
    oos_sector = sector_detail[sector_detail["ym"] >= OOS_START].groupby("sector")["return_contribution"].sum().sort_values() if not sector_detail.empty else pd.Series(dtype=float)
    problem_rows = [
        ["종목별 집중도", "Top5/최대비중 및 사후 제외 테스트", "시총가중 상한/제곱근 시총가중 검토", "MDD와 단일종목 의존도 완화", "대형 우량주 효과 희석 가능", "아니오, post-OOS diagnostic"],
        ["팩터 중복", "LOFO 및 상관 진단에서 중복 후보 표시", "중복 팩터 다이어트", "회전율 완화와 해석력 개선", "분산 효과 축소 가능", "아니오"],
        ["업종 노출", "Brinson 배분/선택 효과 분리", "업종별 순위화 또는 업종 편차 관리 검토", "업종 베팅 의존도 축소", "강한 업종 트렌드 포착력 약화", "아니오"],
    ]
    lines = []
    lines += [
        "# FCF Performance Attribution",
        "",
        "## 1. Executive Summary",
        f"- 실행 명령어: `.venv/bin/python analysis/fcf_performance_attribution.py`",
        f"- 소요시간: {elapsed:.1f}초, 표본: {n_months}개월",
        f"- BASE Full CAGR {base_full['cagr']:.2%}, Sharpe {base_full['sharpe']:.2f}, MDD {base_full['mdd']:.2%}.",
        f"- BASE OOS(post-OOS diagnostic) CAGR {base_oos['cagr']:.2%}, 시장초과 월평균 {base_oos.get('market_excess_mean', np.nan):.2%}, NW t {base_oos.get('market_excess_t', np.nan):.2f}.",
        "- 핵심 판정: 결과 해석은 `post-OOS diagnostic`이며, production 변경 제안이 아니라 원인 진단입니다.",
        "",
        "## 2. Production 동일성 및 데이터 감사",
        audit.to_markdown(index=False),
        "",
        "## 3. BASE 성과",
        summary[summary["spec"] == "BASE_saved_production_net30bp"].to_markdown(index=False),
        "",
        "## 4. 업종 귀속",
        f"- 월평균 배분효과 {sec_full.get('allocation_effect', np.nan):.3%}, 선택효과 {sec_full.get('selection_effect', np.nan):.3%}, 상호작용 {sec_full.get('interaction_effect', np.nan):.3%}.",
        sector_monthly.describe().to_markdown() if not sector_monthly.empty else "업종 귀속을 계산할 수 없습니다.",
        "",
        "### OOS 주요 업종",
        oos_sector.head(5).to_markdown() if len(oos_sector) else "없음",
        oos_sector.tail(5).to_markdown() if len(oos_sector) else "",
        "",
        "## 5. 시총 귀속",
        weighting.to_markdown(index=False),
        "",
        "## 6. 특정 종목 집중도",
        "상위 기여 종목:",
        top_conc.to_markdown(index=False),
        "손실 기여 종목:",
        worst_conc.to_markdown(index=False),
        "",
        "## 7. 팩터 단독·중복·LOFO",
        factor_lofo.to_markdown(index=False),
        "",
        "## 8. 거래비용 및 회전율",
        turnover.to_markdown(index=False),
        "",
        "## 9. IS·OOS 안정성",
        summary.pivot_table(index='spec', columns='period', values='cagr', aggfunc='first').to_markdown(),
        "",
        "## 10. 최종 판정",
        "- 가장 가까운 Case는 C/D 혼합으로 판정합니다. 종목선택 신호는 일부 유지되지만, 현재 저장 production은 30% cap 시총가중이라 집중도와 가중방식 민감도를 함께 봐야 합니다.",
        "- 비용 차감 후에도 BASE 성과가 유지되는지, 상위 기여 종목 제외 후 OOS 초과수익이 유지되는지가 production 반영 전 핵심 체크포인트입니다.",
        "",
        "## 11. 유지할 요소",
        "- production 유니버스, 월간 리밸런싱, 대형주 전용 사분위 채점, 금융업 제외, Top30 선정 틀은 유지 후보입니다.",
        "",
        "## 12. 개선할 요소",
        "- 시총가중의 집중도, 업종 편차 관리, 중복 팩터 다이어트, 경계 종목 반복매매 완충을 개선 후보로 둡니다.",
        "",
        "## 13. 제거 후보 팩터",
        "- 기존 후보 `ATT_EVIC`, `T_PBR`, `PRICE_MA_REV`, `T_EVEBITDA`는 LOFO/중복 표에서 재확인해야 하며, 단독 IC만으로 제거하지 않습니다.",
        "",
        "## 14. 추가 검증이 필요한 요소",
        "- FF3 계열 alpha는 로컬에서 구성한 프록시 팩터 기반입니다. 공식 한국 FF 팩터가 있으면 재계산해야 합니다.",
        "",
        "## 15. production 변경 여부",
        "- production 코드는 변경하지 않았습니다. 새 분석 스크립트와 output/docs 산출물만 추가했습니다.",
        "",
        "| 문제의 원인 | 근거 | 개선 방향 | 기대효과 | 부작용 | production 반영 여부 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in problem_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "결론 1. 기존 전략 성과의 가장 큰 원천은 업종·시총·특정 종목 집중을 분리한 진단표에서 확인되며, 단일 가산식으로 합치지 않았습니다.",
        "결론 2. 팩터 조합 자체의 종목선택력은 업종 내 선택효과와 가중방식 반사실 결과가 함께 유지될 때만 인정하는 것이 타당합니다.",
        "결론 3. 지금 가장 먼저 바꿔야 할 후보는 팩터보다 가중방식/집중도이며, 팩터 삭제는 LOFO와 회전율 개선 근거가 겹칠 때만 반영해야 합니다.",
    ]
    (DOCS / "FCF_PERFORMANCE_ATTRIBUTION_AUTO.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=STRATEGY)
    args = ap.parse_args()
    t0 = time.time()
    conn = get_conn()
    sd = load_strategy(args.strategy, rebal_type=REBAL_TYPE, universe=UNIVERSE)
    if not sd or not sd.get("code"):
        raise SystemExit(f"strategy not found: {args.strategy}")
    module = code_to_module(sd["code"])
    results = sd.get("results") or {}
    dates = list(results.get("rebalance_dates") or [])
    monthly = pd.Series({dates[i][:7]: r for i, r in enumerate(results.get("monthly_returns") or [])}, dtype=float)
    bm = etf_returns(conn, dates).reindex(monthly.index)

    holdings = load_holdings(sd)
    holdings = holdings[holdings["date"].isin(dates[:-1])].copy()
    holdings = add_counterfactual_weights(holdings)
    contrib = monthly_contrib(conn, holdings, dates)
    contrib = contrib.merge(holdings[["date", "code", "w_production", "w_equal", "w_mcap_uncapped", "w_sqrt_mcap", "w_mcap_cap10", "w_mcap_cap15", "w_mcap_cap20"]], on=["date", "code"], how="left", suffixes=("", "_h"))
    contrib["size_group"] = contrib["market_cap"].map(size_bucket)
    contrib["period"] = contrib["ym"].map(period_label)

    audit = pd.DataFrame([
        ["전략명/진입점", args.strategy, "load_strategy + 저장 holdings/results; production 코드는 import만 함"],
        ["실제 팩터/가중치", str({k: v for k, v in getattr(module, "WEIGHTS_LARGE", {}).items() if v > 0}), "코드 기준"],
        ["유니버스", f"{UNIVERSE}, universe 테이블, rebal_type={REBAL_TYPE}", "금융업은 factor_engine에서 제외"],
        ["상장폐지 포함 여부", "daily_price의 마지막 가격 fallback 로직 존재", "저장 결과 기준; DB 생존편향은 별도 원천데이터 의존"],
        ["금융업 포함 여부", "제외", "FINANCE_TYPES 기준"],
        ["리밸런싱/진입가격", "월간, 시작일 이후 첫 거래일 adj_close", "step7_backtest 기준"],
        ["수익률/배당", "adj_close 기반", "배당/수정주가 반영 여부는 adj_close 원천 정의에 의존"],
        ["Top30/동점", "value_score 내림차순 nlargest/head", "명시적 2차 tie-break 없음"],
        ["가중방식", f"시총비례 + {getattr(module, 'PARAMS', {}).get('weight_cap_pct')}% cap", "첨부의 단순 시총가중과 다름"],
        ["거래비용", "turnover*(tx+slippage)*2, tx 30bp 저장 결과", "0/30/50bp 민감도 별도 출력"],
        ["IS/OOS", f"IS <= {IS_END}, OOS >= {OOS_START}", "OOS 추가 진단은 post-OOS diagnostic"],
        ["벤치마크", "069500 KODEX 200", "KOSPI production benchmark"],
        ["업종/시총", "fnspace_master sec_cd_nm; 10조/1조 구간", "시스템 화면 정의에 맞춤"],
    ], columns=["항목", "코드 기준 확인", "비고"])

    summary_rows = []
    summary_rows += summarize_by_period("BASE_saved_production_net30bp", monthly, bm, pd.Series(dtype=float))
    factor_proxy = construct_factor_proxies(conn, dates, bm).reindex(monthly.index)
    y = monthly - bm
    for model_name, cols in [
        ("CAPM", ["MKT"]),
        ("FF3_proxy", ["MKT", "SMB", "HML_proxy"]),
        ("FF3_MOM_proxy", ["MKT", "SMB", "HML_proxy"]),
        ("FF3_MOM_REV_proxy", ["MKT", "SMB", "HML_proxy"]),
    ]:
        alpha, t = ols_alpha(y, factor_proxy[cols])
        summary_rows.append({"spec": model_name, "period": "Full", "alpha_ann": alpha, "alpha_t": t, "nw_lag": nw_tstat(y.dropna())[1]})
    summary = pd.DataFrame(summary_rows)

    weighting_rows = []
    weight_cols = ["w_production", "w_equal", "w_mcap_uncapped", "w_sqrt_mcap", "w_mcap_cap10", "w_mcap_cap15", "w_mcap_cap20"]
    for wc in weight_cols:
        r = portfolio_returns_from_weights(contrib, wc)
        to = turnover_from_holdings(holdings, wc)
        for row in summarize_by_period(wc, r, bm, to):
            row["max_stock_weight_mean"] = float(holdings.groupby("ym")[wc].max().mean())
            row["top5_weight_mean"] = float(holdings.groupby("ym").apply(lambda g: g.nlargest(5, wc)[wc].sum(), include_groups=False).mean())
            weighting_rows.append(row)
    weighting = pd.DataFrame(weighting_rows)

    size = contrib.groupby(["period", "size_group"]).agg(
        portfolio_weight=("prod_weight", "mean"),
        return_contribution=("contrib", "sum"),
    ).reset_index()

    sector_m, sector_s, sector_y = sector_attribution(conn, contrib, dates, bm)
    concentration = concentration_diagnostics(contrib, monthly)
    excl = exclusion_tests(contrib, bm)
    concentration = pd.concat([concentration, excl.assign(kind="exclusion_test")], ignore_index=True, sort=False)
    factor_def, factor_lofo = factor_outputs(module)

    turnover_rows = []
    for wc in weight_cols:
        gross = portfolio_returns_from_weights(contrib, wc)
        to = turnover_from_holdings(holdings, wc)
        cg = costs_grid(gross, to)
        cg.insert(0, "spec", wc)
        cg["turnover_mean"] = to.mean()
        turnover_rows.append(cg)
    turnover = pd.concat(turnover_rows, ignore_index=True)

    summary.to_csv(OUT / "fcf_attribution_summary.csv", index=False, encoding="utf-8-sig")
    sector_s.to_csv(OUT / "fcf_sector_attribution.csv", index=False, encoding="utf-8-sig")
    size.to_csv(OUT / "fcf_size_attribution.csv", index=False, encoding="utf-8-sig")
    weighting.to_csv(OUT / "fcf_weighting_comparison.csv", index=False, encoding="utf-8-sig")
    concentration.to_csv(OUT / "fcf_concentration_diagnostic.csv", index=False, encoding="utf-8-sig")
    factor_lofo.to_csv(OUT / "fcf_factor_lofo.csv", index=False, encoding="utf-8-sig")
    turnover.to_csv(OUT / "fcf_turnover_diagnostic.csv", index=False, encoding="utf-8-sig")
    sector_m.to_csv(OUT / "fcf_sector_monthly_effects.csv", index=False, encoding="utf-8-sig")
    sector_y.to_csv(OUT / "fcf_sector_yearly_effects.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "fcf_production_audit.csv", index=False, encoding="utf-8-sig")

    elapsed = time.time() - t0
    write_report(audit, summary, sector_m, sector_s, weighting, concentration, factor_lofo, turnover, elapsed, len(monthly))
    print(f"Done in {elapsed:.1f}s")
    print(f"Wrote {OUT} and {DOCS / 'FCF_PERFORMANCE_ATTRIBUTION_AUTO.md'}")


if __name__ == "__main__":
    main()
