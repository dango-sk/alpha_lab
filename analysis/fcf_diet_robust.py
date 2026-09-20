"""
analysis/fcf_diet_robust.py   (실험 스크립트, production 미수정)

[post-OOS diagnostic] BASE vs DIET-4 강건성 검증.
  DIET-4 개선이 '구조적 팩터 개선'인지 '최근 주도 대형주(반도체) 노출 확대'인지 구분한다.

  - 새 팩터 조합 탐색 없음 / 가중치 조정 없음 / production 코드 수정 없음
  - production 함수(run_backtest, score_stocks_from_strategy, _apply_mcap_cap)만 재사용
  - 기간: Full(100M) / IS(<=2024-06) / OOS_post-OOS diagnostic(>=2024-07)

실행: .venv/bin/python analysis/fcf_diet_robust.py
산출: output/robust_*.csv, output/robust_rolling_diff.png, docs/FCF_DIET4_ROBUSTNESS.md
"""
import sys, time, math, json
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
load_dotenv(REPO / ".env"); sys.path.insert(0, str(REPO / "analysis"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

from lib.data import load_strategy                                        # noqa: E402
from lib.factor_engine import code_to_module, score_stocks_from_strategy  # noqa: E402
from step7_backtest import get_db, _apply_mcap_cap                        # noqa: E402
import fcf_diet_compare as dc                                             # noqa: E402

OUT = REPO / "output"; DOCS = REPO / "docs"
OUT.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
IS_END, OOS_START, OOS = dc.IS_END, dc.OOS_START, dc.OOS_LABEL
BM = dc.BM
COST_BP = 30

# 사용자 지정 DIET-4 고정 가중치 (drop_factors 결과와 일치해야 함 → assert)
DIET4_W = {"T_PER": 0.0625, "F_PER": 0.0625, "T_EVEBITDA": 0.0, "F_EVEBITDA": 0.0625,
           "T_PBR": 0.0, "F_PBR": 0.0625, "ATT_PBR": 0.0625, "ATT_EVIC": 0.0,
           "ATT_PER": 0.1250, "ATT_EVEBIT": 0.1250, "T_SPSG": 0.0625,
           "F_EPS_M": 0.1875, "PRICE_MA_REV": 0.0, "FCF_YIELD": 0.1875}

LEADERS = {"000660": "SK하이닉스", "005930": "삼성전자", "402340": "SK스퀘어",
           "000270": "기아", "012330": "현대모비스"}
LEADER_SETS = [("SK하이닉스", ["000660"]),
               ("SK하이닉스+삼성전자", ["000660", "005930"]),
               ("SK하이닉스+삼성전자+SK스퀘어", ["000660", "005930", "402340"]),
               ("주도 5종목 전체", list(LEADERS))]
SEMI_KEYS = ("전기전자", "전기·전자", "반도체", "IT", "전기,전자")
FOCUS_FACTORS = ["F_EPS_M", "FCF_YIELD", "ATT_EVEBIT", "PRICE_MA_REV", "T_EVEBITDA"]


# ── 유틸 ────────────────────────────────────────────────────
def mdd(r):
    r = pd.Series(r).dropna()
    if r.empty: return np.nan
    c = (1 + r).cumprod()
    return float((c / c.cummax() - 1).min())


def periods(idx):
    return {"Full": list(idx), "IS": [m for m in idx if m <= IS_END],
            OOS: [m for m in idx if m >= OOS_START]}


def sector_map(conn, codes):
    out = {}
    codes = list(codes)
    for i in range(0, len(codes), 900):
        ch = codes[i:i + 900]; ph = ",".join(["?"] * len(ch))
        rows = conn.execute(f"""
            SELECT DISTINCT ON (stock_code) stock_code, sec_cd_nm
            FROM fnspace_master WHERE stock_code IN ({ph})
            ORDER BY stock_code, snapshot_date DESC
        """, tuple("A" + c for c in ch)).fetchall()
        out.update({str(c)[1:]: (s or "Unknown") for c, s in rows})
    return {c: out.get(c, "Unknown") for c in codes}


def is_semi(sec):
    return any(k in (sec or "") for k in SEMI_KEYS)


def port_from_stocks(sc, drop=(), recap=False, cap=0.30):
    """종목단위 df(sc: ym,code,weight,ret,market_cap) → 월별 gross 수익률·턴오버.
       drop: 제외 종목. recap=True면 잔여 종목을 시총가중+cap 재적용(합 100%)."""
    rets, w_by = {}, {}
    for ym, g in sc.groupby("ym"):
        g = g[~g["code"].isin(drop)]
        if g.empty:
            rets[ym] = 0.0; w_by[ym] = {}; continue
        if recap:
            raw = [max(float(v or 0), 0.0) for v in g["market_cap"]]
            w = _apply_mcap_cap(raw, cap=cap) if sum(raw) > 0 else [1 / len(g)] * len(g)
        else:
            w = list(g["weight"])
        r = np.array([np.nan if pd.isna(x) else float(x) for x in g["ret"]])
        w = np.array(w, float)
        ok = ~np.isnan(r)
        rets[ym] = float((w[ok] * r[ok]).sum())     # 결측 종목은 암묵적 현금 (production과 동일)
        w_by[ym] = dict(zip(g["code"], w))
    s = pd.Series(rets).sort_index()
    yms = list(s.index); to = {}
    for i, ym in enumerate(yms):
        cur = w_by[ym]
        prev = w_by[yms[i - 1]] if i else {}
        to[ym] = 1.0 if i == 0 else 0.5 * sum(abs(cur.get(k, 0) - prev.get(k, 0)) for k in set(cur) | set(prev))
    return s, pd.Series(to).sort_index()


def net(gross, turnover, bp=COST_BP):
    return gross - turnover * (bp / 10000.0) * 2


def summ(r):
    r = pd.Series(r).dropna()
    p = dc.perf(r)
    return {"months": p.get("months", 0), "cagr": p.get("cagr", np.nan),
            "sharpe": p.get("sharpe", np.nan), "mdd": p.get("mdd", np.nan),
            "monthly_mean": p.get("monthly_mean", np.nan)}


# ── 메인 ────────────────────────────────────────────────────
def main():
    t0 = time.time()
    sd = load_strategy(dc.STRATEGY, "monthly", "KOSPI")
    base_mod = code_to_module(sd["code"])
    c4, w4 = dc.drop_factors(sd["code"], dc.DROP4)
    diet_mod = code_to_module(c4)
    # 항상 0인 미사용 팩터(T_PCF/F_SPSG/OBV_SLOPE/MFI)는 비교에서 제외 → 활성 가중치만 대조
    got = {k: round(v, 6) for k, v in getattr(diet_mod, "WEIGHTS_LARGE", {}).items() if v > 0}
    exp = {k: round(v, 6) for k, v in DIET4_W.items() if v > 0}
    assert got == exp, f"DIET-4 가중치 불일치\n계산={got}\n지정={exp}"
    zero_ok = [k for k, v in DIET4_W.items() if v == 0
               and getattr(diet_mod, "WEIGHTS_LARGE", {}).get(k, 0.0) != 0.0]
    assert not zero_ok, f"0이어야 할 팩터가 0이 아님: {zero_ok}"
    print(f"✅ DIET-4 가중치 = 사용자 지정값과 일치 ({time.time()-t0:.0f}s)", flush=True)

    res = {}
    for name, mod in [("BASE", base_mod), ("DIET-4", diet_mod)]:
        r = dc.run(mod, name, 30)                      # tx=0 → 슬리피지만 반영
        ret, hold, to, slip, _ = dc.unpack(r) + (r,)
        res[name] = dict(ret=ret, hold=hold, to=to, slip=slip)
        print(f"  {name} 백테스트 완료 ({time.time()-t0:.0f}s)", flush=True)

    conn = get_db()
    dates = sorted(res["BASE"]["hold"]["date"].unique())
    net30 = {k: net(v["ret"], v["to"]) for k, v in res.items()}
    idx = net30["BASE"].dropna().index.intersection(net30["DIET-4"].dropna().index)
    for k in net30: net30[k] = net30[k].reindex(idx)

    # 벤치마크
    bm = pd.Series({d0[:7]: (dc.price_map(conn, [BM], d1, "last").get(BM) /
                             dc.price_map(conn, [BM], d0, "first").get(BM) - 1.0)
                    for d0, d1 in zip(dates[:-1], dates[1:])}).sort_index().reindex(idx)

    # 종목 단위 수익·기여
    sc = {k: dc.stock_returns(conn, v["hold"], dates) for k, v in res.items()}
    all_codes = sorted(set(sc["BASE"]["code"]) | set(sc["DIET-4"]["code"]))
    nm = dc.name_map(conn, all_codes); sec = sector_map(conn, all_codes)
    for k in sc:
        sc[k]["name"] = sc[k]["code"].map(nm); sc[k]["sector"] = sc[k]["code"].map(sec)
        sc[k]["is_semi"] = sc[k]["sector"].map(is_semi)
    print(f"  종목 수익·업종 매핑 완료 ({time.time()-t0:.0f}s)", flush=True)

    # 재구성 검증 (leave-out 계산의 기준선이 production과 일치하는지)
    recon = {}
    for k in sc:
        g, t = port_from_stocks(sc[k])
        recon[k] = net(g, t).reindex(idx)
    recon_mae = {k: float((recon[k] - net30[k]).abs().mean()) for k in sc}
    print("  재구성 MAE(bp): " + ", ".join(f"{k} {v*10000:.2f}" for k, v in recon_mae.items()))

    # ── 2. 월별 상대성과 ──
    D = (net30["DIET-4"] - net30["BASE"]).dropna()
    rel_cum = (1 + net30["DIET-4"]).cumprod() / (1 + net30["BASE"]).cumprod()
    rows = []
    for p, ix in periods(idx).items():
        d = D.reindex(ix).dropna()
        t, lag = dc.nw_t(d)
        rc = rel_cum.reindex(ix).dropna()
        rows.append({"period": p, "months": len(d), "mean_diff": d.mean(),
                     "ann_diff": (1 + d.mean()) ** 12 - 1, "nw_t": t, "nw_lag": lag,
                     "win_rate": float((d > 0).mean()),
                     "win_gt_1pp": float((d >= 0.01).mean()), "lose_lt_1pp": float((d <= -0.01).mean()),
                     "mean_pos": float(d[d > 0].mean()), "mean_neg": float(d[d < 0].mean()),
                     "worst_month": float(d.min()), "worst_ym": str(d.idxmin()),
                     "max_rel_dd": float((rc / rc.cummax() - 1).min())})
    monthly = pd.DataFrame({"benchmark": bm, "BASE_net30": net30["BASE"],
                            "DIET4_net30": net30["DIET-4"], "diff": D,
                            "rel_cum": rel_cum})
    monthly.index.name = "ym"
    monthly.to_csv(OUT / "robust_monthly_diff.csv", encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "robust_relative_summary.csv", index=False, encoding="utf-8-sig")

    # ── 3. 소수 월 의존도 ──
    top_rows = []
    for kind, ser in [("top10", D.nlargest(10)), ("bottom10", D.nsmallest(10))]:
        for ym, dv in ser.items():
            cb = sc["BASE"][sc["BASE"]["ym"] == ym].set_index("code")
            cd = sc["DIET-4"][sc["DIET-4"]["ym"] == ym].set_index("code")
            codes = set(cb.index) | set(cd.index)
            dif = pd.Series({c: float(cd["contrib"].get(c, 0.0) or 0) - float(cb["contrib"].get(c, 0.0) or 0)
                             for c in codes}).sort_values(key=abs, ascending=False)
            for c in dif.head(3).index:
                top_rows.append({"kind": kind, "ym": ym, "diff": dv,
                                 "BASE_ret": net30["BASE"].get(ym), "DIET4_ret": net30["DIET-4"].get(ym),
                                 "code": c, "name": nm.get(c, ""), "sector": sec.get(c, ""),
                                 "contrib_diff": float(dif[c]),
                                 "BASE_weight": float(cb["weight"].get(c, 0.0) or 0),
                                 "DIET4_weight": float(cd["weight"].get(c, 0.0) or 0),
                                 "market_ret": float(bm.get(ym, np.nan))})
    pd.DataFrame(top_rows).to_csv(OUT / "robust_top_months.csv", index=False, encoding="utf-8-sig")

    ex_rows = []
    for n in [0, 1, 3, 5, 10]:
        drop_ym = set(D.nlargest(n).index) if n else set()
        keep = [m for m in idx if m not in drop_ym]
        b, d = net30["BASE"].reindex(keep), net30["DIET-4"].reindex(keep)
        ex_rows.append({"exclude_top_n": n, "months": len(keep),
                        "BASE_cagr": summ(b)["cagr"], "DIET4_cagr": summ(d)["cagr"],
                        "cagr_diff": summ(d)["cagr"] - summ(b)["cagr"],
                        "BASE_sharpe": summ(b)["sharpe"], "DIET4_sharpe": summ(d)["sharpe"],
                        "BASE_mdd": summ(b)["mdd"], "DIET4_mdd": summ(d)["mdd"],
                        "mean_diff": float((d - b).mean()),
                        "diet4_still_better": bool(summ(d)["cagr"] > summ(b)["cagr"]),
                        "excluded_months": ",".join(sorted(drop_ym))})
    pd.DataFrame(ex_rows).to_csv(OUT / "robust_top_month_exclusion.csv", index=False, encoding="utf-8-sig")

    # ── 4. Rolling ──
    roll_rows = []
    for w in [12, 24, 36]:
        for i in range(len(idx) - w + 1):
            ix = list(idx)[i:i + w]
            b, d = net30["BASE"].reindex(ix), net30["DIET-4"].reindex(ix)
            sb, sdd = summ(b), summ(d)
            roll_rows.append({"window": w, "start": ix[0], "end": ix[-1],
                              "BASE_cagr": sb["cagr"], "DIET4_cagr": sdd["cagr"],
                              "BASE_sharpe": sb["sharpe"], "DIET4_sharpe": sdd["sharpe"],
                              "BASE_mdd": sb["mdd"], "DIET4_mdd": sdd["mdd"],
                              "cum_diff": float((1 + d).prod() - (1 + b).prod())})
    roll = pd.DataFrame(roll_rows)
    roll.to_csv(OUT / "robust_rolling.csv", index=False, encoding="utf-8-sig")
    roll_sum = roll.assign(
        cagr_win=roll.DIET4_cagr > roll.BASE_cagr,
        sharpe_win=roll.DIET4_sharpe > roll.BASE_sharpe,
        mdd_win=roll.DIET4_mdd > roll.BASE_mdd,          # mdd는 음수 → 큰 쪽이 얕음
    ).assign(both=lambda x: x.cagr_win & x.sharpe_win).groupby("window")[
        ["cagr_win", "sharpe_win", "mdd_win", "both"]].mean().reset_index()
    roll_sum.to_csv(OUT / "robust_rolling_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    for a, w in zip(ax, [12, 24, 36]):
        s = roll[roll.window == w]
        a.axhline(0, color="gray", lw=0.8)
        a.plot(pd.to_datetime(s["end"]), s["cum_diff"] * 100, lw=1.4)
        a.axvline(pd.Timestamp(OOS_START), color="red", ls="--", lw=1, label="OOS start 2024-07")
        a.set_title(f"Rolling {w}M cumulative return diff (DIET-4 - BASE, %p)")
        a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "robust_rolling_diff.png", dpi=130); plt.close(fig)

    # ── 5. 종목 leave-out ──
    lo_rows = []
    def add_lo(method, label, drop):
        for k in ["BASE", "DIET-4"]:
            g, t = port_from_stocks(sc[k], drop=drop, recap=(method == "recap"))
            r = net(g, t).reindex(idx)
            for p, ix in periods(idx).items():
                s = summ(r.reindex(ix))
                lo_rows.append({"method": method, "exclude": label, "spec": k, "period": p, **s})
    add_lo("contrib_subtract", "없음(기준)", [])
    add_lo("recap", "없음(기준)", [])
    for label, codes in LEADER_SETS:
        for k in ["BASE", "DIET-4"]:
            # 5-1 기여도 차감 (비중 재조정 없음)
            sub = sc[k][sc[k]["code"].isin(codes)].groupby("ym")["contrib"].sum().reindex(idx).fillna(0)
            r = (net30[k] - sub).reindex(idx)
            for p, ix in periods(idx).items():
                lo_rows.append({"method": "contrib_subtract", "exclude": label, "spec": k,
                                "period": p, **summ(r.reindex(ix))})
        add_lo("recap", label, codes)
    lo = pd.DataFrame(lo_rows)
    piv = lo.pivot_table(index=["method", "exclude", "period"], columns="spec",
                         values=["cagr", "sharpe", "mdd", "monthly_mean"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["cagr_diff"] = piv["cagr_DIET-4"] - piv["cagr_BASE"]
    piv["mean_diff"] = piv["monthly_mean_DIET-4"] - piv["monthly_mean_BASE"]
    base_imp = {(m, p): v for (m, p), v in
                piv[piv["exclude"] == "없음(기준)"].set_index(["method", "period"])["mean_diff"].items()}
    piv["dependency"] = [1 - (r["mean_diff"] / base_imp[(r["method"], r["period"])])
                         if base_imp.get((r["method"], r["period"])) else np.nan
                         for _, r in piv.iterrows()]
    piv["diet4_still_better"] = piv["cagr_diff"] > 0
    piv.to_csv(OUT / "robust_stock_leaveout.csv", index=False, encoding="utf-8-sig")

    # ── 6. 업종(전기전자/반도체) ──
    semi_codes = sorted({c for k in sc for c in sc[k].loc[sc[k]["is_semi"], "code"].unique()})
    pd.DataFrame([{"code": c, "name": nm.get(c, ""), "sector": sec.get(c, ""),
                   "BASE_months": int((sc["BASE"]["code"] == c).sum()),
                   "DIET4_months": int((sc["DIET-4"]["code"] == c).sum()),
                   "BASE_mean_w": float(sc["BASE"].loc[sc["BASE"].code == c, "weight"].mean() or 0),
                   "DIET4_mean_w": float(sc["DIET-4"].loc[sc["DIET-4"].code == c, "weight"].mean() or 0),
                   "BASE_contrib": float(sc["BASE"].loc[sc["BASE"].code == c, "contrib"].sum()),
                   "DIET4_contrib": float(sc["DIET-4"].loc[sc["DIET-4"].code == c, "contrib"].sum())}
                  for c in semi_codes]).to_csv(OUT / "robust_semi_universe.csv", index=False, encoding="utf-8-sig")

    sec_rows = []
    for k in ["BASE", "DIET-4"]:
        sub = sc[k][sc[k]["is_semi"]].groupby("ym")["contrib"].sum().reindex(idx).fillna(0)
        r_sub = (net30[k] - sub).reindex(idx)
        g, t = port_from_stocks(sc[k], drop=semi_codes, recap=True)
        r_cap = net(g, t).reindex(idx)
        for p, ix in periods(idx).items():
            sec_rows.append({"method": "contrib_subtract", "spec": k, "period": p, **summ(r_sub.reindex(ix))})
            sec_rows.append({"method": "recap", "spec": k, "period": p, **summ(r_cap.reindex(ix))})
            sec_rows.append({"method": "원본", "spec": k, "period": p, **summ(net30[k].reindex(ix))})
            sec_rows.append({"method": "semi_weight", "spec": k, "period": p,
                             "months": len(ix),
                             "cagr": float(sc[k][sc[k].ym.isin(ix) & sc[k].is_semi].groupby("ym")["weight"].sum().mean()),
                             "sharpe": np.nan, "mdd": np.nan, "monthly_mean": np.nan})
    pd.DataFrame(sec_rows).to_csv(OUT / "robust_sector_leaveout.csv", index=False, encoding="utf-8-sig")

    # ── 7. 시장 상태 ──
    # 대형/중형: KOSPI 유니버스 시총 상위 100 vs 101~300 시총가중 (프로젝트에 중형 벤치마크 ETF 없음 → 정의 명시)
    from step7_backtest import get_universe_stocks
    lm_rows = {}
    sec_all = dict(sec)                      # 유니버스 업종 캐시 (배치 조회)
    for d0, d1 in zip(dates[:-1], dates[1:]):
        uni = get_universe_stocks(conn, d0, "monthly")
        if not uni: continue
        miss = [c for c in uni if c not in sec_all]
        if miss: sec_all.update(sector_map(conn, miss))
        ph = ",".join(["?"] * len(uni))
        mc = dict(conn.execute(f"""
            SELECT dp.stock_code, dp.market_cap FROM daily_price dp
            JOIN (SELECT stock_code, MIN(trade_date) d FROM daily_price
                  WHERE stock_code IN ({ph}) AND trade_date >= ? GROUP BY stock_code) t
              ON dp.stock_code=t.stock_code AND dp.trade_date=t.d
        """, (*uni, d0)).fetchall())
        p0 = dc.price_map(conn, uni, d0, "first"); p1 = dc.price_map(conn, uni, d1, "last")
        rows = [(c, float(mc.get(c) or 0), p1[c] / p0[c] - 1)
                for c in uni if p0.get(c) and p1.get(c) and (mc.get(c) or 0) > 0]
        rows.sort(key=lambda x: -x[1])
        def cw(sub):
            tot = sum(m for _, m, _ in sub)
            return sum(m * r for _, m, r in sub) / tot if tot else np.nan
        semi_sub = [x for x in rows if is_semi(sec_all.get(x[0]))]
        lm_rows[d0[:7]] = {"large": cw(rows[:100]), "mid": cw(rows[100:300]),
                           "semi": cw(semi_sub), "all": cw(rows)}
    lm = pd.DataFrame(lm_rows).T.sort_index().reindex(idx)
    states = pd.DataFrame({
        "market": np.where(bm >= 0, "UP", "DOWN"),
        "size": np.where(lm["large"] >= lm["mid"], "LARGE-LEAD", "MID-LEAD"),
        "semi": np.where(lm["semi"] >= bm, "SEMI-STRONG", "SEMI-WEAK"),
    }, index=idx)
    st_rows = []
    for col in ["market", "size", "semi"]:
        for st, g in states.groupby(col):
            ix = g.index
            b, d = net30["BASE"].reindex(ix), net30["DIET-4"].reindex(ix)
            dd = (d - b).dropna()
            t, _ = dc.nw_t(dd)
            tt = float(dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd)))) if len(dd) > 2 and dd.std(ddof=1) > 0 else np.nan
            st_rows.append({"axis": col, "state": st, "months": len(ix),
                            "BASE_mean": float(b.mean()), "DIET4_mean": float(d.mean()),
                            "diff": float(dd.mean()), "nw_t": t, "t_iid": tt,
                            "win_rate": float((dd > 0).mean()),
                            "BASE_mdd": mdd(b), "DIET4_mdd": mdd(d)})
    pd.DataFrame(st_rows).to_csv(OUT / "robust_market_states.csv", index=False, encoding="utf-8-sig")
    states.assign(**lm).to_csv(OUT / "robust_state_labels.csv", encoding="utf-8-sig")

    # ── 8. 편입 종목 특성 ──
    SCORE_MAP = getattr(base_mod, "SCORE_MAP", {})
    fs_rows = []
    for d in dates:
        try:
            _, df = score_stocks_from_strategy(conn, d, base_mod, return_df=True)
        except Exception as e:                       # noqa: BLE001
            print(f"  ⚠ score_df {d} 실패: {e}"); continue
        if df is None or df.empty: continue
        df = df.copy(); df["code"] = df["stock_code"].str.lstrip("A")
        cols = {f: SCORE_MAP[f] for f in FOCUS_FACTORS if f in SCORE_MAP and SCORE_MAP[f] in df.columns}
        for _, r in df.iterrows():
            fs_rows.append({"ym": d[:7], "code": r["code"],
                            **{f: float(r[c]) if pd.notna(r[c]) else np.nan for f, c in cols.items()}})
    fscore = pd.DataFrame(fs_rows)
    fscore.to_csv(OUT / "robust_factor_scores.csv", index=False, encoding="utf-8-sig")
    print(f"  팩터 점수 추출 완료 ({time.time()-t0:.0f}s)", flush=True)

    b_idx = sc["BASE"].set_index(["ym", "code"]); d_idx = sc["DIET-4"].set_index(["ym", "code"])
    grp_rows = []
    for ym in idx:
        bset = set(sc["BASE"].loc[sc["BASE"].ym == ym, "code"])
        dset = set(sc["DIET-4"].loc[sc["DIET-4"].ym == ym, "code"])
        for c in dset - bset: grp_rows.append({"ym": ym, "code": c, "group": "DIET4_only"})
        for c in bset - dset: grp_rows.append({"ym": ym, "code": c, "group": "BASE_only"})
        for c in bset & dset:
            wb = float(b_idx.loc[(ym, c), "weight"]); wd = float(d_idx.loc[(ym, c), "weight"])
            if abs(wd - wb) >= 0.02:
                grp_rows.append({"ym": ym, "code": c, "group": "common_weight_shift",
                                 "weight_shift": wd - wb})
    gdf = pd.DataFrame(grp_rows)
    gdf = gdf.merge(fscore, on=["ym", "code"], how="left")
    src = pd.concat([sc["BASE"], sc["DIET-4"]]).drop_duplicates(["ym", "code"])[
        ["ym", "code", "ret", "market_cap", "name", "sector", "is_semi"]]
    gdf = gdf.merge(src, on=["ym", "code"], how="left")
    gdf.to_csv(OUT / "robust_selection_diff.csv", index=False, encoding="utf-8-sig")

    ch_rows = []
    for p, ix in periods(idx).items():
        sub = gdf[gdf.ym.isin(ix)]
        for g, gg in sub.groupby("group"):
            ch_rows.append({"period": p, "group": g, "n": len(gg),
                            **{f: float(gg[f].mean()) if f in gg else np.nan for f in FOCUS_FACTORS},
                            "mean_next_ret": float(gg["ret"].mean()),
                            "median_mcap_bn": float(gg["market_cap"].median() / 1e9),
                            "semi_share": float(gg["is_semi"].mean())})
    chdf = pd.DataFrame(ch_rows)
    # 월별 paired difference (DIET4_only 평균 - BASE_only 평균)
    pair_rows = []
    for f in FOCUS_FACTORS + ["ret"]:
        m = gdf[gdf.group.isin(["DIET4_only", "BASE_only"])].groupby(["ym", "group"])[f].mean().unstack()
        if {"DIET4_only", "BASE_only"} - set(m.columns): continue
        dd = (m["DIET4_only"] - m["BASE_only"]).dropna()
        t, _ = dc.nw_t(dd)
        pair_rows.append({"metric": f, "n_months": len(dd), "paired_mean_diff": float(dd.mean()),
                          "nw_t": t, "share_positive": float((dd > 0).mean())})
    chdf.to_csv(OUT / "robust_selection_characteristics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pair_rows).to_csv(OUT / "robust_selection_paired.csv", index=False, encoding="utf-8-sig")

    # ── 9. 거래비용·반복매매 ──
    cost_rows = []
    for k in ["BASE", "DIET-4"]:
        for bp in [0, 30, 50]:
            r = net(res[k]["ret"], res[k]["to"], bp).reindex(idx)
            for p, ix in periods(idx).items():
                cost_rows.append({"spec": k, "cost_bp": bp, "period": p, **summ(r.reindex(ix)),
                                  "turnover_mean": float(res[k]["to"].reindex(ix).mean())})
    cdf = pd.DataFrame(cost_rows)
    # 경계 반복매매 (rank 25~35, 3개월 내 재진입)
    churn_rows = []
    for k, mod in [("BASE", base_mod), ("DIET-4", diet_mod)]:
        held = []
        for d in dates:
            uni = set(get_universe_stocks(conn, d, "monthly"))
            if not uni: continue
            cands = [(c, s) for c, s in score_stocks_from_strategy(conn, d, mod) if c in uni]
            rk = {c: i + 1 for i, (c, _) in enumerate(cands)}
            held.append((d[:7], {c for c, _ in cands[:30]}, rk))
        for i, (ym, cur, rk) in enumerate(held):
            exits = held[i - 1][1] - cur if i else set()
            fut = set().union(*[held[j][1] for j in range(i + 1, min(i + 4, len(held)))]) if i + 1 < len(held) else set()
            churn_rows.append({"spec": k, "ym": ym,
                               "n_boundary_25_35": sum(1 for c in cur if 25 <= rk.get(c, 999) <= 35),
                               "n_exit": len(exits),
                               "n_reentry_3m": len(exits & fut),
                               "reentry_rate_3m": len(exits & fut) / len(exits) if exits else np.nan})
    churn = pd.DataFrame(churn_rows)
    churn.to_csv(OUT / "robust_churn.csv", index=False, encoding="utf-8-sig")
    ch_sum = churn.assign(period=lambda x: np.where(x.ym <= IS_END, "IS", OOS)).groupby(["spec"])[
        ["n_boundary_25_35", "n_exit", "n_reentry_3m", "reentry_rate_3m"]].mean().reset_index()
    # 비용절감 기여분: (DIET4 turnover 감소)×30bp×2 vs 총 개선분
    to_b = res["BASE"]["to"].reindex(idx); to_d = res["DIET-4"]["to"].reindex(idx)
    cost_save = ((to_b - to_d) * (COST_BP / 10000) * 2)
    cost_share = {p: float(cost_save.reindex(ix).mean() / D.reindex(ix).mean())
                  if D.reindex(ix).mean() else np.nan for p, ix in periods(idx).items()}
    cdf["_"] = ""
    cdf.to_csv(OUT / "robust_cost_turnover.csv", index=False, encoding="utf-8-sig")
    ch_sum.to_csv(OUT / "robust_churn_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"period": p, "cost_saving_share_of_improvement": v,
                   "mean_cost_saving": float(cost_save.reindex(periods(idx)[p]).mean()),
                   "mean_improvement": float(D.reindex(periods(idx)[p]).mean())}
                  for p, v in cost_share.items()]).to_csv(
        OUT / "robust_cost_attribution.csv", index=False, encoding="utf-8-sig")

    # 집중도
    conc = []
    for k in ["BASE", "DIET-4"]:
        h = res[k]["hold"]
        for p, ix in periods(idx).items():
            hh = h[h.ym.isin(ix)]
            conc.append({"spec": k, "period": p,
                         "max_weight": float(hh.groupby("ym")["weight"].max().mean()),
                         "top3": float(hh.groupby("ym")["weight"].apply(lambda s: s.nlargest(3).sum()).mean()),
                         "top5": float(hh.groupby("ym")["weight"].apply(lambda s: s.nlargest(5).sum()).mean()),
                         "hhi": float(hh.groupby("ym")["weight"].apply(lambda s: (s ** 2).sum()).mean())})
    pd.DataFrame(conc).to_csv(OUT / "robust_concentration.csv", index=False, encoding="utf-8-sig")

    meta = {"elapsed_sec": round(time.time() - t0, 1), "months": len(idx),
            "recon_mae_bp": {k: round(v * 10000, 2) for k, v in recon_mae.items()},
            "cost_saving_share": cost_share,
            "rolling_summary": roll_sum.to_dict("records"),
            "semi_n_codes": len(semi_codes)}
    (OUT / "robust_run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n✅ 완료 ({time.time()-t0:.0f}s) → output/robust_*.csv, robust_rolling_diff.png")


if __name__ == "__main__":
    main()
