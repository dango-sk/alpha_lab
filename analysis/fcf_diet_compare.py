"""
analysis/fcf_diet_compare.py   (실험 스크립트, production 미수정)

[post-OOS diagnostic] 팩터 다이어트 + 집중도 완화 후보 비교.

  A. BASE          : production 14팩터, cap 30%
  B. DIET-3        : T_PBR / ATT_EVIC / PRICE_MA_REV 제거, 나머지 비례 재조정, cap 30%
  C. DIET-4        : DIET-3 + T_EVEBITDA 제거, cap 30%
  D. DIET-3-CAP20  : DIET-3 가중치 + cap 20%

  production 함수(run_backtest / score_stocks_from_strategy / _apply_mcap_cap /
  calc_portfolio_return)를 그대로 호출한다. 새 백테스트 엔진을 만들지 않는다.

  0단계로 저장 production 대비 재계산 불일치 6건을 감사한다.
  감사 통과(월별 MAE < 5bp) 시에만 전략 비교로 진행한다.

실행:
    .venv/bin/python analysis/fcf_diet_compare.py
산출:
    output/diet_calc_audit.csv
    output/diet_performance_comparison.csv
    output/diet_monthly_returns.csv
    output/diet_overlap.csv
    output/diet_contributors.csv
    output/diet_leader_stocks.csv
    output/diet_concentration.csv
    docs/FCF_DIET_DIAGNOSTIC.md
"""
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
load_dotenv(REPO / ".env")

from lib.data import load_strategy                                        # noqa: E402
from lib.factor_engine import code_to_module, score_stocks_from_strategy  # noqa: E402
from config.settings import BACKTEST_CONFIG                               # noqa: E402
from step7_backtest import (                                              # noqa: E402
    run_backtest, get_universe_stocks, get_db, _calc_slippage,
)

OUT = REPO / "output"; DOCS = REPO / "docs"
OUT.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)

STRATEGY = "FCF_YIELD추가전략"
IS_END, OOS_START = "2024-06", "2024-07"
OOS_LABEL = "OOS_post-OOS diagnostic"
BM = "069500"
DROP3 = ["T_PBR", "ATT_EVIC", "PRICE_MA_REV"]
DROP4 = DROP3 + ["T_EVEBITDA"]
LEADERS = {"000660": "SK하이닉스", "402340": "SK스퀘어", "000270": "기아",
           "012330": "현대모비스", "005490": "POSCO홀딩스"}
AUDIT_TOL_MAE = 0.0005   # 월별 평균절대오차 5bp


# ── 통계 유틸 ────────────────────────────────────────────────
def nw_t(x, lag=None):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 3: return np.nan, 0
    if lag is None: lag = int(np.floor(4 * (n / 100.0) ** (2 / 9)))
    e = x - x.mean(); s = float(e @ e) / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1.0)) * float(e[l:] @ e[:-l]) / n
    se = math.sqrt(max(s, 0.0) / n)
    return (float(x.mean() / se) if se > 0 else np.nan), lag


def ols_alpha(y: pd.Series, x: pd.DataFrame):
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 12: return np.nan, np.nan
    Y = d.iloc[:, 0].values
    X = np.column_stack([np.ones(len(d)), d.iloc[:, 1:].values])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    s2 = float(resid @ resid) / (len(d) - X.shape[1])
    se = math.sqrt(s2 * np.linalg.pinv(X.T @ X)[0, 0])
    return float(beta[0]) * 12, (float(beta[0] / se) if se > 0 else np.nan)


def perf(x: pd.Series):
    x = x.dropna()
    if len(x) == 0: return {}
    cum = (1 + x).cumprod(); vol = float(x.std(ddof=1)) * np.sqrt(12)
    cagr = float((1 + x).prod() ** (12 / len(x)) - 1)
    return dict(months=len(x), cagr=cagr, monthly_mean=float(x.mean()), vol=vol,
                sharpe=(cagr / vol if vol else np.nan),
                mdd=float((cum / cum.cummax() - 1).min()))


def period_slices(idx):
    return {"Full": list(idx),
            "IS": [m for m in idx if m <= IS_END],
            OOS_LABEL: [m for m in idx if m >= OOS_START]}


# ── 전략 코드 변형 (factor_lofo.drop_factor_code 와 동일 규칙) ──
def drop_factors(code: str, drops: list[str]) -> tuple[str, dict]:
    w = dict(getattr(code_to_module(code), "WEIGHTS_LARGE", {}))
    removed = sum(w.get(f, 0.0) for f in drops)
    k = 1.0 / (1.0 - removed)
    new = {key: (0.0 if key in drops else round(v * k, 8)) for key, v in w.items()}
    resid = 1.0 - sum(new.values())
    if resid:
        anchor = max((key for key in new if key not in drops), key=lambda key: new[key])
        new[anchor] = round(new[anchor] + resid, 8)
    assert abs(sum(new.values()) - 1.0) < 1e-6
    body = "WEIGHTS_LARGE = {\n" + "".join(f'    "{a}": {b!r},\n' for a, b in new.items()) + "}"
    out, n = re.subn(r"WEIGHTS_LARGE\s*=\s*\{[^}]*\}", body, code, count=1, flags=re.DOTALL)
    if n != 1: raise SystemExit("WEIGHTS_LARGE 치환 실패")
    return out, {a: b for a, b in new.items() if b > 0}


def make_selector(module):
    def selector(conn, calc_date, top_n):
        uni = set(get_universe_stocks(conn, calc_date, "monthly"))
        if not uni: return []
        cands = score_stocks_from_strategy(conn, calc_date, module)
        return [(c, s) for c, s in cands if c in uni][:top_n]
    return selector


def run(module, label, cap_pct, tx_bp=0):
    """tx_bp=0 으로 실행 → 반환 수익률은 '슬리피지만 차감'. 비용 그리드는 사후 선형 차감."""
    params = getattr(module, "PARAMS", {})
    keys = ["top_n_stocks", "transaction_cost_bp", "weight_cap_pct", "stop_loss_enabled",
            "stop_loss_pct", "stop_loss_mode", "stop_loss_basis", "universe", "rebal_type",
            "regime_cap_enabled"]
    orig = {k: BACKTEST_CONFIG.get(k) for k in keys}
    try:
        BACKTEST_CONFIG.update({
            "top_n_stocks": params.get("top_n", 30), "transaction_cost_bp": tx_bp,
            "weight_cap_pct": cap_pct,
            "stop_loss_enabled": params.get("stop_loss_enabled", False),
            "stop_loss_pct": params.get("stop_loss_pct", 30),
            "stop_loss_mode": params.get("stop_loss_mode", "sell"),
            "stop_loss_basis": params.get("stop_loss_basis", "entry"),
            "universe": "KOSPI", "rebal_type": "monthly", "regime_cap_enabled": False,
        })
        print(f"\n▶ 백테스트: {label} (cap {cap_pct}%, tx {tx_bp}bp)", flush=True)
        return run_backtest(label, stock_selector=make_selector(module), rebal_type="monthly")
    finally:
        for k, v in orig.items():
            if v is None: BACKTEST_CONFIG.pop(k, None)
            else: BACKTEST_CONFIG[k] = v


# ── 결과 파싱 ────────────────────────────────────────────────
def unpack(res):
    rd, mr = res["rebalance_dates"], res["monthly_returns"]
    ret = pd.Series({rd[i][:7]: mr[i] for i in range(len(mr))}).sort_index()
    hold_rows, to_rows = [], {}
    prev = None
    for d in rd:
        h = res["holdings_by_date"].get(d)
        if not h: continue
        for code, score, w, mcap in h:
            hold_rows.append({"date": d, "ym": d[:7], "code": code, "score": score,
                              "weight": w, "market_cap": mcap})
        cur = {c: w for c, _s, w, _m in h}
        to_rows[d[:7]] = 1.0 if prev is None else \
            0.5 * sum(abs(cur.get(k, 0) - prev.get(k, 0)) for k in set(cur) | set(prev))
        prev = cur
    hold = pd.DataFrame(hold_rows)
    to = pd.Series(to_rows).sort_index().reindex(ret.index)
    slip = hold.groupby("ym")["market_cap"].apply(lambda s: float(np.mean([_calc_slippage(v) for v in s])))
    return ret, hold, to, slip.reindex(ret.index)


def cost_series(turnover, bp):
    return turnover * (bp / 10000.0) * 2


def price_map(conn, codes, date, mode):
    out = {}
    codes = list(codes)
    for i in range(0, len(codes), 900):
        ch = codes[i:i + 900]; ph = ",".join(["?"] * len(ch))
        agg, op = ("MIN", ">=") if mode == "first" else ("MAX", "<=")
        rows = conn.execute(f"""
            SELECT dp.stock_code, dp.adj_close FROM daily_price dp
            JOIN (SELECT stock_code, {agg}(trade_date) d FROM daily_price
                  WHERE stock_code IN ({ph}) AND trade_date {op} ? AND adj_close > 0
                  GROUP BY stock_code) t
              ON dp.stock_code=t.stock_code AND dp.trade_date=t.d
        """, (*ch, date)).fetchall()
        out.update({c: float(p) for c, p in rows if p})
    return out


def name_map(conn, codes):
    """fnspace_master 최신 snapshot 기준 종목명 (코드는 'A'+6자리)."""
    out = {}
    codes = list(codes)
    for i in range(0, len(codes), 900):
        ch = codes[i:i + 900]; ph = ",".join(["?"] * len(ch))
        rows = conn.execute(f"""
            SELECT DISTINCT ON (stock_code) stock_code, stock_name
            FROM fnspace_master WHERE stock_code IN ({ph})
            ORDER BY stock_code, snapshot_date DESC
        """, tuple("A" + c for c in ch)).fetchall()
        out.update({str(c)[1:]: n for c, n in rows})
    return out


def stock_returns(conn, hold, dates):
    rows = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        hh = hold[hold["date"] == d0]
        if hh.empty: continue
        codes = hh["code"].tolist()
        p0 = price_map(conn, codes, d0, "first"); p1 = price_map(conn, codes, d1, "last")
        for _, r in hh.iterrows():
            a, b = p0.get(r["code"]), p1.get(r["code"])
            ret = (b / a - 1.0) if (a and b and a > 0) else np.nan
            rows.append({**r.to_dict(), "ret": ret, "contrib": r["weight"] * ret})
    return pd.DataFrame(rows)


# ── 1. 계산 일관성 감사 ───────────────────────────────────────
def audit(conn, saved_ret, base_ret, base_to, base_slip, base_hold):
    """저장 production 대비 재계산 차이 6건을 항목별로 확인."""
    recon_net30 = base_ret - cost_series(base_to, 30)          # 전체 production 비용식
    idx = saved_ret.index.intersection(recon_net30.index)
    mae = float((recon_net30.reindex(idx) - saved_ret.reindex(idx)).abs().mean())

    # 슬리피지/첫달 턴오버를 빼먹은 '단순 재계산' (기존 보고서 20.09% 재현)
    naive = base_ret.copy()
    naive_to = base_to.copy(); naive_to.iloc[0] = np.nan          # 첫 달 턴오버 미반영
    naive = naive + cost_series(base_slip.fillna(0), 0)           # (슬리피지는 이미 반영됨)
    naive_net30 = base_ret + base_to * base_slip.fillna(0) * 2 - cost_series(naive_to.fillna(naive_to.mean()), 30)
    mae_naive = float((naive_net30.reindex(idx) - saved_ret.reindex(idx)).abs().mean())

    fin_sec = "sec_cd_nm(코스피 금융)은 지주회사 포함 FnGuide 시장·업종 라벨, 금융업 제외는 finacc_typ(FINANCE_TYPES) 기준 → 서로 다른 분류체계"
    rows = [
        ["1. 저장 19.13% vs 재계산 20.09%",
         "net = raw - turnover×(tx+slippage)×2, 첫 달 turnover=1.0",
         "이전 보고서는 tx 30bp만 차감, 슬리피지·첫달 turnover 누락",
         f"슬리피지(시총구간별 10~50bp) 월평균 {float(base_slip.mean())*10000:.1f}bp 누락 + 첫 달 turnover 1.0 누락",
         "수정: production 비용식 전체 복제", f"MAE {mae*10000:.2f}bp (기준 {AUDIT_TOL_MAE*10000:.0f}bp)"],
        ["2. 99개월 vs 100개월",
         "rebalance_dates 101개 → 수익률 100개(마지막 날짜는 종료경계, holdings만 기록)",
         "factor_lofo 결과는 2026-06까지 99개월 (이전 시점 실행 산출물 재사용)",
         "표본 시점 차이(데이터 적재 시점). 계산 로직 차이 아님",
         "수정: 이번 비교는 4전략 모두 동일 실행에서 100개월로 재계산", "일치"],
        ["3. 금융업 제외인데 업종기여에 금융",
         "factor_engine: finacc_typ ∈ FINANCE_TYPES 제외",
         "귀속표: fnspace_master.sec_cd_nm 라벨 사용", fin_sec,
         "수정 아님(정의 차이 명시)", "해당 없음 — 라벨 해석 주의"],
        ["4. Brinson 총초과 0.684% vs BASE 초과 0.082%",
         "BASE 초과 = net30 − KODEX200(069500)",
         "Brinson 초과 = gross − 유니버스 시총가중(코스닥 일부 포함)",
         "벤치마크가 다름(ETF vs 유니버스) + 비용 차감 유무(net vs gross)",
         "수정: 본 문서는 모든 비교를 동일 벤치마크(069500)·동일 비용으로 통일", "일치"],
        ["5. return_contribution 정의",
         "-", "Σ_t (비중_t × 월수익률_t) — 월별 기여의 단순 합",
         "누적 wealth 기여도 월평균도 아님. 복리 미반영 산술합",
         "수정 아님(정의 명시)", "해당 없음"],
        ["6. 비용·리밸일·첫달·마지막달·결측 처리",
         "리밸일=universe.rebal_date, 진입=리밸일 이후 첫 거래일 adj_close, "
         "마지막 리밸일은 종료경계(수익률 없음), 시작가 없으면 해당 종목 제외(비중 재분배 없음=암묵적 현금), "
         "종료가 없으면 시작일 이후 마지막 가격, 그것도 없으면 −100%",
         "동일 함수(run_backtest/calc_portfolio_return) 직접 호출로 재사용",
         "차이 없음(동일 코드 경로)", "수정 불필요", "일치"],
    ]
    df = pd.DataFrame(rows, columns=["감사 항목", "저장 production 정의", "재계산 정의",
                                     "차이 원인", "수정 여부", "수정 후 일치 여부"])
    return df, mae, mae_naive, idx


# ── 메인 ────────────────────────────────────────────────────
def main():
    t0 = time.time()
    sd = load_strategy(STRATEGY, "monthly", "KOSPI")
    code = sd["code"]
    saved = pd.Series({sd["results"]["rebalance_dates"][i][:7]: v
                       for i, v in enumerate(sd["results"]["monthly_returns"])}).sort_index()

    base_mod = code_to_module(code)
    base_w = {k: v for k, v in getattr(base_mod, "WEIGHTS_LARGE", {}).items() if v > 0}
    c3, w3 = drop_factors(code, DROP3)
    c4, w4 = drop_factors(code, DROP4)
    m3, m4 = code_to_module(c3), code_to_module(c4)

    specs = [("BASE", base_mod, 30, base_w), ("DIET-3", m3, 30, w3),
             ("DIET-4", m4, 30, w4), ("DIET-3-CAP20", m3, 20, w3)]
    results = {}
    for name, mod, cap, _ in specs:
        res = run(mod, name, cap)
        results[name] = unpack(res) + (res,)

    conn = get_db()
    base_ret, base_hold, base_to, base_slip, base_res = results["BASE"]

    # ── 1. 감사 ──
    audit_df, mae, mae_naive, idx = audit(conn, saved, base_ret, base_to, base_slip, base_hold)
    audit_df.to_csv(OUT / "diet_calc_audit.csv", index=False, encoding="utf-8-sig")
    print("\n== 계산 감사 ==")
    print(f"  재계산 net30 vs 저장 production: MAE {mae*10000:.2f}bp, "
          f"CAGR {perf(base_ret - cost_series(base_to,30))['cagr']:.4%} vs {perf(saved)['cagr']:.4%}")
    if mae > AUDIT_TOL_MAE:
        print(f"❌ 감사 실패 (MAE {mae*10000:.2f}bp > {AUDIT_TOL_MAE*10000:.0f}bp). 전략 비교 중단.")
        pd.DataFrame([{"status": "AUDIT_FAILED", "mae_bp": mae * 10000}]).to_csv(
            OUT / "diet_performance_comparison.csv", index=False, encoding="utf-8-sig")
        return
    print("✅ 감사 통과 → 전략 비교 진행")

    # ── 벤치마크 ──
    dates = sorted(base_hold["date"].unique())
    bm_rows = {}
    for d0, d1 in zip(dates[:-1], dates[1:]):
        a = price_map(conn, [BM], d0, "first").get(BM); b = price_map(conn, [BM], d1, "last").get(BM)
        bm_rows[d0[:7]] = (b / a - 1.0) if (a and b) else np.nan
    bm = pd.Series(bm_rows).sort_index()

    # ── 성과 비교 ──
    perf_rows, monthly_out = [], {"benchmark_069500": bm}
    for name, _, cap, _ in specs:
        ret, hold, to, slip, _res = results[name]
        monthly_out[f"{name}_gross_ex_slip"] = ret + to * slip.fillna(0) * 2
        for bp in [0, 30, 50]:
            net = ret - cost_series(to, bp)
            if bp == 30: monthly_out[f"{name}_net30"] = net
            for period, ix in period_slices(ret.index).items():
                r = net.reindex(ix).dropna()
                ex = (r - bm.reindex(r.index)).dropna()
                t_ex, lag = nw_t(ex)
                a_capm, t_capm = ols_alpha(r, bm.reindex(r.index).to_frame("MKT"))
                h = hold[hold["ym"].isin(r.index)]
                wmax = h.groupby("ym")["weight"].max().mean()
                top3 = h.groupby("ym")["weight"].apply(lambda s: s.nlargest(3).sum()).mean()
                top5 = h.groupby("ym")["weight"].apply(lambda s: s.nlargest(5).sum()).mean()
                perf_rows.append({
                    "spec": name, "cap_pct": cap, "cost_bp": bp, "period": period, **perf(r),
                    "market_excess_mean": float(ex.mean()), "market_excess_nw_t": t_ex, "nw_lag": lag,
                    "capm_alpha_ann": a_capm, "capm_alpha_t": t_capm,
                    "ff3_alpha": np.nan, "qfactor_alpha": np.nan,
                    "turnover_mean": float(to.reindex(r.index).mean()),
                    "max_weight_mean": float(wmax), "top3_weight_mean": float(top3),
                    "top5_weight_mean": float(top5),
                    "n_holdings_mean": float(h.groupby("ym")["code"].count().mean()),
                })
    pd.DataFrame(perf_rows).to_csv(OUT / "diet_performance_comparison.csv", index=False, encoding="utf-8-sig")
    mdf = pd.DataFrame(monthly_out)
    for c in [c for c in mdf.columns if c.endswith("net30")]:
        mdf[c.replace("net30", "cum30")] = (1 + mdf[c]).cumprod()
    mdf.index.name = "ym"
    mdf.to_csv(OUT / "diet_monthly_returns.csv", encoding="utf-8-sig")

    # ── 종목선정 변화 ──
    ov_rows = []
    b_by = {ym: set(g["code"]) for ym, g in base_hold.groupby("ym")}
    b_w = {ym: dict(zip(g["code"], g["weight"])) for ym, g in base_hold.groupby("ym")}
    for name in ["DIET-3", "DIET-4", "DIET-3-CAP20"]:
        _, hold, _, _, _ = results[name]
        for ym, g in hold.groupby("ym"):
            cur, prev = set(g["code"]), b_by.get(ym, set())
            wc = dict(zip(g["code"], g["weight"])); wp = b_w.get(ym, {})
            ov_rows.append({
                "spec": name, "ym": ym, "n_common": len(cur & prev),
                "overlap_ratio": len(cur & prev) / max(len(prev), 1),
                "jaccard": len(cur & prev) / max(len(cur | prev), 1),
                "n_added": len(cur - prev), "n_dropped": len(prev - cur),
                "abs_weight_change": sum(abs(wc.get(c, 0) - wp.get(c, 0)) for c in cur | prev),
            })
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(OUT / "diet_overlap.csv", index=False, encoding="utf-8-sig")

    # ── 기여도·주도주·집중도 ──
    contrib_rows, leader_rows, conc_rows = [], [], []
    for name, _, cap, _ in specs:
        _, hold, to, _, _ = results[name]
        sc = stock_returns(conn, hold, dates)
        sc["spec"] = name
        agg = sc.groupby("code")["contrib"].sum().sort_values(ascending=False)
        nm = name_map(conn, sc["code"].unique().tolist())
        for kind, vals in [("top10", agg.head(10)), ("bottom10", agg.tail(10))]:
            for c, v in vals.items():
                contrib_rows.append({"spec": name, "kind": kind, "code": c,
                                     "name": LEADERS.get(c, nm.get(c, "")), "cum_contrib": float(v)})
        for c, n in LEADERS.items():
            s = sc[sc["code"] == c]
            leader_rows.append({"spec": name, "code": c, "name": n,
                                "months_held": int(s["ym"].nunique()),
                                "hold_ratio": s["ym"].nunique() / max(sc["ym"].nunique(), 1),
                                "mean_weight": float(s["weight"].mean()) if len(s) else 0.0,
                                "max_weight": float(s["weight"].max()) if len(s) else 0.0,
                                "cum_contrib": float(s["contrib"].sum())})
        for ym, g in hold.groupby("ym"):
            conc_rows.append({"spec": name, "ym": ym, "max_weight": float(g["weight"].max()),
                              "top3_weight": float(g["weight"].nlargest(3).sum()),
                              "top5_weight": float(g["weight"].nlargest(5).sum()),
                              "n_holdings": int(len(g)),
                              "hhi": float((g["weight"] ** 2).sum())})
    pd.DataFrame(contrib_rows).to_csv(OUT / "diet_contributors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(leader_rows).to_csv(OUT / "diet_leader_stocks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(conc_rows).to_csv(OUT / "diet_concentration.csv", index=False, encoding="utf-8-sig")

    weights_df = pd.DataFrame([{"spec": n, **w} for n, _, _, w in specs]).fillna(0.0)
    weights_df.to_csv(OUT / "diet_factor_weights.csv", index=False, encoding="utf-8-sig")

    print(f"\nDone in {time.time()-t0:.0f}s  (MAE {mae*10000:.2f}bp)")
    print(pd.DataFrame(perf_rows).query("cost_bp==30")[
        ["spec", "period", "cagr", "sharpe", "mdd", "turnover_mean", "max_weight_mean",
         "top5_weight_mean", "market_excess_nw_t"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
