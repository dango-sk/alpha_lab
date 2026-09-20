"""
analysis/fcf_base_compare.py   (실험 스크립트, production 미수정)

FCF_YIELD추가전략(FCF 15%)이 유의미한지 1차 검증.

  [액션 2] BASE_no_FCF = FCF만 제거하고 나머지 팩터를 비례 확대(×1/0.85)한 기준전략
  [액션 3] 월별 증분수익 FCF_INC_t = R_t(FCF15) - R_t(BASE) 통계
           (월평균/연환산/Newey-West t/증분IR/승률/tx 전후/누적/rolling 24·36M)
  [액션 4] 종목 교체 내역 분해 (both / FCF진입 / FCF제외 / 둘다미보유)
           + R(FCF진입) - R(FCF제외) 검정 + 종목별 기여도 집중도

두 전략은 종목선정 외 모든 설정(top30·cap30%·손절OFF·퀄리티필터) 동일.
거래비용 전(0bp)·후(30bp) 두 번 백테스트한다.

실행:
    .venv/bin/python analysis/fcf_base_compare.py
산출:
    analysis/results/fcf_base_monthly.csv      월별 수익/증분/rolling
    analysis/results/fcf_base_switch.csv       월별 종목 교체·수익 분해
    analysis/results/fcf_base_names.csv        종목별 누적 기여 (집중도 확인용)
"""
import os, sys, json, re
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from lib.data import load_strategy
from lib.factor_engine import code_to_module, score_stocks_from_strategy, prefetch_all_data
from config.settings import BACKTEST_CONFIG
from step7_backtest import (
    run_backtest, get_universe_stocks, get_db,
)

BULL = "FCF_YIELD추가전략"
CUTOFF = "2026-07"          # 예정(forward) 리밸 월은 수익률이 없으므로 제외
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
# [액션 2] BASE_no_FCF 코드 생성
# ══════════════════════════════════════════════════════════════
def make_base_code(code: str) -> tuple[str, dict, dict]:
    """FCF_YIELD=0, 나머지 비중을 1/(1-w_fcf)배로 비례 확대한 전략 코드 반환."""
    mod = code_to_module(code)
    w = dict(getattr(mod, "WEIGHTS_LARGE", {}))
    w_fcf = w.get("FCF_YIELD", 0.0)
    if w_fcf <= 0:
        raise SystemExit("원 전략에 FCF_YIELD 비중이 없다 — 비교 의미 없음")
    k = 1.0 / (1.0 - w_fcf)
    new_w = {key: (0.0 if key == "FCF_YIELD" else round(v * k, 8)) for key, v in w.items()}
    # 반올림 잔차(≤1e-5)는 최대 비중 팩터에 흡수시켜 합계 1.0을 정확히 맞춘다
    resid = 1.0 - sum(new_w.values())
    if abs(resid) > 0:
        anchor = max((key for key in new_w if key != "FCF_YIELD"), key=lambda key: new_w[key])
        new_w[anchor] = round(new_w[anchor] + resid, 8)
    s = sum(new_w.values())
    assert abs(s - 1.0) < 1e-9, f"가중치 합 {s}"

    body = "WEIGHTS_LARGE = {\n" + "".join(f'    "{key}": {v!r},\n' for key, v in new_w.items()) + "}"
    new_code, n = re.subn(r"WEIGHTS_LARGE\s*=\s*\{[^}]*\}", body, code, count=1, flags=re.DOTALL)
    if n != 1:
        raise SystemExit("WEIGHTS_LARGE 블록 치환 실패")
    return new_code, w, new_w


# ══════════════════════════════════════════════════════════════
# 백테스트 러너 (두 전략 동일 설정, tx만 스위치)
# ══════════════════════════════════════════════════════════════
def make_selector(module, rebal_type="monthly"):
    def selector(conn, calc_date, top_n):
        uni = get_universe_stocks(conn, calc_date, rebal_type)
        if not uni:
            return []
        cands = score_stocks_from_strategy(conn, calc_date, module)
        return [(c, s) for c, s in cands if c in uni][:top_n]
    return selector


def run(module, label, tx_bp):
    params = getattr(module, "PARAMS", {})
    keys = ["top_n_stocks", "transaction_cost_bp", "weight_cap_pct",
            "stop_loss_enabled", "stop_loss_pct", "stop_loss_mode", "stop_loss_basis",
            "universe", "rebal_type", "regime_cap_enabled"]
    orig = {k: BACKTEST_CONFIG.get(k) for k in keys}
    try:
        BACKTEST_CONFIG["top_n_stocks"]        = params.get("top_n", 30)
        BACKTEST_CONFIG["transaction_cost_bp"] = tx_bp
        BACKTEST_CONFIG["weight_cap_pct"]      = params.get("weight_cap_pct", 30)
        BACKTEST_CONFIG["stop_loss_enabled"]   = params.get("stop_loss_enabled", False)
        BACKTEST_CONFIG["stop_loss_pct"]       = params.get("stop_loss_pct", 30)
        BACKTEST_CONFIG["stop_loss_mode"]      = params.get("stop_loss_mode", "sell")
        BACKTEST_CONFIG["stop_loss_basis"]     = params.get("stop_loss_basis", "entry")
        BACKTEST_CONFIG["universe"]            = "KOSPI"
        BACKTEST_CONFIG["rebal_type"]          = "monthly"
        BACKTEST_CONFIG["regime_cap_enabled"]  = False
        print(f"\n▶ 백테스트: {label} (tx {tx_bp}bp)", flush=True)
        return run_backtest(label, stock_selector=make_selector(module), rebal_type="monthly")
    finally:
        for k, v in orig.items():
            if v is None:
                BACKTEST_CONFIG.pop(k, None)
            else:
                BACKTEST_CONFIG[k] = v


# ══════════════════════════════════════════════════════════════
# 통계 도구
# ══════════════════════════════════════════════════════════════
def nw_tstat(x, lag=None):
    """평균=0 귀무가설의 Newey-West(HAC) t값. lag 기본 = floor(4*(n/100)^(2/9))."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    s = float(e @ e) / n
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * float(e[l:] @ e[:-l]) / n
    se = np.sqrt(max(s, 0.0) / n)
    return (x.mean() / se if se > 0 else np.nan), lag


def ann(x):
    """월수익 배열 → (기하 연환산, 연환산 변동성)"""
    x = np.asarray(x, float)
    g = float(np.prod(1 + x)) ** (12.0 / len(x)) - 1.0
    return g, float(x.std(ddof=1)) * np.sqrt(12)


def summarize(inc, tag):
    m = float(np.mean(inc))
    ann_arith = m * 12
    t, lag = nw_tstat(inc)
    ir = (m / np.std(inc, ddof=1) * np.sqrt(12)) if np.std(inc, ddof=1) > 0 else np.nan
    win = float(np.mean(np.asarray(inc) > 0))
    print(f"  [{tag}] 월평균 {m*100:+.3f}%  연환산(산술) {ann_arith*100:+.2f}%  "
          f"NW t={t:+.2f}(lag{lag})  증분IR {ir:+.2f}  승률 {win*100:.1f}%  n={len(inc)}")
    return dict(tag=tag, mean=m, ann=ann_arith, t=t, lag=lag, ir=ir, win=win, n=len(inc))


# ══════════════════════════════════════════════════════════════
# [액션 4] 종목 교체 분해
# ══════════════════════════════════════════════════════════════
def stock_returns(conn, codes, start, end):
    """{code: 기간수익률} — run_backtest의 adj_close 규칙과 동일(상폐 시 마지막 시세)."""
    if not codes:
        return {}
    codes = list(codes)
    ph = ",".join(["?"] * len(codes))
    srows = conn.execute(f"""
        SELECT dp.stock_code, dp.adj_close FROM daily_price dp
        INNER JOIN (SELECT stock_code, MIN(trade_date) d FROM daily_price
                    WHERE stock_code IN ({ph}) AND trade_date >= ? AND adj_close > 0
                    GROUP BY stock_code) t
          ON dp.stock_code = t.stock_code AND dp.trade_date = t.d
    """, (*codes, start)).fetchall()
    erows = conn.execute(f"""
        SELECT dp.stock_code, dp.adj_close FROM daily_price dp
        INNER JOIN (SELECT stock_code, MAX(trade_date) d FROM daily_price
                    WHERE stock_code IN ({ph}) AND trade_date <= ? AND adj_close > 0
                    GROUP BY stock_code) t
          ON dp.stock_code = t.stock_code AND dp.trade_date = t.d
    """, (*codes, end)).fetchall()
    sm = {r[0]: r[1] for r in srows}
    em = {r[0]: r[1] for r in erows}
    out = {}
    for c in codes:
        p0, p1 = sm.get(c), em.get(c)
        if p0 and p1 and p0 > 0:
            out[c] = p1 / p0 - 1.0
    return out


def switch_analysis(res_f, res_b):
    """holdings_by_date 비교 → 진입/제외 종목의 다음 기간 수익률 (동일가중, 단순 비교)."""
    hf, hb = res_f["holdings_by_date"], res_b["holdings_by_date"]
    dates = [d for d in res_f["rebalance_dates"] if d in hf and d in hb]
    dates = [d for d in dates if d[:7] <= CUTOFF]
    rd = res_f["rebalance_dates"]
    nxt = {rd[i]: rd[i + 1] for i in range(len(rd) - 1)}

    conn = get_db()
    rows, name_rows = [], []
    try:
        for d in dates:
            e = nxt.get(d)
            if not e:
                continue
            F = {c: (w, s) for c, s, w, _mc in hf[d]}
            B = {c: (w, s) for c, s, w, _mc in hb[d]}
            both = set(F) & set(B)
            only_f = set(F) - set(B)      # FCF 때문에 새로 진입
            only_b = set(B) - set(F)      # FCF 때문에 제외
            rets = stock_returns(conn, set(F) | set(B), d, e)

            def avg(codes, wmap=None):
                v = [rets[c] for c in codes if c in rets]
                return float(np.mean(v)) if v else np.nan

            def wavg(codes, hmap):
                num = sum(hmap[c][0] * rets[c] for c in codes if c in rets)
                den = sum(hmap[c][0] for c in codes if c in rets)
                return num / den if den > 0 else np.nan

            r_in, r_out = avg(only_f), avg(only_b)
            # 비중 가중 기여 차이 = FCF 전략 초과의 '선정 효과' 근사
            contrib = (sum(F[c][0] * rets.get(c, 0.0) for c in only_f)
                       - sum(B[c][0] * rets.get(c, 0.0) for c in only_b))
            rows.append(dict(
                date=d, ym=d[:7], n_both=len(both), n_in=len(only_f), n_out=len(only_b),
                overlap=len(both) / max(len(F), 1),
                ret_in=r_in, ret_out=r_out, spread=(r_in - r_out) if (r_in == r_in and r_out == r_out) else np.nan,
                wret_in=wavg(only_f, F), wret_out=wavg(only_b, B),
                w_in=sum(F[c][0] for c in only_f), w_out=sum(B[c][0] for c in only_b),
                contrib=contrib,
            ))
            for c in only_f:
                name_rows.append(dict(date=d, ym=d[:7], code=c, side="IN",
                                      weight=F[c][0], ret=rets.get(c, np.nan),
                                      contrib=F[c][0] * rets.get(c, 0.0)))
            for c in only_b:
                name_rows.append(dict(date=d, ym=d[:7], code=c, side="OUT",
                                      weight=B[c][0], ret=rets.get(c, np.nan),
                                      contrib=-B[c][0] * rets.get(c, 0.0)))
    finally:
        conn.close()
    return pd.DataFrame(rows), pd.DataFrame(name_rows)


# ══════════════════════════════════════════════════════════════
def main():
    sd = load_strategy(BULL, rebal_type="monthly", universe="KOSPI")
    fcf_code = sd["code"]
    base_code, w_old, w_new = make_base_code(fcf_code)

    print("═" * 78)
    print("  [액션 2] BASE_no_FCF 가중치 (FCF 제거 + 비례 확대)")
    print("═" * 78)
    for k in w_old:
        if w_old[k] or w_new[k]:
            print(f"   {k:14s} {w_old[k]*100:6.2f}%  →  {w_new[k]*100:6.2f}%")
    print(f"   {'합계':14s} {sum(w_old.values())*100:6.2f}%  →  {sum(w_new.values())*100:6.2f}%")

    m_fcf, m_base = code_to_module(fcf_code), code_to_module(base_code)

    conn = get_db(); prefetch_all_data(conn); conn.close()

    out = {}
    for tx in (0, 30):
        out[("FCF", tx)] = run(m_fcf, f"FCF15_tx{tx}", tx)
        out[("BASE", tx)] = run(m_base, f"BASE_noFCF_tx{tx}", tx)

    # ── 월별 정렬 (수익 i번째 ↔ rebalance_dates[i]) ──
    def series(res):
        rd, mr = res["rebalance_dates"], res["monthly_returns"]
        s = {rd[i][:7]: mr[i] for i in range(len(mr))}
        return {k: v for k, v in s.items() if k <= CUTOFF}

    s = {k: series(v) for k, v in out.items()}
    months = sorted(set(s[("FCF", 30)]) & set(s[("BASE", 30)]))
    df = pd.DataFrame({"ym": months})
    for tx in (0, 30):
        df[f"fcf_tx{tx}"] = [s[("FCF", tx)][m] for m in months]
        df[f"base_tx{tx}"] = [s[("BASE", tx)][m] for m in months]
        df[f"inc_tx{tx}"] = df[f"fcf_tx{tx}"] - df[f"base_tx{tx}"]

    print("\n" + "═" * 78)
    print(f"  [액션 3] 증분수익 FCF_INC = R(FCF15) - R(BASE)   {months[0]}~{months[-1]} {len(months)}개월")
    print("═" * 78)
    for tx in (0, 30):
        cf, cb = df[f"fcf_tx{tx}"].values, df[f"base_tx{tx}"].values
        (gf, vf), (gb, vb) = ann(cf), ann(cb)
        print(f"\n  ── tx {tx}bp ──")
        print(f"  FCF15     CAGR {gf*100:6.2f}%  vol {vf*100:5.2f}%  Sharpe {gf/vf:5.2f}")
        print(f"  BASE      CAGR {gb*100:6.2f}%  vol {vb*100:5.2f}%  Sharpe {gb/vb:5.2f}")
        print(f"  기하 차이 {(gf-gb)*100:+.2f}%p")
        summarize(df[f"inc_tx{tx}"].values, f"tx{tx}bp")

    # 누적 증분 + rolling
    for tx in (0, 30):
        df[f"cum_inc_tx{tx}"] = (1 + df[f"inc_tx{tx}"]).cumprod() - 1     # 증분수익 자체의 복리
        df[f"cum_gap_tx{tx}"] = ((1 + df[f"fcf_tx{tx}"]).cumprod()
                                 - (1 + df[f"base_tx{tx}"]).cumprod())   # 누적 자산 격차
        for W in (24, 36):
            df[f"roll{W}_tx{tx}"] = (df[f"inc_tx{tx}"].rolling(W)
                                     .apply(lambda x: np.prod(1 + x) ** (12 / W) - 1, raw=True))

    for tx in (0, 30):
        for W in (24, 36):
            r = df[f"roll{W}_tx{tx}"].dropna()
            if len(r):
                print(f"  rolling {W}M 증분(연환산, tx{tx}bp): 평균 {r.mean()*100:+.2f}%  "
                      f"양(+) 비율 {float((r>0).mean())*100:.0f}%  최저 {r.min()*100:+.2f}%  최고 {r.max()*100:+.2f}%")
    print(f"\n  누적 증분(복리, tx30bp) {df['cum_inc_tx30'].iloc[-1]*100:+.1f}%   "
          f"누적 자산격차 {df['cum_gap_tx30'].iloc[-1]*100:+.1f}%p")

    # 연도별
    df["yr"] = df["ym"].str[:4]
    yr = df.groupby("yr").agg(fcf=("fcf_tx30", lambda x: np.prod(1 + x) - 1),
                              base=("base_tx30", lambda x: np.prod(1 + x) - 1))
    yr["inc"] = yr["fcf"] - yr["base"]
    print("\n  연도별 (tx30bp)")
    for y, r in yr.iterrows():
        print(f"    {y}  FCF {r.fcf*100:+7.2f}%   BASE {r.base*100:+7.2f}%   차이 {r.inc*100:+6.2f}%p")

    df.to_csv(OUT / "fcf_base_monthly.csv", index=False, encoding="utf-8-sig")

    # ── [액션 4] ──
    print("\n" + "═" * 78)
    print("  [액션 4] 종목 교체 분해 (tx30bp 백테스트의 holdings 기준)")
    print("═" * 78)
    sw, names = switch_analysis(out[("FCF", 30)], out[("BASE", 30)])
    sw.to_csv(OUT / "fcf_base_switch.csv", index=False, encoding="utf-8-sig")

    print(f"  월평균 공통보유 {sw.n_both.mean():.1f}종목 (중복률 {sw.overlap.mean()*100:.1f}%), "
          f"교체 {sw.n_in.mean():.1f}종목/월  (교체비중 {sw.w_in.mean()*100:.1f}%)")
    sp = sw["spread"].dropna().values
    print(f"  R(진입) 평균 {sw.ret_in.mean()*100:+.2f}%  vs  R(제외) 평균 {sw.ret_out.mean()*100:+.2f}%")
    summarize(sp, "동일가중 spread(진입-제외)")
    summarize(sw["contrib"].dropna().values, "비중가중 기여차(선정효과)")

    if len(names):
        names.to_csv(OUT / "fcf_base_names.csv", index=False, encoding="utf-8-sig")
        agg = names.groupby("code")["contrib"].sum().sort_values()
        tot = float(names["contrib"].sum())
        top = agg.abs().sort_values(ascending=False).head(10)
        share = float(agg.reindex(top.index).sum() / tot) if tot else np.nan
        print(f"\n  종목 집중도: 누적 기여 합 {tot*100:+.1f}%p, 상위10종목이 {share*100:.0f}% 설명")
        print("  ▲ 기여 상위5:", ", ".join(f"{c}({v*100:+.1f}%p)" for c, v in agg.tail(5)[::-1].items()))
        print("  ▼ 기여 하위5:", ", ".join(f"{c}({v*100:+.1f}%p)" for c, v in agg.head(5).items()))

    print(f"\n저장: {OUT}/fcf_base_monthly.csv, fcf_base_switch.csv, fcf_base_names.csv")


if __name__ == "__main__":
    main()
