"""
analysis/fcf_turnover_buffer.py  (실험 스크립트, production 미수정)

[분석 E 보완] 회전율 분해 + 사전지정 보유완충(buffer) 규칙 post-OOS diagnostic.

  1) BASE(현행 Top30) 재실행 → 월별 full-universe 랭크 기록
     - 신규편입/퇴출로 인한 turnover 분해
     - 순위 경계(25~35위) 종목의 반복 진입·퇴출 비중
  2) BUFFER 사양(사전지정, 최적화 없음): 신규편입 25위 이내, 기존보유 40위까지 유지,
     보유 종목 수는 production과 동일하게 30개로 맞춘다.
  3) 두 사양의 0/30/50bp 성과 비교.

실행:
    .venv/bin/python analysis/fcf_turnover_buffer.py
산출:
    output/fcf_turnover_decomposition.csv
    output/fcf_buffer_diagnostic.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
load_dotenv(REPO / ".env")

from lib.data import load_strategy                                    # noqa: E402
from lib.factor_engine import code_to_module, score_stocks_from_strategy  # noqa: E402
from config.settings import BACKTEST_CONFIG                           # noqa: E402
from step7_backtest import run_backtest, get_universe_stocks          # noqa: E402

OUT = REPO / "output"; OUT.mkdir(exist_ok=True)
STRATEGY = "FCF_YIELD추가전략"
CUTOFF = "2026-07"
IS_END, OOS_START = "2024-06", "2024-07"
ENTRY_RANK, HOLD_RANK, TOP_N = 25, 40, 30


def nw_t(x, lag=None):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 3: return np.nan, 0
    if lag is None: lag = int(np.floor(4 * (n / 100.0) ** (2 / 9)))
    e = x - x.mean(); s = float(e @ e) / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1.0)) * float(e[l:] @ e[:-l]) / n
    se = np.sqrt(max(s, 0.0) / n)
    return (float(x.mean() / se) if se > 0 else np.nan), lag


def perf(x):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) == 0: return {}
    cum = np.cumprod(1 + x); vol = float(x.std(ddof=1)) * np.sqrt(12)
    cagr = float(np.prod(1 + x)) ** (12 / len(x)) - 1
    return dict(months=len(x), cagr=cagr, monthly_mean=float(x.mean()), vol=vol,
                sharpe=(cagr / vol if vol else np.nan),
                mdd=float((cum / np.maximum.accumulate(cum) - 1).min()))


class Selector:
    """rank 기록 + (옵션) buffer 규칙."""
    def __init__(self, module, buffer=False):
        self.module, self.buffer = module, buffer
        self.ranks = {}          # date -> {code: rank}
        self.held = []           # 직전 보유 종목

    def __call__(self, conn, calc_date, top_n):
        uni = set(get_universe_stocks(conn, calc_date, "monthly"))
        if not uni: return []
        cands = [(c, s) for c, s in score_stocks_from_strategy(conn, calc_date, self.module) if c in uni]
        self.ranks[calc_date] = {c: i + 1 for i, (c, _) in enumerate(cands)}
        if not self.buffer:
            sel = cands[:top_n]
        else:
            rank = self.ranks[calc_date]
            score = dict(cands)
            keep = [c for c in self.held if rank.get(c, 10**9) <= HOLD_RANK]
            keep = sorted(keep, key=lambda c: rank[c])[:top_n]
            for c, _ in cands[:ENTRY_RANK]:
                if len(keep) >= top_n: break
                if c not in keep: keep.append(c)
            for c, _ in cands:                      # 30개 미달 시 순위대로 보충
                if len(keep) >= top_n: break
                if c not in keep: keep.append(c)
            keep = sorted(keep, key=lambda c: rank.get(c, 10**9))[:top_n]
            sel = [(c, score[c]) for c in keep]
        self.held = [c for c, _ in sel]
        return sel


def run(module, buffer, label):
    params = getattr(module, "PARAMS", {})
    keys = ["top_n_stocks", "transaction_cost_bp", "weight_cap_pct", "stop_loss_enabled",
            "stop_loss_pct", "stop_loss_mode", "stop_loss_basis", "universe", "rebal_type",
            "regime_cap_enabled"]
    orig = {k: BACKTEST_CONFIG.get(k) for k in keys}
    sel = Selector(module, buffer=buffer)
    try:
        BACKTEST_CONFIG.update({
            "top_n_stocks": params.get("top_n", TOP_N), "transaction_cost_bp": 0,
            "weight_cap_pct": params.get("weight_cap_pct", 30),
            "stop_loss_enabled": params.get("stop_loss_enabled", False),
            "stop_loss_pct": params.get("stop_loss_pct", 30),
            "stop_loss_mode": params.get("stop_loss_mode", "sell"),
            "stop_loss_basis": params.get("stop_loss_basis", "entry"),
            "universe": "KOSPI", "rebal_type": "monthly", "regime_cap_enabled": False,
        })
        print(f"\n▶ {label}", flush=True)
        res = run_backtest(label, stock_selector=sel, rebal_type="monthly")
    finally:
        for k, v in orig.items():
            if v is None: BACKTEST_CONFIG.pop(k, None)
            else: BACKTEST_CONFIG[k] = v
    return res, sel


def decompose(res):
    """월별 turnover를 신규편입/퇴출로 분해."""
    h, rd = res["holdings_by_date"], res["rebalance_dates"]
    rows, prev = [], None
    for d in rd:
        if d not in h: continue
        cur = {c: w for c, _s, w, _mc in h[d]}
        if prev is not None:
            new = {c: w for c, w in cur.items() if c not in prev}
            gone = {c: w for c, w in prev.items() if c not in cur}
            same = sum(abs(cur[c] - prev[c]) for c in set(cur) & set(prev))
            rows.append({"ym": d[:7], "turnover": 0.5 * (sum(new.values()) + sum(gone.values()) + same),
                         "entry_turnover": 0.5 * sum(new.values()),
                         "exit_turnover": 0.5 * sum(gone.values()),
                         "reweight_turnover": 0.5 * same,
                         "n_new": len(new), "n_exit": len(gone)})
        prev = cur
    return pd.DataFrame(rows)


def boundary_churn(res, ranks):
    """직전 리밸에서 편출된 종목이 3개월 내 재편입되는 비율 + 경계(25~35위) 비중."""
    h, rd = res["holdings_by_date"], res["rebalance_dates"]
    dates = [d for d in rd if d in h]
    held = {d: {c for c, _s, _w, _mc in h[d]} for d in dates}
    rows = []
    for i, d in enumerate(dates):
        r = ranks.get(d, {})
        cur = held[d]
        boundary = [c for c in cur if 25 <= r.get(c, 999) <= 35]
        exits = held[dates[i - 1]] - cur if i else set()
        future = set().union(*[held[dates[j]] for j in range(i + 1, min(i + 4, len(dates)))]) if i else set()
        rows.append({"ym": d[:7], "n_boundary_25_35": len(boundary),
                     "boundary_share": len(boundary) / max(len(cur), 1),
                     "n_exit_prev": len(exits),
                     "reentry_within_3m": len(exits & future) / max(len(exits), 1) if exits else np.nan})
    return pd.DataFrame(rows)


def series(res):
    rd, mr = res["rebalance_dates"], res["monthly_returns"]
    s = pd.Series({rd[i][:7]: mr[i] for i in range(len(mr))}).sort_index()
    return s[s.index <= CUTOFF]


def main():
    sd = load_strategy(STRATEGY, "monthly", "KOSPI")
    module = code_to_module(sd["code"])

    base_res, base_sel = run(module, False, "BASE_rerun")
    buf_res, buf_sel = run(module, True, "BUFFER_25_40")

    dec_b, dec_f = decompose(base_res), decompose(buf_res)
    dec_b.insert(0, "spec", "BASE"); dec_f.insert(0, "spec", "BUFFER_25_40")
    ch_b = boundary_churn(base_res, base_sel.ranks); ch_b.insert(0, "spec", "BASE")
    ch_f = boundary_churn(buf_res, buf_sel.ranks); ch_f.insert(0, "spec", "BUFFER_25_40")
    dec = pd.concat([dec_b, dec_f], ignore_index=True).merge(
        pd.concat([ch_b, ch_f], ignore_index=True), on=["spec", "ym"], how="outer")
    dec.to_csv(OUT / "fcf_turnover_decomposition.csv", index=False, encoding="utf-8-sig")

    rows = []
    for spec, res, dc in [("BASE", base_res, dec_b), ("BUFFER_25_40_postOOS", buf_res, dec_f)]:
        g = series(res)
        to = dc.set_index("ym")["turnover"].reindex(g.index)
        for period, idx in {"Full": g.index, "IS": [m for m in g.index if m <= IS_END],
                            "OOS_post-OOS diagnostic": [m for m in g.index if m >= OOS_START]}.items():
            r = g.reindex(idx).dropna()
            t_ = to.reindex(r.index).fillna(to.mean())
            for bp in [0, 30, 50]:
                net = r - t_ * (bp / 10000.0) * 2
                tt, lag = nw_t(net)
                rows.append({"spec": spec, "period": period, "cost_bp": bp, **perf(net),
                             "turnover_mean": float(t_.mean()),
                             "entry_turnover_mean": float(dc.set_index("ym")["entry_turnover"].reindex(r.index).mean()),
                             "exit_turnover_mean": float(dc.set_index("ym")["exit_turnover"].reindex(r.index).mean()),
                             "nw_t_mean_ret": tt, "nw_lag": lag})
    pd.DataFrame(rows).to_csv(OUT / "fcf_buffer_diagnostic.csv", index=False, encoding="utf-8-sig")
    print("\nWrote output/fcf_turnover_decomposition.csv, output/fcf_buffer_diagnostic.csv")


if __name__ == "__main__":
    main()
