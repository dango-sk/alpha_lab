"""
analysis/fcf_loso.py  (실험 스크립트, production 미수정)

[분석 C 보완] Leave-one-stock-out (주요 기여 종목 대상, post-OOS diagnostic).
  output/fcf_monthly_contrib.csv 를 입력으로 사용한다 (fcf_size_and_repro.py 가 생성).
  제외 후 남은 종목 비중은 매월 비례 재조정한다.

실행: .venv/bin/python analysis/fcf_loso.py
산출: output/fcf_loso_diagnostic.csv
"""
import math
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).parent.parent
OUT = REPO / "output"
IS_END, OOS_START = "2024-06", "2024-07"
TARGETS = ["SK하이닉스", "SK스퀘어", "기아", "현대모비스", "POSCO홀딩스", "삼성전자",
           "HD한국조선해양", "LG이노텍", "현대차", "LG"]


def nw(x, lag=None):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 3: return np.nan, 0
    if lag is None: lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean(); s = float(e @ e) / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * float(e[l:] @ e[:-l]) / n
    se = math.sqrt(max(s, 0) / n)
    return (float(x.mean() / se) if se > 0 else np.nan), lag


def ret(df, wc="w_production"):
    x = df.dropna(subset=["ret"]).copy()
    x["w2"] = x[wc] / x.groupby("ym")[wc].transform("sum")
    return x.assign(v=x.w2 * x["ret"]).groupby("ym")["v"].sum().sort_index()


def turnover(df, wc="w_production"):
    out, prev = {}, None
    for ym, g in df.groupby("ym", sort=True):
        w = g[wc] / g[wc].sum()
        cur = dict(zip(g["code"], w))
        if prev is not None:
            out[ym] = 0.5 * sum(abs(cur.get(k, 0) - prev.get(k, 0)) for k in set(cur) | set(prev))
        prev = cur
    return pd.Series(out, dtype=float)


def stats(s):
    s = s.dropna(); cum = (1 + s).cumprod()
    cagr = float((1 + s).prod() ** (12 / len(s)) - 1)
    vol = float(s.std(ddof=1)) * np.sqrt(12)
    return dict(months=len(s), cagr=cagr, monthly_mean=float(s.mean()), vol=vol,
                sharpe=(cagr / vol if vol else np.nan),
                mdd=float((cum / cum.cummax() - 1).min()))


def main():
    c = pd.read_csv(OUT / "fcf_monthly_contrib.csv", dtype={"code": str})
    base = ret(c); base_to = turnover(c)
    rows = []
    for name, sub in [("BASE", c)] + [(f"LOSO_{n}", c[c["name"] != n]) for n in TARGETS]:
        r_all, to_all = ret(sub), turnover(sub)
        for per, idx in {"Full": base.index,
                         "IS": [m for m in base.index if m <= IS_END],
                         "OOS_post-OOS diagnostic": [m for m in base.index if m >= OOS_START]}.items():
            r = r_all.reindex(idx).dropna()
            d = (r - base.reindex(r.index)).dropna()
            t, lag = nw(d)
            rows.append({"spec": name, "period": per, **stats(r),
                         "turnover_mean": float(to_all.reindex(r.index).mean()),
                         "d_turnover": float(to_all.reindex(r.index).mean() - base_to.reindex(r.index).mean()),
                         "d_monthly_mean": float(d.mean()), "d_nw_t": t, "nw_lag": lag})
    pd.DataFrame(rows).to_csv(OUT / "fcf_loso_diagnostic.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
