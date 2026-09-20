"""
analysis/fcf_holdings_turnover.py  (실험 스크립트, production 미수정)

FCF_YIELD추가전략의
  (1) 최근(=이번 8월 예정 리밸 포함) 보유 종목 리스트
  (2) 월별 턴오버 시계열/요약

fcf_base_compare.py와 동일한 백테스트 설정(top30·cap30%·손절OFF·KOSPI·monthly)을 쓴다.

실행:
    .venv/bin/python analysis/fcf_holdings_turnover.py
산출:
    analysis/results/fcf_turnover.csv     월별 턴오버
    analysis/results/fcf_holdings_last.csv 최근 리밸 보유종목
"""
import sys, json
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import pandas as pd
from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from lib.data import load_strategy
from lib.factor_engine import code_to_module, prefetch_all_data
from config.settings import BACKTEST_CONFIG
from step7_backtest import run_backtest, get_universe_stocks, get_db
from analysis.fcf_base_compare import BULL, make_selector, run  # 동일 설정 재사용

OUT = REPO / "analysis" / "results"
OUT.mkdir(exist_ok=True)

NAMES = {}
p = REPO / "data" / "code_to_name.json"
if p.exists():
    NAMES = json.load(open(p, encoding="utf-8"))


def main():
    conn = get_db(); prefetch_all_data(conn); conn.close()

    code = load_strategy(BULL)
    mod = code_to_module(code)
    res = run(mod, "FCF", 30)                 # tx 30bp, 설정은 fcf_base_compare와 동일
    hbd = res["holdings_by_date"]             # {date: [(code, score, weight, mcap), ...]}
    dates = sorted(hbd)

    # ── (1) 최근 리밸 보유종목 ────────────────────────────────
    last = dates[-1]
    rows = [{"date": last, "code": c, "name": NAMES.get(c, ""),
             "weight": w, "score": s} for c, s, w, _m in hbd[last]]
    df = pd.DataFrame(rows).sort_values("weight", ascending=False)
    df.to_csv(OUT / "fcf_holdings_last.csv", index=False, encoding="utf-8-sig")
    print(f"\n■ 최근 리밸 {last} — {len(df)}종목")
    for i, r in enumerate(df.itertuples(), 1):
        print(f"  {i:2d}. {r.code} {r.name:<12} {r.weight*100:5.2f}%")

    # ── (2) 월별 턴오버 ──────────────────────────────────────
    tr = []
    for prev, cur in zip(dates, dates[1:]):
        wp = {c: w for c, _s, w, _m in hbd[prev]}
        wc = {c: w for c, _s, w, _m in hbd[cur]}
        codes = set(wp) | set(wc)
        # 이름 기준 교체율 + 비중 기준 단방향 턴오버(=Σ|Δw|/2)
        name_to = len(set(wc) - set(wp)) / max(len(wc), 1)
        w_to = sum(abs(wc.get(c, 0) - wp.get(c, 0)) for c in codes) / 2
        tr.append({"date": cur, "n_new": len(set(wc) - set(wp)),
                   "n_hold": len(wc), "name_turnover": name_to,
                   "weight_turnover": w_to})
    t = pd.DataFrame(tr)
    t.to_csv(OUT / "fcf_turnover.csv", index=False, encoding="utf-8-sig")

    print(f"\n■ 월별 턴오버 (n={len(t)})")
    for col in ["name_turnover", "weight_turnover"]:
        s = t[col]
        print(f"  {col:16s} 평균 {s.mean()*100:5.1f}%  중앙 {s.median()*100:5.1f}%  "
              f"min {s.min()*100:4.1f}%  max {s.max()*100:5.1f}%  "
              f"연환산(×12) {s.mean()*12*100:6.0f}%")
    print("\n  최근 12개월:")
    for r in t.tail(12).itertuples():
        print(f"    {r.date}  신규 {r.n_new:2d}/{r.n_hold:2d}  "
              f"이름 {r.name_turnover*100:5.1f}%  비중 {r.weight_turnover*100:5.1f}%")


if __name__ == "__main__":
    main()
