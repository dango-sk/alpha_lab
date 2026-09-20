"""
analysis/fcf_diet_churn.py  (실험 스크립트, production 미수정)

[post-OOS diagnostic] 후보별 경계 종목 반복 진입·이탈 진단.
  production selector만 재사용(가격 조회 없음)하므로 빠르다.

실행: .venv/bin/python analysis/fcf_diet_churn.py
산출: output/diet_churn.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
load_dotenv(REPO / ".env"); sys.path.insert(0, str(REPO / "analysis"))

from lib.data import load_strategy                                        # noqa: E402
from lib.factor_engine import code_to_module, score_stocks_from_strategy  # noqa: E402
from step7_backtest import get_universe_stocks, get_db, get_rebalance_dates  # noqa: E402
import fcf_diet_compare as dc                                             # noqa: E402

OUT = REPO / "output"


def main():
    conn = get_db()
    sd = load_strategy(dc.STRATEGY, "monthly", "KOSPI")
    mods = {"BASE": code_to_module(sd["code"]),
            "DIET-3": code_to_module(dc.drop_factors(sd["code"], dc.DROP3)[0]),
            "DIET-4": code_to_module(dc.drop_factors(sd["code"], dc.DROP4)[0])}
    dates = get_rebalance_dates(conn, "monthly")
    rows = []
    for name, mod in mods.items():
        held, ranks = [], []
        for d in dates:
            uni = set(get_universe_stocks(conn, d, "monthly"))
            if not uni: continue
            cands = [(c, s) for c, s in score_stocks_from_strategy(conn, d, mod) if c in uni]
            rk = {c: i + 1 for i, (c, _) in enumerate(cands)}
            held.append((d[:7], {c for c, _ in cands[:30]}, rk))
        for i, (ym, cur, rk) in enumerate(held):
            boundary = [c for c in cur if 25 <= rk.get(c, 999) <= 35]
            exits = held[i - 1][1] - cur if i else set()
            fut = set().union(*[held[j][1] for j in range(i + 1, min(i + 4, len(held)))]) if i and i + 1 < len(held) else set()
            rows.append({"spec": name, "ym": ym, "n_boundary_25_35": len(boundary),
                         "boundary_share": len(boundary) / max(len(cur), 1),
                         "n_exit": len(exits),
                         "reentry_within_3m": len(exits & fut) / max(len(exits), 1) if exits else np.nan})
        print(f"  {name} 완료", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "diet_churn.csv", index=False, encoding="utf-8-sig")
    print(df.groupby("spec")[["n_boundary_25_35", "boundary_share", "n_exit", "reentry_within_3m"]]
          .mean().round(4).to_string())


if __name__ == "__main__":
    main()
