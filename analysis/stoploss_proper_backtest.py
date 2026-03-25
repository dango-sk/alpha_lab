"""
손절 방식 비교 — 턴오버+거래비용 정확 반영
A0 전략 (KOSPI, top30, cap10%, 30bp, monthly)
손절 기준: -15%
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import BACKTEST_CONFIG
from scripts.step7_backtest import run_backtest, make_engine_selector

SCENARIOS = [
    {"label": "손절 없음",       "stop_loss_enabled": False, "stop_loss_mode": "sell"},
    {"label": "전량 매도",       "stop_loss_enabled": True,  "stop_loss_mode": "sell"},
    {"label": "비중 50% 축소",   "stop_loss_enabled": True,  "stop_loss_mode": "reduce"},
]

def run():
    results = []
    for sc in SCENARIOS:
        BACKTEST_CONFIG["stop_loss_enabled"] = sc["stop_loss_enabled"]
        BACKTEST_CONFIG["stop_loss_pct"]     = 15
        BACKTEST_CONFIG["stop_loss_mode"]    = sc["stop_loss_mode"]

        selector = make_engine_selector("A0", rebal_type="monthly", min_market_cap=0)
        r = run_backtest("A0", stock_selector=selector, rebal_type="monthly")
        if r is None:
            print(f"[ERROR] {sc['label']} 실패")
            continue

        results.append({
            "label":        sc["label"],
            "total_return": r["total_return"],
            "cagr":         r["cagr"],
            "sharpe":       r["sharpe"],
            "mdd":          r["mdd"],
            "avg_turnover": r["avg_turnover"],
        })
        print(f"[{sc['label']}] 총수익 {r['total_return']*100:.1f}%  CAGR {r['cagr']*100:.1f}%  Sharpe {r['sharpe']:.2f}  MDD {r['mdd']*100:.1f}%  평균턴오버 {r['avg_turnover']*100:.1f}%")

    print()
    print(f"{'':20s} {'총수익':>8} {'CAGR':>7} {'Sharpe':>7} {'MDD':>7} {'평균턴오버':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['label']:20s} {r['total_return']*100:7.1f}%  {r['cagr']*100:6.1f}%  {r['sharpe']:6.2f}  {r['mdd']*100:6.1f}%  {r['avg_turnover']*100:8.1f}%")

if __name__ == "__main__":
    run()
