"""
FCF_YIELD 레짐 조합 백테스트

Bull 장세 → FCF_YIELD추가전략
Bear 장세 → FCF_YIELD_BEAR전략

실행: python analysis/fcf_yield_regime_backtest.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.data import run_regime_combo_backtest
from lib.charts import cumulative_return_chart

BULL_KEY    = "FCF_YIELD추가전략"
BEAR_KEY    = "FCF_YIELD_BEAR전략"
REGIME_MODE = "ai"
MA_WINDOW   = 50


def _save_json(results: dict, out_dir: Path) -> Path:
    r  = results.get("REGIME_COMBO", {})
    bm = results.get("KOSPI", {})

    payload = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "bull_key": BULL_KEY,
            "bear_key": BEAR_KEY,
            "regime_mode": REGIME_MODE,
        },
        "REGIME_COMBO": {
            "total_return": r.get("total_return"),
            "cagr":         r.get("cagr"),
            "mdd":          r.get("mdd"),
            "sharpe":       r.get("sharpe"),
        },
        "KOSPI": {
            "total_return": bm.get("total_return"),
            "cagr":         bm.get("cagr"),
            "mdd":          bm.get("mdd"),
            "sharpe":       bm.get("sharpe"),
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"fcf_yield_regime_backtest_{ts}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    results = run_regime_combo_backtest(
        bull_key=BULL_KEY,
        bear_key=BEAR_KEY,
        regime_mode=REGIME_MODE,
        ma_window=MA_WINDOW,
    )

    if not results or "error" in results:
        print(f"오류: {results}")
        return

    r = results.get("REGIME_COMBO")
    bm = results.get("KOSPI")

    print("\n" + "=" * 70)
    print(f"  {'전략':<40}  {'누적':>7}  {'CAGR':>7}  {'MDD':>6}  {'Sharpe':>6}")
    print("  " + "-" * 68)
    for label, res in [(f"레짐조합 (Bull:{BULL_KEY[:10]} / Bear:{BEAR_KEY[:10]})", r), ("KODEX 200", bm)]:
        if not res:
            continue
        print(
            f"  {label:<40}  "
            f"누적 {res['total_return']:>+7.1%}  "
            f"CAGR {res['cagr']:>+7.1%}  "
            f"MDD {res['mdd']:>6.1%}  "
            f"Sharpe {res['sharpe']:>5.2f}"
        )
    print("=" * 70)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    json_path = _save_json(results, out_dir)
    print(f"\n[JSON]  Saved: {json_path}")

    fig = cumulative_return_chart(results)
    fig.update_layout(
        title="Cumulative Return (%)",
        yaxis_title="Return (%)",
    )
    out_path = out_dir / "fcf_yield_regime_backtest.html"
    fig.write_html(str(out_path))
    print(f"[CHART] Saved: {out_path}")


if __name__ == "__main__":
    main()
