"""
C method stop-loss sensitivity analysis (fine-grained)
  Stop-loss rates: 13% / 14% / 15% (baseline) / 16% / 17%

Identical to compare_stoploss_sensitivity.py C method design:
  - Bull: stop_loss_pct=9999  (no stop, peak tracking only)
  - Bear: carry_over inherited, stop-loss applied at specified rate
"""
import sys, json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _run_c_with_sl(bull_key, bear_key, sl_pct):
    """Run C-method (natural carry-over design) with the given stop-loss rate."""
    from lib.data import run_regime_combo_backtest
    import step7_backtest as bt

    _orig_calc_plain = bt.calc_portfolio_return
    _orig_calc_sl    = bt.calc_portfolio_return_with_stoploss
    _shared_carry    = [{}]

    def _calc_plain_nat(conn, stocks, start_date, end_date):
        ret, _evs, new_carry = _orig_calc_sl(
            conn, stocks, start_date, end_date,
            stop_loss_pct=9999, stop_loss_mode="sell", stop_loss_basis="peak",
            carry_over=_shared_carry[0],
        )
        _shared_carry[0] = new_carry
        return ret

    def _calc_sl_nat(conn, stocks, start_date, end_date, **kw):
        kw["carry_over"]    = _shared_carry[0]
        kw["stop_loss_pct"] = sl_pct
        ret, evs, new_carry = _orig_calc_sl(conn, stocks, start_date, end_date, **kw)
        _shared_carry[0] = new_carry
        return ret, evs, new_carry

    _shared_carry[0] = {}
    bt.calc_portfolio_return               = _calc_plain_nat
    bt.calc_portfolio_return_with_stoploss = _calc_sl_nat
    try:
        result = run_regime_combo_backtest(
            bull_key=bull_key, bear_key=bear_key,
            universe="KOSPI", rebal_type="monthly", regime_mode="ai",
        )
    finally:
        bt.calc_portfolio_return               = _orig_calc_plain
        bt.calc_portfolio_return_with_stoploss = _orig_calc_sl

    r = result.get("REGIME_COMBO", {})
    if "mdd" in r:
        r["mdd"] = -abs(r["mdd"])
    return r


def run_comparison():
    bull_key = "수정전략_코스피_cap30%_top30_tx30bp_월간"
    bear_key  = "cap30%_손절율15%(고점)"
    sl_rates  = [13, 14, 15, 16, 17]

    results = {}
    for sl in sl_rates:
        print(f"\n{'='*60}")
        print(f"  C method  stop-loss {sl}%  running...")
        print(f"{'='*60}")
        results[sl] = _run_c_with_sl(bull_key, bear_key, sl)

    # ── Performance comparison table ──
    print(f"\n{'='*80}")
    print("  C Method - Stop Loss Rate Sensitivity (fine-grained: 13~17%)")
    print(f"  Bull: {bull_key}")
    print(f"  Bear: {bear_key}")
    print(f"{'='*80}")

    header = f"  {'Metric':<20}" + "".join(f"  {str(sl)+'%':>10}" for sl in sl_rates)
    print(header)
    print(f"  {'-'*70}")

    for key, label, fmt in [
        ("total_return",       "Total Return",       ".1%"),
        ("cagr",               "CAGR",               ".1%"),
        ("mdd",                "MDD",                ".1%"),
        ("sharpe",             "Sharpe",             ".2f"),
        ("avg_monthly_return", "Avg Monthly Ret",    ".2%"),
        ("monthly_std",        "Monthly Std",        ".2%"),
    ]:
        row = f"  {label:<20}"
        for sl in sl_rates:
            v = results[sl].get(key, 0)
            row += f"  {format(v, fmt):>10}"
        print(row)

    # ── Top months with largest spread across rates ──
    dates   = results[15].get("rebalance_dates", [])
    monthly = {sl: results[sl].get("monthly_returns", []) for sl in sl_rates}
    lengths = [len(monthly[sl]) for sl in sl_rates]

    if dates and all(l == len(dates) for l in lengths):
        rows = []
        for i, d in enumerate(dates):
            spread = max(monthly[sl][i] for sl in sl_rates) - min(monthly[sl][i] for sl in sl_rates)
            rows.append({"date": d, **{sl: monthly[sl][i] for sl in sl_rates}, "spread": spread})
        rows.sort(key=lambda x: x["spread"], reverse=True)

        print(f"\n  Top 5 months by return spread across stop-loss rates:")
        print(f"  {'Date':>12}" + "".join(f"  {str(sl)+'%':>9}" for sl in sl_rates))
        for r in rows[:5]:
            print(f"  {r['date']:>12}" + "".join(f"  {r[sl]*100:>+8.2f}%" for sl in sl_rates))

        diff_months = [r for r in rows if r["spread"] > 0.001]
        print(f"\n  Months with difference: {len(diff_months)} / {len(rows)} total")

    # ── Output paths ──
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Cumulative return chart ──
    colors   = ["#E53935", "#FB8C00", "#43A047", "#1E88E5", "#8E24AA"]
    rb_dates = results[15].get("rebalance_dates", [])
    monthly  = {sl: results[sl].get("monthly_returns", []) for sl in sl_rates}

    if rb_dates and all(monthly[sl] for sl in sl_rates):
        _, ax = plt.subplots(figsize=(13, 6))
        for sl, color in zip(sl_rates, colors):
            pv    = np.cumprod(1 + np.array(monthly[sl]))
            label = f"{sl}% (baseline)" if sl == 15 else f"{sl}%"
            lw    = 2.4 if sl == 15 else 1.6
            ax.plot(range(len(pv)), pv, label=label, color=color, linewidth=lw)

        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        step     = max(1, len(rb_dates) // 10)
        tick_pos = list(range(0, len(rb_dates), step))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([rb_dates[j][:7] for j in tick_pos], rotation=45, ha="right", fontsize=8)
        ax.set_title("C Method - Fine-Grained Stop Loss Comparison (13~17%)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative Return (1.0 = initial)")
        ax.legend(title="Stop Loss Rate", fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        chart_path = out_dir / f"stoploss_sensitivity2_{stamp}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"\n  Chart saved: {chart_path.relative_to(Path(__file__).parent.parent)}")

    # ── JSON save ──
    out_path = out_dir / f"stoploss_sensitivity2_{stamp}.json"

    metric_keys = ["total_return", "cagr", "mdd", "sharpe",
                   "avg_monthly_return", "monthly_std",
                   "monthly_returns", "rebalance_dates", "portfolio_values"]
    payload = {
        "meta": {
            "bull_key":     bull_key,
            "bear_key":     bear_key,
            "sl_rates":     sl_rates,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "results": {
            str(sl): {k: results[sl].get(k) for k in metric_keys}
            for sl in sl_rates
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  Results saved: {out_path.relative_to(Path(__file__).parent.parent)}")


if __name__ == "__main__":
    run_comparison()
