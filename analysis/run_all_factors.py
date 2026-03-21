"""
전체 팩터 롱숏 + 전략 비교 한번에 실행
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.factor_longshort import run_factor_longshort, FACTOR_DEFS
from analysis.strategy_backtest import run_comparison

def main():
    # 1. 전체 팩터 롱숏
    print("\n" + "=" * 70)
    print("  PART 1: 전체 팩터 롱숏 백테스트")
    print("=" * 70)

    all_results = {}
    for factor_name in FACTOR_DEFS:
        result = run_factor_longshort(factor_name)
        if result:
            all_results[factor_name] = {
                "label": result["label"],
                "sharpe": result["sharpe"],
                "annualized": result["annualized"],
                "cumulative_ls": result["cumulative_ls"],
                "t_stat": result["t_stat"],
                "p_value": result["p_value"],
                "win_rate": result["win_rate"],
                "mean_monthly": result["mean_monthly"],
            }

    # 요약 테이블
    print("\n" + "=" * 90)
    print("  팩터 롱숏 요약 (Sharpe 순)")
    print("=" * 90)
    print(f"  {'팩터':<35} {'Sharpe':>8} {'연환산':>8} {'누적L/S':>8} {'t-stat':>8} {'p-value':>8} {'승률':>6}")
    print(f"  {'─'*85}")

    sorted_factors = sorted(all_results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    for name, r in sorted_factors:
        sig = "**" if r["p_value"] < 0.05 else "  "
        print(f"  {r['label']:<35} {r['sharpe']:>+7.3f} {r['annualized']*100:>+7.1f}% {r['cumulative_ls']*100:>+7.1f}% {r['t_stat']:>+7.3f} {r['p_value']:>8.4f}{sig} {r['win_rate']*100:>5.1f}%")

    print(f"  {'─'*85}")
    print("  ** = 통계적으로 유의 (p < 0.05)")

    # 저장
    save_path = ROOT / "cache" / "all_factor_summary.json"
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {save_path}")

    # 2. 전략 비교
    print("\n" + "=" * 70)
    print("  PART 2: 전략 비교 백테스트")
    print("=" * 70)
    run_comparison()


if __name__ == "__main__":
    main()
