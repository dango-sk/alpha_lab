"""
백테스트 3경로 일관성 sanity check.

레짐 조합에 bull=bear=같은 전략을 넣으면 단독 백테스트와 동일한 결과가 나와야 한다.
파라미터 누락(예: min_market_cap)이 생기면 여기서 잡힌다.

사용법:
    python -m analysis.sanity_backtest_paths FCF_YIELD추가전략
    python -m analysis.sanity_backtest_paths   # 기본: A1
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from lib.data import run_regime_combo_backtest, load_strategy


def main(strategy_name: str, regime_mode: str = "cycle"):
    print(f"\n{'=' * 60}")
    print(f"Sanity check: 단독 vs 조합(bull=bear={strategy_name})")
    print(f"regime_mode={regime_mode}")
    print('=' * 60)

    # 1. 단독 캐시
    single = load_strategy(strategy_name, rebal_type="monthly", universe="KOSPI")
    single_res = single.get("results", {})
    if not single_res:
        print(f"[ERROR] 단독 캐시 없음: {strategy_name}")
        return
    single_cagr = single_res.get("cagr")
    single_total = single_res.get("total_return")
    single_mdd = single_res.get("mdd")
    print(f"\n[단독 캐시]")
    print(f"  total_return: {single_total}")
    print(f"  CAGR:         {single_cagr}")
    print(f"  MDD:          {single_mdd}")

    # 2. 조합 backtest with bull=bear=same
    print(f"\n[조합 재실행 중...]")
    combo = run_regime_combo_backtest(
        bull_key=strategy_name,
        bear_key=strategy_name,
        universe="KOSPI",
        rebal_type="monthly",
        regime_mode=regime_mode,
    )
    if not combo or "REGIME_COMBO" not in combo:
        print(f"[ERROR] 조합 결과 없음")
        return
    combo_res = combo["REGIME_COMBO"]
    combo_cagr = combo_res.get("cagr")
    combo_total = combo_res.get("total_return")
    combo_mdd = combo_res.get("mdd")
    print(f"\n[조합 (bull=bear={strategy_name})]")
    print(f"  total_return: {combo_total}")
    print(f"  CAGR:         {combo_cagr}")
    print(f"  MDD:          {combo_mdd}")

    # 3. Diff
    print(f"\n[Diff]")
    def fmt(a, b):
        if a is None or b is None:
            return "N/A"
        try:
            return f"{float(b) - float(a):+.4f}"
        except (TypeError, ValueError):
            return f"{b} vs {a}"
    print(f"  Δ total_return: {fmt(single_total, combo_total)}")
    print(f"  Δ CAGR:         {fmt(single_cagr, combo_cagr)}")
    print(f"  Δ MDD:          {fmt(single_mdd, combo_mdd)}")

    # 4. 월별 수익률 비교 (있으면)
    single_monthly = single_res.get("monthly_returns") or single_res.get("returns")
    combo_monthly = combo_res.get("monthly_returns") or combo_res.get("returns")
    if single_monthly and combo_monthly and isinstance(single_monthly, list) and isinstance(combo_monthly, list):
        n = min(len(single_monthly), len(combo_monthly))
        diffs = []
        for i in range(n):
            try:
                d = float(combo_monthly[i]) - float(single_monthly[i])
                if abs(d) > 1e-6:
                    diffs.append((i, single_monthly[i], combo_monthly[i], d))
            except (TypeError, ValueError):
                pass
        print(f"\n[월별 수익률 diff (|Δ| > 1e-6)]")
        print(f"  총 {n}개월 중 {len(diffs)}개월 어긋남")
        for i, s, c, d in diffs[:10]:
            print(f"    [{i:3d}] single={s:.6f}  combo={c:.6f}  Δ={d:+.6f}")
        if len(diffs) > 10:
            print(f"    ... ({len(diffs) - 10}개 더)")
    else:
        print(f"\n[월별 수익률] 비교 불가 (캐시에 monthly_returns 없음)")


if __name__ == "__main__":
    strat = sys.argv[1] if len(sys.argv) > 1 else "A1"
    mode = sys.argv[2] if len(sys.argv) > 2 else "cycle"
    main(strat, mode)
