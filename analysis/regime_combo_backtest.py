"""
레짐별 전략 조합 백테스트

KOSPI 200 50일 이동평균 기준으로 상승/하락장을 구분하고,
각 장세마다 다른 전략을 적용했을 때의 성과를 비교.

예) 상승장 → A0, 하락장 → ATT2 조합
"""
import sys
import numpy as np
from pathlib import Path
from itertools import product

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lib.db import get_conn
from lib.data import load_all_results, STRATEGY_LABELS


def get_kospi200_50d_signal(conn, rb_dates: list) -> list:
    """각 리밸런싱 시점에서 KOSPI 200이 50일 이동평균 상회/하회 판단.

    월별 리밸런싱 날짜 기준으로, 직전 50 거래일 일별 종가 평균과 비교.
    Returns: ['bull' or 'bear'] — rb_dates와 동일 길이
    """
    signals = []
    for d in rb_dates:
        rows = conn.execute("""
            SELECT close FROM daily_price
            WHERE stock_code = '069500'
              AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 51
        """, (d,)).fetchall()

        prices = [r[0] for r in rows if r[0]]
        if len(prices) < 51:
            signals.append("bull")
            continue

        current = prices[0]          # 가장 최근 (= 리밸런싱 당일)
        ma50 = np.mean(prices[1:51]) # 직전 50일 평균
        signals.append("bear" if current < ma50 else "bull")

    return signals


def calc_stats(monthly_returns: list) -> dict:
    rets = np.array(monthly_returns)
    if len(rets) == 0:
        return {}
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    total = cum[-1] - 1
    n_years = len(rets) / 12
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = rets.mean() / rets.std() * np.sqrt(12) if rets.std() > 0 else 0
    mdd = float(dd.min())
    win_rate = float((rets > 0).mean())
    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
    }


def main():
    conn = get_conn()

    # ── 1. 기존 백테스트 결과 로드 ──
    print("백테스트 결과 로딩 중...")
    all_results = load_all_results(universe="KOSPI", rebal_type="monthly")

    # 전략 키 목록 (벤치마크 제외)
    strat_keys = [k for k in all_results if k not in ("KOSPI", "KOSDAQ")]
    if not strat_keys:
        print("저장된 전략 결과가 없습니다. 파이프라인을 먼저 실행해주세요.")
        return

    # 리밸런싱 날짜는 첫 번째 전략 기준
    ref_key = strat_keys[0]
    rb_dates = all_results[ref_key]["rebalance_dates"]
    n = len(rb_dates) - 1  # 월 수

    print(f"전략: {[STRATEGY_LABELS.get(k, k) for k in strat_keys]}")
    print(f"기간: {rb_dates[0]} ~ {rb_dates[-1]} ({n}개월)")

    # ── 2. 50일 MA 레짐 신호 ──
    print("\nKOSPI 200 50일 MA 신호 계산 중...")
    signals = get_kospi200_50d_signal(conn, rb_dates[:-1])  # 각 시작일 기준
    bull_months = sum(1 for s in signals if s == "bull")
    bear_months = sum(1 for s in signals if s == "bear")
    print(f"  상승장(bull): {bull_months}개월 ({bull_months/n:.0%})")
    print(f"  하락장(bear): {bear_months}개월 ({bear_months/n:.0%})")

    # 구간 시각화 (간단히)
    bear_periods = [rb_dates[i] for i, s in enumerate(signals) if s == "bear"]
    print(f"  하락장 시점 (첫 8개): {', '.join(bear_periods[:8])}{'...' if len(bear_periods) > 8 else ''}")

    # ── 3. 각 전략 월별 수익률 배열 준비 ──
    monthly_rets = {}
    for k in strat_keys:
        r = all_results[k]
        rets = r["monthly_returns"]
        # rb_dates 길이 맞추기 (n개월)
        monthly_rets[k] = np.array(rets[:n])

    kospi_rets = np.array(all_results.get("KOSPI", {}).get("monthly_returns", [0]*n)[:n])

    # ── 4. 단일 전략 기준 성과 ──
    print("\n" + "=" * 70)
    print("▶ 단일 전략 성과 (기준)")
    print("=" * 70)
    print(f"{'전략':<28} {'수익률':>8} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'승률':>6}")
    print("-" * 64)

    single_stats = {}
    for k in strat_keys:
        s = calc_stats(list(monthly_rets[k]))
        single_stats[k] = s
        label = STRATEGY_LABELS.get(k, k)[:26]
        print(f"{label:<28} {s['total_return']:>+7.1%} {s['cagr']:>+7.1%} "
              f"{s['sharpe']:>7.2f} {s['mdd']:>7.1%} {s['win_rate']:>5.0%}")

    kospi_stats = calc_stats(list(kospi_rets))
    print(f"{'KOSPI 200':<28} {kospi_stats['total_return']:>+7.1%} {kospi_stats['cagr']:>+7.1%} "
          f"{kospi_stats['sharpe']:>7.2f} {kospi_stats['mdd']:>7.1%} {kospi_stats['win_rate']:>5.0%}")

    # ── 5. 레짐 조합 백테스트 ──
    print("\n" + "=" * 70)
    print("▶ 레짐 조합 백테스트 (상승장 전략 × 하락장 전략)")
    print("  KOSPI 200 50일 MA 기준 레짐 구분")
    print("=" * 70)

    combo_results = {}
    for bull_key, bear_key in product(strat_keys, strat_keys):
        combined = []
        for i, sig in enumerate(signals):
            if sig == "bull":
                combined.append(float(monthly_rets[bull_key][i]))
            else:
                combined.append(float(monthly_rets[bear_key][i]))

        s = calc_stats(combined)
        combo_results[(bull_key, bear_key)] = s

    # 결과 정렬 (Sharpe 기준)
    sorted_combos = sorted(combo_results.items(), key=lambda x: x[1]["sharpe"], reverse=True)

    bull_label_w = max(len(STRATEGY_LABELS.get(k, k)) for k in strat_keys) + 2
    bear_label_w = bull_label_w

    print(f"\n{'상승장 전략':<{bull_label_w}} {'하락장 전략':<{bear_label_w}} "
          f"{'수익률':>8} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'승률':>6}")
    print("-" * (bull_label_w + bear_label_w + 46))

    for (bk, dk), s in sorted_combos:
        bl = STRATEGY_LABELS.get(bk, bk)[:bull_label_w-1]
        dl = STRATEGY_LABELS.get(dk, dk)[:bear_label_w-1]
        marker = " ★" if bk == dk else ""  # 단일 전략과 동일한 조합 표시
        print(f"{bl:<{bull_label_w}} {dl:<{bear_label_w}} "
              f"{s['total_return']:>+7.1%} {s['cagr']:>+7.1%} "
              f"{s['sharpe']:>7.2f} {s['mdd']:>7.1%} {s['win_rate']:>5.0%}{marker}")

    # ── 6. 베스트 조합 상세 ──
    best_combo, best_stats = sorted_combos[0]
    bull_k, bear_k = best_combo
    print(f"\n{'='*70}")
    print(f"★ Best 조합: 상승장 [{STRATEGY_LABELS.get(bull_k, bull_k)}] "
          f"× 하락장 [{STRATEGY_LABELS.get(bear_k, bear_k)}]")
    print(f"  CAGR: {best_stats['cagr']:+.1%}, Sharpe: {best_stats['sharpe']:.2f}, "
          f"MDD: {best_stats['mdd']:.1%}")

    # 하락장 구간 성과 비교
    bear_indices = [i for i, s in enumerate(signals) if s == "bear"]
    if bear_indices:
        print(f"\n하락장 구간 ({len(bear_indices)}개월) 월평균 수익률 비교:")
        for k in strat_keys:
            bear_rets_k = [float(monthly_rets[k][i]) for i in bear_indices]
            print(f"  {STRATEGY_LABELS.get(k, k)}: {np.mean(bear_rets_k)*100:+.2f}%")
        kospi_bear = [float(kospi_rets[i]) for i in bear_indices]
        print(f"  KOSPI 200: {np.mean(kospi_bear)*100:+.2f}%")

    bull_indices = [i for i, s in enumerate(signals) if s == "bull"]
    if bull_indices:
        print(f"\n상승장 구간 ({len(bull_indices)}개월) 월평균 수익률 비교:")
        for k in strat_keys:
            bull_rets_k = [float(monthly_rets[k][i]) for i in bull_indices]
            print(f"  {STRATEGY_LABELS.get(k, k)}: {np.mean(bull_rets_k)*100:+.2f}%")
        kospi_bull = [float(kospi_rets[i]) for i in bull_indices]
        print(f"  KOSPI 200: {np.mean(kospi_bull)*100:+.2f}%")

    conn.close()
    print("\n완료!")


if __name__ == "__main__":
    main()
