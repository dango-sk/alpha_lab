"""
Step 8: 강건성 검증 (4개 밸류 전략)

검증:
  1. In-Sample vs Out-of-Sample 분할 비교
  2. 통계적 유의성 (paired t-test + bootstrap 95% CI)
  3. 롤링 윈도우 (24개월) 초과수익 일관성
"""
import json
import sqlite3
import sys
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from config.settings import DB_PATH, BACKTEST_CONFIG, CACHE_DIR
from step7_backtest import (
    run_backtest, get_db, get_monthly_rebalance_dates,
    calc_etf_return, calc_etf_monthly_returns, STRATEGIES,
    _numpy_to_python,
)

# ─── 한글 폰트 ───
for fname in fm.findSystemFonts():
    if any(k in fname for k in ["AppleGothic", "NanumGothic", "Malgun", "NotoSansCJK"]):
        matplotlib.rc("font", family=fm.FontProperties(fname=fname).get_name())
        break
matplotlib.rcParams["axes.unicode_minus"] = False

BASELINE_KEY = "ATT2"
N_BOOTSTRAP = 10_000
ROLLING_WINDOW = 24


@contextmanager
def _patch_backtest_period(start, end):
    original_start = BACKTEST_CONFIG["start"]
    original_end = BACKTEST_CONFIG["end"]
    try:
        BACKTEST_CONFIG["start"] = start
        BACKTEST_CONFIG["end"] = end
        yield
    finally:
        BACKTEST_CONFIG["start"] = original_start
        BACKTEST_CONFIG["end"] = original_end


# ═══════════════════════════════════════════════════════
# 테스트 1: IS/OOS 분할
# ═══════════════════════════════════════════════════════

def test_is_oos_split():
    is_start = BACKTEST_CONFIG["start"]
    is_end = BACKTEST_CONFIG["insample_end"]
    oos_start = BACKTEST_CONFIG["oos_start"]
    oos_end = BACKTEST_CONFIG["end"]

    is_results = {}
    oos_results = {}

    for key, strat, desc in STRATEGIES:
        print(f"    IS: {key}...")
        with _patch_backtest_period(is_start, is_end):
            r = run_backtest(strat)
            if r:
                r["strategy"] = key
                is_results[key] = r

        print(f"    OOS: {key}...")
        with _patch_backtest_period(oos_start, oos_end):
            r = run_backtest(strat)
            if r:
                r["strategy"] = key
                oos_results[key] = r

    # 벤치마크
    bm_results = {"is": {}, "oos": {}}
    conn = get_db()
    is_ret = calc_etf_return(conn, "KS200", is_start, is_end)
    oos_ret = calc_etf_return(conn, "KS200", oos_start, oos_end)
    conn.close()

    is_months = len(is_results.get(BASELINE_KEY, {}).get("monthly_returns", []))
    oos_months = len(oos_results.get(BASELINE_KEY, {}).get("monthly_returns", []))

    if is_ret is not None:
        bm_results["is"]["KOSPI"] = {
            "total_return": is_ret,
            "cagr": ((1 + is_ret) ** (12.0 / max(is_months, 1)) - 1.0),
            "name": "KOSPI 200",
        }
    if oos_ret is not None:
        bm_results["oos"]["KOSPI"] = {
            "total_return": oos_ret,
            "cagr": ((1 + oos_ret) ** (12.0 / max(oos_months, 1)) - 1.0),
            "name": "KOSPI 200",
        }

    return {"is_results": is_results, "oos_results": oos_results, "benchmarks": bm_results}


# ═══════════════════════════════════════════════════════
# 테스트 2: 통계적 유의성
# ═══════════════════════════════════════════════════════

def test_statistical_significance():
    full_results = {}
    for key, strat, desc in STRATEGIES:
        print(f"    전체기간: {key}...")
        r = run_backtest(strat)
        if r:
            r["strategy"] = key
            full_results[key] = r

    # vs 벤치마크
    conn = get_db()
    rebalance_dates = full_results[BASELINE_KEY]["rebalance_dates"]
    bm_monthly = np.array(calc_etf_monthly_returns(conn, "KS200", rebalance_dates))
    conn.close()

    bm_significance = {}
    rng = np.random.default_rng(42)

    for key, _, desc in STRATEGIES:
        if key not in full_results:
            continue
        strat_rets = np.array(full_results[key]["monthly_returns"])
        n = min(len(strat_rets), len(bm_monthly))
        diff = strat_rets[:n] - bm_monthly[:n]

        t_stat, p_value = stats.ttest_rel(strat_rets[:n], bm_monthly[:n])
        boot_means = np.array([
            rng.choice(diff, size=len(diff), replace=True).mean()
            for _ in range(N_BOOTSTRAP)
        ])
        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)

        bm_significance[key] = {
            "n_months": n,
            "mean_diff": diff.mean(),
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": ci_lower > 0,
            "boot_means": boot_means,
            "win_rate": (boot_means > 0).mean(),
        }

    return {"full_results": full_results, "bm_significance": bm_significance}


# ═══════════════════════════════════════════════════════
# 테스트 3: 롤링 윈도우
# ═══════════════════════════════════════════════════════

def test_rolling_window(full_results):
    """24개월 슬라이딩 윈도우: 각 전략 vs KOSPI 200"""
    conn = get_db()
    rebalance_dates = full_results[BASELINE_KEY]["rebalance_dates"]
    bm_monthly = np.array(calc_etf_monthly_returns(conn, "KS200", rebalance_dates))
    conn.close()

    rolling_all = {}
    for key, _, desc in STRATEGIES:
        if key not in full_results:
            continue
        strat_monthly = np.array(full_results[key]["monthly_returns"])
        n = min(len(strat_monthly), len(bm_monthly))
        if n < ROLLING_WINDOW:
            continue

        rolling_results = []
        for i in range(n - ROLLING_WINDOW + 1):
            w_strat = strat_monthly[i:i + ROLLING_WINDOW]
            w_bm = bm_monthly[i:i + ROLLING_WINDOW]
            strat_cum = np.prod(1 + w_strat) - 1
            bm_cum = np.prod(1 + w_bm) - 1
            excess = strat_cum - bm_cum

            start_date = rebalance_dates[i] if i < len(rebalance_dates) else ""
            end_idx = i + ROLLING_WINDOW
            end_date = rebalance_dates[end_idx] if end_idx < len(rebalance_dates) else ""
            rolling_results.append({
                "start_date": start_date,
                "end_date": end_date,
                "excess_return": excess,
            })

        positive = sum(1 for r in rolling_results if r["excess_return"] > 0)
        rolling_all[key] = {
            "total_windows": len(rolling_results),
            "positive_windows": positive,
            "win_rate": positive / len(rolling_results),
            "rolling_data": rolling_results,
        }

    return rolling_all


# ═══════════════════════════════════════════════════════
# 결과 출력
# ═══════════════════════════════════════════════════════

def show_results(is_oos_data, stat_data, rolling_all):
    strategy_descs = {k: d for k, _, d in STRATEGIES}

    # ── IS/OOS ──
    print("\n" + "=" * 80)
    print("테스트 1: In-Sample vs Out-of-Sample")
    print(f"   IS:  {BACKTEST_CONFIG['start']} ~ {BACKTEST_CONFIG['insample_end']}")
    print(f"   OOS: {BACKTEST_CONFIG['oos_start']} ~ {BACKTEST_CONFIG['end']}")
    print("=" * 80)

    print(f"\n  {'전략':<34} | {'IS 수익률':>10} {'IS Sharpe':>10} | {'OOS 수익률':>10} {'OOS Sharpe':>10}")
    print(f"  {'─'*80}")

    for key, _, desc in STRATEGIES:
        is_r = is_oos_data["is_results"].get(key, {})
        oos_r = is_oos_data["oos_results"].get(key, {})
        is_ret = f"{is_r['total_return']:+.1%}" if is_r else "N/A"
        is_sh = f"{is_r['sharpe']:.2f}" if is_r else "N/A"
        oos_ret = f"{oos_r['total_return']:+.1%}" if oos_r else "N/A"
        oos_sh = f"{oos_r['sharpe']:.2f}" if oos_r else "N/A"
        print(f"  {desc:<34} | {is_ret:>10} {is_sh:>10} | {oos_ret:>10} {oos_sh:>10}")

    bm = is_oos_data.get("benchmarks", {})
    is_bm = bm.get("is", {}).get("KOSPI", {})
    oos_bm = bm.get("oos", {}).get("KOSPI", {})
    if is_bm or oos_bm:
        print(f"  {'─'*80}")
        is_ret = f"{is_bm['total_return']:+.1%}" if is_bm else "N/A"
        oos_ret = f"{oos_bm['total_return']:+.1%}" if oos_bm else "N/A"
        print(f"  {'BM: KOSPI 200':<34} | {is_ret:>10} {'─':>10} | {oos_ret:>10} {'─':>10}")

    # ── 통계 유의성 ──
    print("\n" + "=" * 80)
    print("테스트 2: 통계적 유의성 vs KOSPI 200 (paired t-test + bootstrap)")
    print("=" * 80)

    for key, _, desc in STRATEGIES:
        sig = stat_data["bm_significance"].get(key)
        if not sig:
            continue
        verdict = "유의" if sig["significant"] else "유의하지 않음"
        print(f"\n  {desc}:")
        print(f"    월평균 초과수익: {sig['mean_diff']*100:+.3f}% (n={sig['n_months']}개월)")
        print(f"    t={sig['t_stat']:.2f}, p={sig['p_value']:.4f}")
        print(f"    95% CI: [{sig['ci_lower']*100:+.3f}%, {sig['ci_upper']*100:+.3f}%]")
        print(f"    Bootstrap 승률: {sig['win_rate']:.1%}")
        print(f"    -> {verdict}")

    # ── 롤링 윈도우 ──
    print("\n" + "=" * 80)
    print(f"테스트 3: 롤링 {ROLLING_WINDOW}개월 윈도우 vs KOSPI 200")
    print("=" * 80)

    for key, _, desc in STRATEGIES:
        rd = rolling_all.get(key)
        if not rd:
            continue
        print(f"  {desc}: 양의 알파 {rd['positive_windows']}/{rd['total_windows']} ({rd['win_rate']:.0%})")

    # ── 종합 ──
    print("\n" + "=" * 80)
    print("종합 판단")
    print("=" * 80)

    for key, _, desc in STRATEGIES:
        sig = stat_data["bm_significance"].get(key, {})
        rd = rolling_all.get(key, {})
        p_val = sig.get("p_value", 1)
        win_rate = rd.get("win_rate", 0)
        is_r = is_oos_data["is_results"].get(key, {})
        oos_r = is_oos_data["oos_results"].get(key, {})

        is_sh = is_r.get("sharpe", 0)
        oos_sh = oos_r.get("sharpe", 0)
        consistency = "OOS 유지" if oos_sh > 0.5 else "OOS 약화"

        print(f"  {desc}: p={p_val:.3f}, 롤링승률={win_rate:.0%}, {consistency}")


# ═══════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════

def generate_chart(stat_data, rolling_all):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#F8F9FA")
    plt.subplots_adjust(hspace=0.4, left=0.08, right=0.95, top=0.93, bottom=0.08)

    colors = {"A0": "#2196F3", "A": "#4CAF50", "A+M": "#FF9800", "ATT2": "#E91E63"}

    # ── 패널 1: Bootstrap 분포 ──
    ax1 = axes[0]
    for key, _, desc in STRATEGIES:
        sig = stat_data["bm_significance"].get(key)
        if not sig:
            continue
        boot_pct = sig["boot_means"] * 100
        ax1.hist(boot_pct, bins=60, alpha=0.5, label=f"{key} (p={sig['p_value']:.3f})",
                 color=colors.get(key, "gray"), edgecolor="white", linewidth=0.3)
    ax1.axvline(0, color="black", linewidth=1.5, label="차이=0")
    ax1.set_xlabel("월간 초과수익률 vs KOSPI 200 (%)")
    ax1.set_ylabel("빈도")
    ax1.set_title("Bootstrap 분포 (전략 vs KOSPI 200)", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── 패널 2: 롤링 윈도우 초과수익 ──
    ax2 = axes[1]
    for key, _, desc in STRATEGIES:
        rd = rolling_all.get(key)
        if not rd:
            continue
        dates = [datetime.strptime(r["start_date"], "%Y-%m-%d")
                 for r in rd["rolling_data"] if r["start_date"]]
        excess = [r["excess_return"] * 100 for r in rd["rolling_data"] if r["start_date"]]
        ax2.plot(dates, excess, linewidth=1.5, label=f"{key} ({rd['win_rate']:.0%})",
                 color=colors.get(key, "gray"))

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax2.set_ylabel("초과수익률 vs KOSPI 200 (%)")
    ax2.set_title(f"롤링 {ROLLING_WINDOW}개월 초과수익률", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    charts_dir = Path(__file__).parent.parent / "charts"
    charts_dir.mkdir(exist_ok=True)
    out_path = charts_dir / "robustness_result.png"
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  차트 저장: {out_path}")


# ═══════════════════════════════════════════════════════
# 캐시 저장/로드
# ═══════════════════════════════════════════════════════

ROBUSTNESS_CACHE = CACHE_DIR / "robustness_results.json"


def save_robustness_cache(is_oos, stat, rolling):
    """강건성 검증 결과를 JSON 캐시로 저장"""
    CACHE_DIR.mkdir(exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(),
        "config": dict(BACKTEST_CONFIG),
        "is_oos": _numpy_to_python(is_oos),
        "stat": _numpy_to_python(stat),
        "rolling": _numpy_to_python(rolling),
    }
    ROBUSTNESS_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"  캐시 저장: {ROBUSTNESS_CACHE}")


def load_robustness_cache():
    """JSON 캐시에서 강건성 결과 로드 (없으면 None)"""
    if not ROBUSTNESS_CACHE.exists():
        return None
    data = json.loads(ROBUSTNESS_CACHE.read_text())
    # boot_means를 numpy array로 복원 (차트에서 사용)
    for key, sig in data["stat"].get("bm_significance", {}).items():
        if "boot_means" in sig:
            sig["boot_means"] = np.array(sig["boot_means"])
    return data["is_oos"], data["stat"], data["rolling"]


# ═══════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("Step 8: 강건성 검증")
    print(f"   기간: {BACKTEST_CONFIG['start']} ~ {BACKTEST_CONFIG['end']}")
    print("=" * 70)

    print("\n  [1/3] IS/OOS 분할 검증...")
    is_oos_data = test_is_oos_split()

    print("\n  [2/3] Bootstrap 95% CI 검증...")
    stat_data = test_statistical_significance()

    print("\n  [3/3] 롤링 윈도우 검증...")
    rolling_all = test_rolling_window(stat_data["full_results"])

    show_results(is_oos_data, stat_data, rolling_all)
    generate_chart(stat_data, rolling_all)
    save_robustness_cache(is_oos_data, stat_data, rolling_all)

    print(f"\n  Step 8 완료!")


if __name__ == "__main__":
    main()
