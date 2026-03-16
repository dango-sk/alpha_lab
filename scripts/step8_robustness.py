"""
Step 8: 강건성 검증

step7 백테스트 캐시 결과를 슬라이싱하여 검증 (재계산 없음, 수 초 완료).

검증:
  1. In-Sample vs Out-of-Sample 분할 비교
  2. 통계적 유의성 (paired t-test + bootstrap 95% CI)
  3. 롤링 윈도우 (24개월) 초과수익 일관성
"""
import json
import sys
import numpy as np
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

from config.settings import BACKTEST_CONFIG, CACHE_DIR
from step7_backtest import STRATEGIES, _numpy_to_python

# ─── 한글 폰트 ───
for fname in fm.findSystemFonts():
    if any(k in fname for k in ["AppleGothic", "NanumGothic", "Malgun", "NotoSansCJK"]):
        matplotlib.rc("font", family=fm.FontProperties(fname=fname).get_name())
        break
matplotlib.rcParams["axes.unicode_minus"] = False

BASELINE_KEY = "A0"
N_BOOTSTRAP = 10_000
ROLLING_WINDOW = 24


# ═══════════════════════════════════════════════════════
# 캐시에서 결과 로드
# ═══════════════════════════════════════════════════════

def _load_cached_results():
    """step7 백테스트 캐시 로드 (PG → JSON fallback). A0 + KOSPI(BM) 반환."""
    # 1) PG
    try:
        from lib.db import get_conn
        conn = get_conn()
        results = {}
        for name in ["A0", "KOSPI"]:
            row = conn.execute("""
                SELECT results_json FROM backtest_cache
                WHERE name = ? AND universe = ? AND rebal_type = ?
            """, (name, "KOSPI", "monthly")).fetchone()
            if row and row[0]:
                data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                results[name] = data
        conn.close()
        if "A0" in results:
            return results
    except Exception as e:
        print(f"  PG 로드 실패 (JSON fallback): {e}")

    # 2) JSON fallback
    cache_path = CACHE_DIR / "backtest_KOSPI_monthly.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        return raw.get("results", raw)

    # 3) legacy
    cache_path = CACHE_DIR / "backtest_results.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        return raw.get("results", raw)

    return {}


def _slice_results(results, start, end):
    """결과를 기간으로 슬라이싱. 통계 재계산."""
    sliced = {}
    for key, val in results.items():
        rb = val.get("rebalance_dates", [])
        mr = val.get("monthly_returns", [])
        pv = val.get("portfolio_values", [])
        if not rb or not mr:
            sliced[key] = dict(val)
            continue

        indices = [i for i, d in enumerate(rb) if start <= d <= end]
        if not indices:
            continue

        v = dict(val)
        v["rebalance_dates"] = [rb[i] for i in indices]

        if pv and len(pv) == len(rb):
            sliced_pv = [pv[i] for i in indices]
            base = sliced_pv[0] if sliced_pv[0] != 0 else 1.0
            v["portfolio_values"] = [p / base for p in sliced_pv]

        if mr and len(mr) == len(rb) - 1:
            mr_indices = [i for i in indices if i < len(mr)]
            v["monthly_returns"] = [mr[i] for i in mr_indices]

        # 통계 재계산
        new_mr = v.get("monthly_returns", [])
        new_pv = v.get("portfolio_values", [])
        if new_mr:
            arr = np.array(new_mr)
            v["total_return"] = np.prod(1 + arr) - 1
            n_years = max(len(new_mr) / 12, 0.5)
            tr = v["total_return"]
            v["cagr"] = (1 + tr) ** (1 / n_years) - 1 if tr > -1 else 0
            v["sharpe"] = float(arr.mean() / arr.std() * np.sqrt(12)) if arr.std() > 1e-8 else 0
            v["months"] = len(new_mr)
            v["monthly_std"] = float(arr.std())
            cum = np.cumprod(1 + arr)
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            v["mdd"] = float(np.min(dd))

        sliced[key] = v
    return sliced


# ═══════════════════════════════════════════════════════
# 테스트 1: IS/OOS 분할 (슬라이싱)
# ═══════════════════════════════════════════════════════

def test_is_oos_split(full_results):
    is_start = BACKTEST_CONFIG["start"]
    is_end = BACKTEST_CONFIG["insample_end"]
    oos_start = BACKTEST_CONFIG["oos_start"]
    oos_end = BACKTEST_CONFIG["end"]

    is_results = _slice_results(full_results, is_start, is_end)
    oos_results = _slice_results(full_results, oos_start, oos_end)

    # BM은 KOSPI 키에서 가져옴
    bm_results = {"is": {}, "oos": {}}
    is_bm = is_results.get("KOSPI")
    oos_bm = oos_results.get("KOSPI")
    if is_bm:
        bm_results["is"]["KOSPI"] = {
            "total_return": is_bm.get("total_return", 0),
            "cagr": is_bm.get("cagr", 0),
            "name": "KOSPI 200",
        }
    if oos_bm:
        bm_results["oos"]["KOSPI"] = {
            "total_return": oos_bm.get("total_return", 0),
            "cagr": oos_bm.get("cagr", 0),
            "name": "KOSPI 200",
        }

    # 전략 결과에서 BM 제외
    is_strat = {k: v for k, v in is_results.items() if k != "KOSPI"}
    oos_strat = {k: v for k, v in oos_results.items() if k != "KOSPI"}

    return {"is_results": is_strat, "oos_results": oos_strat, "benchmarks": bm_results}


# ═══════════════════════════════════════════════════════
# 테스트 2: 통계적 유의성 (슬라이싱)
# ═══════════════════════════════════════════════════════

def test_statistical_significance(full_results):
    a0 = full_results.get(BASELINE_KEY)
    bm = full_results.get("KOSPI")
    if not a0 or not bm:
        print("  A0 또는 KOSPI 결과 없음")
        return {"full_results": full_results, "bm_significance": {}}

    strat_rets = np.array(a0["monthly_returns"])
    bm_rets = np.array(bm.get("monthly_returns", []))
    n = min(len(strat_rets), len(bm_rets))
    if n < 3:
        print("  데이터 부족")
        return {"full_results": full_results, "bm_significance": {}}

    diff = strat_rets[:n] - bm_rets[:n]
    t_stat, p_value = stats.ttest_rel(strat_rets[:n], bm_rets[:n])

    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(diff, size=len(diff), replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)

    bm_significance = {}
    for key, _, desc in STRATEGIES:
        if key not in full_results or key == "KOSPI":
            continue
        s_rets = np.array(full_results[key]["monthly_returns"])
        nn = min(len(s_rets), len(bm_rets))
        d = s_rets[:nn] - bm_rets[:nn]
        t, p = stats.ttest_rel(s_rets[:nn], bm_rets[:nn])
        bm = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(N_BOOTSTRAP)])
        bm_significance[key] = {
            "n_months": nn,
            "mean_diff": d.mean(),
            "t_stat": t,
            "p_value": p,
            "ci_lower": np.percentile(bm, 2.5),
            "ci_upper": np.percentile(bm, 97.5),
            "significant": np.percentile(bm, 2.5) > 0,
            "boot_means": bm,
            "win_rate": (bm > 0).mean(),
        }

    return {"full_results": full_results, "bm_significance": bm_significance}


# ═══════════════════════════════════════════════════════
# 테스트 3: 롤링 윈도우 (슬라이싱)
# ═══════════════════════════════════════════════════════

def test_rolling_window(full_results):
    """24개월 슬라이딩 윈도우: 각 전략 vs KOSPI 200"""
    bm_data = full_results.get("KOSPI")
    if not bm_data:
        return {}

    bm_monthly = np.array(bm_data.get("monthly_returns", []))
    rebalance_dates = bm_data.get("rebalance_dates", [])

    rolling_all = {}
    for key, _, desc in STRATEGIES:
        if key not in full_results or key == "KOSPI":
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
    """강건성 검증 결과를 JSON + PG에 저장"""
    clean_is_oos = _numpy_to_python(is_oos)
    clean_stat = _numpy_to_python(stat)
    clean_rolling = _numpy_to_python(rolling)

    # 1) JSON 캐시
    CACHE_DIR.mkdir(exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(),
        "config": dict(BACKTEST_CONFIG),
        "is_oos": clean_is_oos,
        "stat": clean_stat,
        "rolling": clean_rolling,
    }
    ROBUSTNESS_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"  JSON 캐시 저장: {ROBUSTNESS_CACHE}")

    # 2) PG backtest_cache
    try:
        from lib.db import get_conn
        from psycopg2.extras import Json
        conn = get_conn()
        data = {"is_oos": clean_is_oos, "stat": clean_stat, "rolling": clean_rolling}
        conn.execute("""
            INSERT INTO backtest_cache (name, universe, rebal_type, results_json, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (name, universe, rebal_type)
            DO UPDATE SET results_json = EXCLUDED.results_json, updated_at = NOW()
        """, ("__ROBUSTNESS__", "ALL", "ALL", Json(data)))
        conn.commit()
        conn.close()
        print(f"  PG 강건성 캐시 저장 완료")
    except Exception as e:
        print(f"  PG 저장 실패: {e}")


def load_robustness_cache():
    """JSON 캐시에서 강건성 결과 로드"""
    if not ROBUSTNESS_CACHE.exists():
        return None
    data = json.loads(ROBUSTNESS_CACHE.read_text())
    for key, sig in data["stat"].get("bm_significance", {}).items():
        if "boot_means" in sig:
            sig["boot_means"] = np.array(sig["boot_means"])
    return data["is_oos"], data["stat"], data["rolling"]


# ═══════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("Step 8: 강건성 검증 (캐시 슬라이싱)")
    print(f"   기간: {BACKTEST_CONFIG['start']} ~ {BACKTEST_CONFIG['end']}")
    print("=" * 70)

    print("\n  캐시 로드...")
    full_results = _load_cached_results()
    if not full_results or "A0" not in full_results:
        print("  ✗ 백테스트 캐시가 없습니다. step7을 먼저 실행하세요.")
        return

    a0 = full_results["A0"]
    print(f"  A0: {len(a0.get('monthly_returns',[]))}개월, "
          f"총수익률 {a0.get('total_return',0):+.1%}, "
          f"CAGR {a0.get('cagr',0):+.1%}")
    if "KOSPI" in full_results:
        bm = full_results["KOSPI"]
        print(f"  KOSPI: {len(bm.get('monthly_returns',[]))}개월, "
              f"총수익률 {bm.get('total_return',0):+.1%}")

    print("\n  [1/3] IS/OOS 분할 검증...")
    is_oos_data = test_is_oos_split(full_results)

    print("  [2/3] Bootstrap 95% CI 검증...")
    stat_data = test_statistical_significance(full_results)

    print("  [3/3] 롤링 윈도우 검증...")
    rolling_all = test_rolling_window(full_results)

    show_results(is_oos_data, stat_data, rolling_all)
    generate_chart(stat_data, rolling_all)
    save_robustness_cache(is_oos_data, stat_data, rolling_all)

    print(f"\n  Step 8 완료!")


if __name__ == "__main__":
    main()
