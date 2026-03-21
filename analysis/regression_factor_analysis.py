"""
회귀 팩터 역할 분석
- 전략 A: 기존전략 (밸류 35% + 회귀 30% + 성장 20% + 차별화 15%)
- 전략 B: 회귀 제외 (밸류 50% + 성장 29% + 차별화 21%) — 비중 재배분
- 전략 C: 회귀 only (회귀 100%)

비교 항목: 누적수익률, MDD, Sharpe, 월별 수익률, 하락장/상승장 성과
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import json

from config.settings import BACKTEST_CONFIG
from lib.data import run_strategy_backtest
from lib.factor_engine import DEFAULT_STRATEGY_CODE

# ─── 전략 정의 ───

# A: 기존 전략 (그대로)
STRATEGY_A = DEFAULT_STRATEGY_CODE

# B: 회귀 제외 — ATT 4개를 0으로, 나머지를 비례 재배분
# 원래 비회귀 합계 = 0.70, 이를 1.0으로 스케일
STRATEGY_B = '''"""
Strategy: 회귀 제외 전략
Description: 회귀(ATT) 팩터를 제외하고 밸류+성장+차별화만 사용 (비중 재배분)
"""
SCORING_MODE = {"large": "quartile"}

WEIGHTS_LARGE = {
    "T_PER": .071, "F_PER": .071, "T_EVEBITDA": .071, "F_EVEBITDA": .071,
    "T_PBR": .071, "F_PBR": .071, "T_PCF": .071,
    "T_SPSG": .143, "F_SPSG": .143,
    "F_EPS_M": .214,
}
# 합계: 0.071*7 + 0.143*2 + 0.214 = 0.497 + 0.286 + 0.214 ≈ 1.0

WEIGHTS_SMALL = {}

REGRESSION_MODELS = []
OUTLIER_FILTERS = {}

SCORE_MAP = {
    "T_PER": "t_per_score", "F_PER": "f_per_score",
    "T_EVEBITDA": "t_ev_ebitda_score", "F_EVEBITDA": "f_ev_ebitda_score",
    "T_PBR": "pbr_score", "F_PBR": "f_pbr_score", "T_PCF": "t_pcf_score",
    "T_SPSG": "t_spsg_score", "F_SPSG": "f_spsg_score",
    "F_EPS_M": "f_eps_m_score",
}

SCORING_RULES = {
    "t_per": "rule1", "f_per": "rule1",
    "t_ev_ebitda": "rule1", "f_ev_ebitda": "rule1",
    "pbr": "rule1", "f_pbr": "rule1", "t_pcf": "rule1",
    "t_spsg": "rule2", "f_spsg": "rule2",
    "f_eps_m": "rule2",
}

PARAMS = {"top_n": 30, "tx_cost_bp": 30, "weight_cap_pct": 10}

QUALITY_FILTER = {
    "exclude_spac_etf_reit": True,
    "require_positive_oi": True,
    "require_positive_roe": True,
    "min_avg_volume": 500_000_000,
}
'''

# C: 회귀 only
STRATEGY_C = '''"""
Strategy: 회귀 Only 전략
Description: 회귀(ATT) 매력도 4종만 사용
"""
SCORING_MODE = {"large": "quartile"}

WEIGHTS_LARGE = {
    "ATT_PBR": .25, "ATT_EVIC": .25, "ATT_PER": .25, "ATT_EVEBIT": .25,
}

WEIGHTS_SMALL = {}

REGRESSION_MODELS = [
    ("pbr_roe", "roe", "pbr", "ratio"),
    ("evic_roic", "roic", "ev_ic", "ev_equity"),
    ("fper_epsg", "f_epsg", "f_per", "ratio"),
    ("fevebit_ebitg", "f_ebitg", "f_ev_ebit", "ev_equity_ebit"),
]

OUTLIER_FILTERS = {
    "pbr_roe": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 20},
    "evic_roic": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 51},
    "fper_epsg": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60},
    "fevebit_ebitg": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60},
}

SCORE_MAP = {
    "ATT_PBR": "pbr_roe_attractiveness_score",
    "ATT_EVIC": "evic_roic_attractiveness_score",
    "ATT_PER": "fper_epsg_attractiveness_score",
    "ATT_EVEBIT": "fevebit_ebitg_attractiveness_score",
}

SCORING_RULES = {
    "pbr_roe_attractiveness": "rule2",
    "evic_roic_attractiveness": "rule2",
    "fper_epsg_attractiveness": "rule2",
    "fevebit_ebitg_attractiveness": "rule2",
}

PARAMS = {"top_n": 30, "tx_cost_bp": 30, "weight_cap_pct": 10}

QUALITY_FILTER = {
    "exclude_spac_etf_reit": True,
    "require_positive_oi": True,
    "require_positive_roe": True,
    "min_avg_volume": 500_000_000,
}
'''


def calc_stats(result: dict) -> dict:
    """백테스트 결과에서 주요 지표 산출"""
    rets = np.array(result["monthly_returns"])
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak

    total_return = cum[-1] - 1
    n_years = len(rets) / 12
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = (rets.mean() / rets.std() * np.sqrt(12)) if rets.std() > 0 else 0
    mdd = float(dd.min())

    # 승률 (월별 양수 비율)
    win_rate = (rets > 0).sum() / len(rets)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "monthly_returns": rets,
        "dates": result["rebalance_dates"][:-1],
    }


def regime_analysis(stats_a, stats_b, stats_c, kospi_rets):
    """상승장/하락장별 성과 분해"""
    up_mask = kospi_rets >= 0
    down_mask = kospi_rets < 0

    results = {}
    for name, s in [("기존전략(A)", stats_a), ("회귀제외(B)", stats_b), ("회귀Only(C)", stats_c)]:
        rets = s["monthly_returns"]
        results[name] = {
            "상승장_평균": float(rets[up_mask].mean()) * 100,
            "상승장_월수": int(up_mask.sum()),
            "하락장_평균": float(rets[down_mask].mean()) * 100,
            "하락장_월수": int(down_mask.sum()),
            "전체_평균": float(rets.mean()) * 100,
        }
    return results


def monthly_excess_analysis(stats_a, stats_b):
    """회귀 팩터의 월별 기여도 = 기존전략 - 회귀제외"""
    excess = stats_a["monthly_returns"] - stats_b["monthly_returns"]
    return {
        "dates": stats_a["dates"],
        "excess": excess,
        "positive_months": int((excess > 0).sum()),
        "negative_months": int((excess < 0).sum()),
        "avg_positive": float(excess[excess > 0].mean()) * 100 if (excess > 0).any() else 0,
        "avg_negative": float(excess[excess < 0].mean()) * 100 if (excess < 0).any() else 0,
    }


def main():
    print("=" * 60)
    print("회귀 팩터 역할 분석")
    print("=" * 60)

    strategies = {
        "기존전략 (회귀 30% 포함)": STRATEGY_A,
        "회귀 제외 (밸류+성장+차별화)": STRATEGY_B,
        "회귀 Only (100%)": STRATEGY_C,
    }

    all_results = {}
    for name, code in strategies.items():
        print(f"\n▶ {name} 백테스트 실행 중...")
        result = run_strategy_backtest(code)
        if result and "CUSTOM" in result:
            all_results[name] = result
            r = result["CUSTOM"]
            s = calc_stats(r)
            print(f"  누적수익률: {s['total_return']:.1%}")
            print(f"  CAGR:      {s['cagr']:.1%}")
            print(f"  Sharpe:    {s['sharpe']:.2f}")
            print(f"  MDD:       {s['mdd']:.1%}")
            print(f"  월 승률:   {s['win_rate']:.1%}")
        else:
            print(f"  ❌ 백테스트 실패: {result}")

    if len(all_results) < 3:
        print("\n❌ 일부 전략 실패. 분석 중단.")
        return

    # 통계
    stats_a = calc_stats(all_results["기존전략 (회귀 30% 포함)"]["CUSTOM"])
    stats_b = calc_stats(all_results["회귀 제외 (밸류+성장+차별화)"]["CUSTOM"])
    stats_c = calc_stats(all_results["회귀 Only (100%)"]["CUSTOM"])

    # KOSPI
    kospi = all_results["기존전략 (회귀 30% 포함)"].get("KOSPI", {})
    kospi_rets = np.array(kospi.get("monthly_returns", [0] * len(stats_a["monthly_returns"])))

    # 길이 맞추기
    min_len = min(len(stats_a["monthly_returns"]), len(stats_b["monthly_returns"]),
                  len(stats_c["monthly_returns"]), len(kospi_rets))
    stats_a["monthly_returns"] = stats_a["monthly_returns"][:min_len]
    stats_b["monthly_returns"] = stats_b["monthly_returns"][:min_len]
    stats_c["monthly_returns"] = stats_c["monthly_returns"][:min_len]
    stats_a["dates"] = stats_a["dates"][:min_len]
    kospi_rets = kospi_rets[:min_len]

    # ── 레짐 분석 ──
    print("\n" + "=" * 60)
    print("상승장/하락장 성과 비교")
    print("=" * 60)
    regime = regime_analysis(stats_a, stats_b, stats_c, kospi_rets)
    for name, data in regime.items():
        print(f"\n  {name}:")
        print(f"    상승장 ({data['상승장_월수']}개월): 월평균 {data['상승장_평균']:+.2f}%")
        print(f"    하락장 ({data['하락장_월수']}개월): 월평균 {data['하락장_평균']:+.2f}%")
        print(f"    전체: 월평균 {data['전체_평균']:+.2f}%")

    # ── 회귀 기여도 분석 ──
    print("\n" + "=" * 60)
    print("회귀 팩터 기여도 (기존전략 - 회귀제외)")
    print("=" * 60)
    excess = monthly_excess_analysis(stats_a, stats_b)
    print(f"  회귀가 도움된 달: {excess['positive_months']}개월 (평균 +{excess['avg_positive']:.2f}%p)")
    print(f"  회귀가 손해난 달: {excess['negative_months']}개월 (평균 {excess['avg_negative']:.2f}%p)")

    # 연도별 기여도
    print("\n  연도별 회귀 기여도:")
    yearly = {}
    for d, e in zip(excess["dates"], excess["excess"]):
        year = d[:4]
        yearly.setdefault(year, []).append(e)
    for year in sorted(yearly):
        arr = np.array(yearly[year])
        cum_contrib = np.prod(1 + stats_a["monthly_returns"][[i for i, d in enumerate(stats_a["dates"]) if d[:4] == year]]) - \
                      np.prod(1 + stats_b["monthly_returns"][[i for i, d in enumerate(stats_a["dates"]) if d[:4] == year]])
        print(f"    {year}: 누적 기여 {cum_contrib*100:+.2f}%p, 월평균 {arr.mean()*100:+.2f}%p")

    # ── 회귀 팩터 약점 구간 ──
    print("\n" + "=" * 60)
    print("회귀 팩터 약점 분석")
    print("=" * 60)

    # 1) 하락장에서 회귀 Only 성과
    down_c = stats_c["monthly_returns"][kospi_rets < 0]
    down_a = stats_a["monthly_returns"][kospi_rets < 0]
    print(f"\n  하락장 월평균 수익률:")
    print(f"    기존전략: {down_a.mean()*100:+.2f}%")
    print(f"    회귀Only: {down_c.mean()*100:+.2f}%")

    # 2) 연속 손실 구간
    neg_streak = 0
    max_neg_streak = 0
    for r in stats_c["monthly_returns"]:
        if r < 0:
            neg_streak += 1
            max_neg_streak = max(max_neg_streak, neg_streak)
        else:
            neg_streak = 0
    print(f"\n  회귀Only 최대 연속 손실: {max_neg_streak}개월")

    # 3) 회귀가 큰 손실 낸 달 Top 5
    excess_arr = excess["excess"]
    worst_idx = np.argsort(excess_arr)[:5]
    print(f"\n  회귀가 가장 손해 본 달 Top 5:")
    for idx in worst_idx:
        print(f"    {excess['dates'][idx]}: {excess_arr[idx]*100:+.2f}%p (KOSPI {kospi_rets[idx]*100:+.1f}%)")

    # 결과 저장
    output = {
        "summary": {
            "기존전략": {"cagr": f"{stats_a['cagr']:.1%}", "sharpe": f"{stats_a['sharpe']:.2f}", "mdd": f"{stats_a['mdd']:.1%}"},
            "회귀제외": {"cagr": f"{stats_b['cagr']:.1%}", "sharpe": f"{stats_b['sharpe']:.2f}", "mdd": f"{stats_b['mdd']:.1%}"},
            "회귀Only": {"cagr": f"{stats_c['cagr']:.1%}", "sharpe": f"{stats_c['sharpe']:.2f}", "mdd": f"{stats_c['mdd']:.1%}"},
        },
        "regime": regime,
        "excess_monthly": {
            "dates": excess["dates"],
            "values": [float(v) for v in excess["excess"]],
        },
    }
    out_path = ROOT / "analysis" / "regression_factor_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
