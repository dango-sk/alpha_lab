"""
팩터 비중 최적화 분석
- 팩터 롱숏 결과를 근거로 비중 조정 시뮬레이션
- 비중 vs 기여도 갭 분석
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import json
from lib.data import run_strategy_backtest
from lib.factor_engine import DEFAULT_STRATEGY_CODE


# ─── 전략 정의 ───

# A0: 현재 전략 (그대로)
STRATEGY_A0 = DEFAULT_STRATEGY_CODE

# B: 역효과 팩터 3개 제거 (PBR 2개 + EV/EBITDA 2개 + ATT_PER)
# 제거: T_PBR(.05), F_PBR(.05), T_EVEBITDA(.05), F_EVEBITDA(.05), ATT_PER(.10) = 0.30
# 남은 0.70 → 1.0으로 스케일 (각 × 10/7)
STRATEGY_NO_WEAK = '''"""
Strategy: 역효과 팩터 제거
Description: 단독 역효과인 PBR, EV/EBITDA, ATT:F.PER~EPSG 제거, 나머지 비례 재배분
"""
SCORING_MODE = {"large": "quartile"}

WEIGHTS_LARGE = {
    "T_PER": .071, "F_PER": .071, "T_PCF": .071,
    "ATT_PBR": .071, "ATT_EVIC": .071, "ATT_EVEBIT": .143,
    "T_SPSG": .143, "F_SPSG": .143,
    "F_EPS_M": .214,
}

WEIGHTS_SMALL = {}

REGRESSION_MODELS = [
    ("pbr_roe", "roe", "pbr", "ratio"),
    ("evic_roic", "roic", "ev_ic", "ev_equity"),
    ("fevebit_ebitg", "f_ebitg", "f_ev_ebit", "ev_equity_ebit"),
]

OUTLIER_FILTERS = {
    "pbr_roe": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 20},
    "evic_roic": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 51},
    "fevebit_ebitg": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60},
}

SCORE_MAP = {
    "T_PER": "t_per_score", "F_PER": "f_per_score", "T_PCF": "t_pcf_score",
    "ATT_PBR": "pbr_roe_attractiveness_score", "ATT_EVIC": "evic_roic_attractiveness_score",
    "ATT_EVEBIT": "fevebit_ebitg_attractiveness_score",
    "T_SPSG": "t_spsg_score", "F_SPSG": "f_spsg_score",
    "F_EPS_M": "f_eps_m_score",
}

SCORING_RULES = {
    "t_per": "rule1", "f_per": "rule1", "t_pcf": "rule1",
    "pbr_roe_attractiveness": "rule2",
    "evic_roic_attractiveness": "rule2",
    "fevebit_ebitg_attractiveness": "rule2",
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

# C: 모멘텀/성장 강화 (EPS모멘텀 15→25%, 성장 20→25%, 밸류 35→25%, 회귀 30→25%)
STRATEGY_MOMENTUM_UP = '''"""
Strategy: 모멘텀/성장 강화
Description: 팩터 롱숏에서 강력한 EPS모멘텀과 매출성장 비중 확대
"""
SCORING_MODE = {"large": "quartile"}

WEIGHTS_LARGE = {
    "T_PER": .036, "F_PER": .036, "T_EVEBITDA": .036, "F_EVEBITDA": .036,
    "T_PBR": .036, "F_PBR": .036, "T_PCF": .034,
    "ATT_PBR": .04, "ATT_EVIC": .04, "ATT_PER": .085, "ATT_EVEBIT": .085,
    "T_SPSG": .125, "F_SPSG": .125,
    "F_EPS_M": .25,
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
    "T_PER": "t_per_score", "F_PER": "f_per_score",
    "T_EVEBITDA": "t_ev_ebitda_score", "F_EVEBITDA": "f_ev_ebitda_score",
    "T_PBR": "pbr_score", "F_PBR": "f_pbr_score", "T_PCF": "t_pcf_score",
    "ATT_PBR": "pbr_roe_attractiveness_score", "ATT_EVIC": "evic_roic_attractiveness_score",
    "ATT_PER": "fper_epsg_attractiveness_score", "ATT_EVEBIT": "fevebit_ebitg_attractiveness_score",
    "T_SPSG": "t_spsg_score", "F_SPSG": "f_spsg_score",
    "F_EPS_M": "f_eps_m_score",
}

SCORING_RULES = {
    "t_per": "rule1", "f_per": "rule1",
    "t_ev_ebitda": "rule1", "f_ev_ebitda": "rule1",
    "pbr": "rule1", "f_pbr": "rule1", "t_pcf": "rule1",
    "pbr_roe_attractiveness": "rule2",
    "evic_roic_attractiveness": "rule2",
    "fper_epsg_attractiveness": "rule2",
    "fevebit_ebitg_attractiveness": "rule2",
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

# D: 최적 제안 (역효과+약한 팩터 제거 + 모멘텀 강화)
# 제거: PBR(밸류), EV/EBITDA(밸류), ATT_PBR(약한 회귀)
# 남는 것: PER+PCF(밸류), 강한 회귀3종, 매출성장, EPS모멘텀
STRATEGY_OPTIMAL = '''"""
Strategy: 최적 제안
Description: 약한 팩터 전부 제거 + 강력 팩터(EPS모멘텀, 매출성장) 비중 확대
"""
SCORING_MODE = {"large": "quartile"}

WEIGHTS_LARGE = {
    "T_PER": .05, "F_PER": .05, "T_PCF": .05,
    "ATT_EVIC": .075, "ATT_PER": .10, "ATT_EVEBIT": .075,
    "T_SPSG": .15, "F_SPSG": .15,
    "F_EPS_M": .30,
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
    "T_PER": "t_per_score", "F_PER": "f_per_score", "T_PCF": "t_pcf_score",
    "ATT_EVIC": "evic_roic_attractiveness_score",
    "ATT_PER": "fper_epsg_attractiveness_score", "ATT_EVEBIT": "fevebit_ebitg_attractiveness_score",
    "T_SPSG": "t_spsg_score", "F_SPSG": "f_spsg_score",
    "F_EPS_M": "f_eps_m_score",
}

SCORING_RULES = {
    "t_per": "rule1", "f_per": "rule1", "t_pcf": "rule1",
    "evic_roic_attractiveness": "rule2",
    "fper_epsg_attractiveness": "rule2",
    "fevebit_ebitg_attractiveness": "rule2",
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


def calc_stats(result: dict) -> dict:
    rets = np.array(result["monthly_returns"])
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak

    total_return = cum[-1] - 1
    n_years = len(rets) / 12
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = (rets.mean() / rets.std() * np.sqrt(12)) if rets.std() > 0 else 0
    mdd = float(dd.min())
    win_rate = (rets > 0).sum() / len(rets)
    turnover = float(np.mean(result.get("turnovers", [0]))) if result.get("turnovers") else 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "turnover": turnover,
        "monthly_returns": rets,
    }


def main():
    strategies = {
        "A0 (현재 14팩터)": {
            "code": STRATEGY_A0,
            "desc": "밸류35+회귀30+성장20+차별화15",
        },
        "역효과 제거": {
            "code": STRATEGY_NO_WEAK,
            "desc": "PBR·EV/EBITDA 제거, 나머지 비례 재배분",
        },
        "모멘텀/성장 강화": {
            "code": STRATEGY_MOMENTUM_UP,
            "desc": "EPS모멘텀 25%, 매출성장 25%, 밸류 25%, 회귀 25%",
        },
        "최적 제안": {
            "code": STRATEGY_OPTIMAL,
            "desc": "역효과 제거 + EPS모멘텀 30%, 매출성장 30%",
        },
    }

    print("=" * 80)
    print("  팩터 비중 최적화 시뮬레이션")
    print("=" * 80)

    results = {}
    kospi_stats = None

    for name, info in strategies.items():
        print(f"\n▶ {name} 백테스트 실행 중...")
        result = run_strategy_backtest(info["code"])
        if result and "CUSTOM" in result:
            s = calc_stats(result["CUSTOM"])
            results[name] = {**s, "desc": info["desc"]}
            print(f"  누적: {s['total_return']:.1%} | CAGR: {s['cagr']:.1%} | "
                  f"Sharpe: {s['sharpe']:.2f} | MDD: {s['mdd']:.1%}")

            if kospi_stats is None and "KOSPI" in result:
                kospi_stats = calc_stats(result["KOSPI"])
        else:
            print(f"  ❌ 실패")

    # ── 비중 vs 기여도 갭 분석 ──
    print("\n" + "=" * 80)
    print("  팩터 비중 vs 롱숏 Sharpe 갭")
    print("=" * 80)

    factor_gap = [
        ("EPS 모멘텀 (차별화)", 15, 0.79, "강력"),
        ("매출 성장률 (성장)", 20, 0.78, "강력"),
        ("EBIT 성장 (성장)", 0, 0.60, "양호 (미포함)"),
        ("가격 모멘텀", 0, 0.45, "양호 (미포함)"),
        ("EV/IC~ROIC 회귀", 5, 0.43, "양호"),
        ("F.PER~이익성장 회귀", 10, 0.0, "미측정(복합)"),
        ("F.EV/EBIT~EBIT성장 회귀", 10, 0.0, "미측정(복합)"),
        ("PBR~ROE 회귀", 5, 0.27, "약함"),
        ("Forward PER (밸류)", 5, 0.22, "약함"),
        ("PBR (밸류)", 10, -0.10, "역효과"),
        ("EV/EBITDA (밸류)", 10, -0.09, "역효과"),
    ]

    print(f"  {'팩터':<25} {'현재 비중':>8} {'롱숏 Sharpe':>12} {'갭 판정':>10}")
    print(f"  {'─'*60}")
    for name, weight, sharpe, verdict in factor_gap:
        marker = "⚠️" if "역효과" in verdict else ("↑" if weight < sharpe * 30 and sharpe > 0.5 else "")
        print(f"  {name:<25} {weight:>7}% {sharpe:>+11.2f} {verdict:>10} {marker}")

    # ── 전략 비교 요약 ──
    print("\n" + "=" * 80)
    print("  전략 비교 요약")
    print("=" * 80)
    print(f"  {'전략':<25} {'누적':>8} {'CAGR':>7} {'Sharpe':>7} {'MDD':>7} {'승률':>6}")
    print(f"  {'─'*65}")

    for name, s in results.items():
        print(f"  {name:<25} {s['total_return']:>+7.1%} {s['cagr']:>+6.1%} "
              f"{s['sharpe']:>+6.2f} {s['mdd']:>+6.1%} {s['win_rate']:>5.0%}")

    if kospi_stats:
        print(f"  {'KOSPI 200':<25} {kospi_stats['total_return']:>+7.1%} {kospi_stats['cagr']:>+6.1%} "
              f"{kospi_stats['sharpe']:>+6.2f} {kospi_stats['mdd']:>+6.1%} {kospi_stats['win_rate']:>5.0%}")

    # A0 대비 개선도
    if "A0 (현재 14팩터)" in results:
        a0 = results["A0 (현재 14팩터)"]
        print(f"\n  A0 대비 개선도:")
        for name, s in results.items():
            if name == "A0 (현재 14팩터)":
                continue
            d_ret = (s['total_return'] - a0['total_return']) * 100
            d_sharpe = s['sharpe'] - a0['sharpe']
            d_mdd = (s['mdd'] - a0['mdd']) * 100
            print(f"    {name}: 누적 {d_ret:+.1f}%p, Sharpe {d_sharpe:+.2f}, MDD {d_mdd:+.1f}%p")

    # 저장
    output = {}
    for name, s in results.items():
        output[name] = {
            "desc": s["desc"],
            "total_return": f"{s['total_return']:.1%}",
            "cagr": f"{s['cagr']:.1%}",
            "sharpe": f"{s['sharpe']:.2f}",
            "mdd": f"{s['mdd']:.1%}",
            "win_rate": f"{s['win_rate']:.0%}",
        }
    if kospi_stats:
        output["KOSPI 200"] = {
            "total_return": f"{kospi_stats['total_return']:.1%}",
            "cagr": f"{kospi_stats['cagr']:.1%}",
            "sharpe": f"{kospi_stats['sharpe']:.2f}",
            "mdd": f"{kospi_stats['mdd']:.1%}",
        }

    save_path = ROOT / "analysis" / "factor_optimization_results.json"
    with open(save_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {save_path}")


if __name__ == "__main__":
    main()
