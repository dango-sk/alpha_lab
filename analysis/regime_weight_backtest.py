"""
레짐별 회귀 팩터 가중치 조절 백테스트

강세장: 회귀(매력도) 비중 UP (30% → 40~45%)
약세장: 회귀(매력도) 비중 DOWN (30% → 10~15%), 성장/매출 비중 UP

기존 A0 전략 대비 성과 비교.
"""
import sys
import numpy as np
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config.settings import BACKTEST_CONFIG
from lib.db import get_conn
from lib.factor_engine import (
    code_to_module, DEFAULT_STRATEGY_CODE,
    score_stocks_from_strategy, prefetch_all_data, clear_prefetch_cache,
)
from step7_backtest import (
    run_backtest, calc_all_benchmarks,
    get_db, get_rebalance_dates, get_universe_stocks, _numpy_to_python,
)


# ═══════════════════════════════════════════════════════
# A0 기본 가중치 (합계 1.0)
#   밸류 35%: T_PER .05, F_PER .05, T_EVEBITDA .05, F_EVEBITDA .05,
#             T_PBR .05, F_PBR .05, T_PCF .05
#   회귀 30%: ATT_PBR .05, ATT_EVIC .05, ATT_PER .10, ATT_EVEBIT .10
#   성장 20%: T_SPSG .10, F_SPSG .10
#   차별화 15%: F_EPS_M .15
# ═══════════════════════════════════════════════════════

# 공통 부분 (변하지 않는 코드)
_STRATEGY_TEMPLATE = '''"""
Strategy: {name}
Description: {desc}
"""

SCORING_MODE = {{"large": "quartile"}}

WEIGHTS_LARGE = {weights}
WEIGHTS_SMALL = {{}}

REGRESSION_MODELS = [
    ("pbr_roe", "roe", "pbr", "ratio"),
    ("evic_roic", "roic", "ev_ic", "ev_equity"),
    ("fper_epsg", "f_epsg", "f_per", "ratio"),
    ("fevebit_ebitg", "f_ebitg", "f_ev_ebit", "ev_equity_ebit"),
]

OUTLIER_FILTERS = {{
    "pbr_roe": {{"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 20}},
    "evic_roic": {{"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 51}},
    "fper_epsg": {{"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60}},
    "fevebit_ebitg": {{"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60}},
}}

SCORE_MAP = {{
    "T_PER": "t_per_score", "F_PER": "f_per_score",
    "T_EVEBITDA": "t_ev_ebitda_score", "F_EVEBITDA": "f_ev_ebitda_score",
    "T_PBR": "pbr_score", "F_PBR": "f_pbr_score", "T_PCF": "t_pcf_score",
    "ATT_PBR": "pbr_roe_attractiveness_score", "ATT_EVIC": "evic_roic_attractiveness_score",
    "ATT_PER": "fper_epsg_attractiveness_score", "ATT_EVEBIT": "fevebit_ebitg_attractiveness_score",
    "T_SPSG": "t_spsg_score", "F_SPSG": "f_spsg_score",
    "F_EPS_M": "f_eps_m_score",
}}

SCORING_RULES = {{
    "t_per": "rule1", "f_per": "rule1",
    "t_ev_ebitda": "rule1", "f_ev_ebitda": "rule1",
    "pbr": "rule1", "f_pbr": "rule1", "t_pcf": "rule1",
    "pbr_roe_attractiveness": "rule2",
    "evic_roic_attractiveness": "rule2",
    "fper_epsg_attractiveness": "rule2",
    "fevebit_ebitg_attractiveness": "rule2",
    "t_spsg": "rule2", "f_spsg": "rule2",
    "f_eps_m": "rule2",
}}

PARAMS = {{"top_n": 30, "tx_cost_bp": 30, "weight_cap_pct": 10}}

QUALITY_FILTER = {{
    "exclude_spac_etf_reit": True,
    "require_positive_oi": True,
    "require_positive_roe": True,
    "min_avg_volume": 500_000_000,
}}
'''


def make_strategy_code(name, desc, weights_dict):
    return _STRATEGY_TEMPLATE.format(
        name=name,
        desc=desc,
        weights=repr(weights_dict),
    )


# ─── 가중치 시나리오 정의 ───

# 기존 A0 (밸류35 + 회귀30 + 성장20 + 모멘텀15)
W_A0 = {
    "T_PER": .05, "F_PER": .05, "T_EVEBITDA": .05, "F_EVEBITDA": .05,
    "T_PBR": .05, "F_PBR": .05, "T_PCF": .05,
    "ATT_PBR": .05, "ATT_EVIC": .05, "ATT_PER": .10, "ATT_EVEBIT": .10,
    "T_SPSG": .10, "F_SPSG": .10,
    "F_EPS_M": .15,
}

# 강세장용: 회귀 UP (30→45%), 밸류 DOWN (35→20%), 성장 유지, 모멘텀 유지
W_BULL_ATT_UP = {
    "T_PER": .03, "F_PER": .03, "T_EVEBITDA": .03, "F_EVEBITDA": .03,
    "T_PBR": .03, "F_PBR": .03, "T_PCF": .02,
    "ATT_PBR": .10, "ATT_EVIC": .10, "ATT_PER": .12, "ATT_EVEBIT": .13,
    "T_SPSG": .08, "F_SPSG": .08,
    "F_EPS_M": .19,
}

# 약세장용 V1: 회귀 DOWN (30→10%), 성장 UP (20→40%), 밸류 유지, 모멘텀 유지
W_BEAR_ATT_DOWN_V1 = {
    "T_PER": .05, "F_PER": .05, "T_EVEBITDA": .05, "F_EVEBITDA": .05,
    "T_PBR": .05, "F_PBR": .05, "T_PCF": .05,
    "ATT_PBR": .00, "ATT_EVIC": .05, "ATT_PER": .025, "ATT_EVEBIT": .025,
    "T_SPSG": .20, "F_SPSG": .20,
    "F_EPS_M": .15,
}

# 약세장용 V2: 회귀 DOWN (30→5%), ATT_EVIC만 유지, 성장 UP (20→50%)
W_BEAR_ATT_DOWN_V2 = {
    "T_PER": .05, "F_PER": .05, "T_EVEBITDA": .05, "F_EVEBITDA": .05,
    "T_PBR": .04, "F_PBR": .04, "T_PCF": .02,
    "ATT_PBR": .00, "ATT_EVIC": .05, "ATT_PER": .00, "ATT_EVEBIT": .00,
    "T_SPSG": .25, "F_SPSG": .25,
    "F_EPS_M": .15,
}

# 약세장용 V3: 회귀 전부 제거, 성장+PCF 올림 (PCF가 약세장에서 유일한 밸류 팩터로 작동)
W_BEAR_NO_ATT = {
    "T_PER": .05, "F_PER": .05, "T_EVEBITDA": .05, "F_EVEBITDA": .05,
    "T_PBR": .03, "F_PBR": .03, "T_PCF": .09,
    "ATT_PBR": .00, "ATT_EVIC": .00, "ATT_PER": .00, "ATT_EVEBIT": .00,
    "T_SPSG": .25, "F_SPSG": .25,
    "F_EPS_M": .15,
}

SCENARIOS = [
    # (이름, 설명, 강세장 가중치, 약세장 가중치)
    ("A0 기준 (레짐 무관)",
     "기존 A0 그대로", W_A0, W_A0),

    ("시나리오1: 강세↑회귀 / 약세↓회귀(10%)",
     "강세=회귀45%, 약세=회귀10%+성장35%",
     W_BULL_ATT_UP, W_BEAR_ATT_DOWN_V1),

    ("시나리오2: 강세↑회귀 / 약세↓회귀(5%,EVIC만)",
     "강세=회귀45%, 약세=EVIC 5%만+성장40%",
     W_BULL_ATT_UP, W_BEAR_ATT_DOWN_V2),

    ("시나리오3: 강세↑회귀 / 약세 회귀제거",
     "강세=회귀45%, 약세=회귀0%+성장40%+PCF9%",
     W_BULL_ATT_UP, W_BEAR_NO_ATT),

    ("시나리오4: 강세=A0 / 약세↓회귀(10%)",
     "강세=A0 그대로, 약세=회귀10%+성장35%",
     W_A0, W_BEAR_ATT_DOWN_V1),

    ("시나리오5: 강세=A0 / 약세 회귀제거",
     "강세=A0 그대로, 약세=회귀0%+성장40%",
     W_A0, W_BEAR_NO_ATT),
]


def calc_stats(rets):
    rets = np.array(rets)
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
        "total": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "avg_monthly": rets.mean(),
    }


def main():
    ma_window = 50

    print("=" * 90)
    print("레짐별 회귀 팩터 가중치 조절 백테스트")
    print(f"  레짐 기준: KOSPI 200 {ma_window}일 MA")
    print(f"  Weight cap: 10%, Top-N: 30, TX cost: 30bp")
    print("=" * 90)

    # ── 레짐 신호 사전 계산 ──
    _conn_regime = get_conn()
    regime_cache = {}

    def _get_regime(calc_date):
        if calc_date in regime_cache:
            return regime_cache[calc_date]
        rows = _conn_regime.execute(
            "SELECT close FROM daily_price WHERE stock_code = '069500' "
            f"AND trade_date <= ? ORDER BY trade_date DESC LIMIT {ma_window + 1}",
            (calc_date,)
        ).fetchall()
        prices = [r[0] for r in rows if r[0]]
        if len(prices) < ma_window + 1:
            result = "Bull"
        else:
            current = prices[0]
            ma = float(np.mean(prices[1:ma_window + 1]))
            result = "Bull" if current >= ma else "Bear"
        regime_cache[calc_date] = result
        return result

    # ── 각 시나리오 백테스트 ──
    all_stats = {}

    for scenario_name, desc, w_bull, w_bear in SCENARIOS:
        print(f"\n{'─'*70}")
        print(f"▶ {scenario_name}")
        print(f"  {desc}")

        # 가중치 합계 검증
        for label, w in [("강세", w_bull), ("약세", w_bear)]:
            total_w = sum(w.values())
            att_w = sum(v for k, v in w.items() if k.startswith("ATT_"))
            growth_w = w.get("T_SPSG", 0) + w.get("F_SPSG", 0)
            print(f"  {label}: 회귀={att_w:.0%}, 성장={growth_w:.0%}, 합계={total_w:.2f}")

        # 전략 모듈 생성
        bull_code = make_strategy_code(f"{scenario_name}_bull", desc, w_bull)
        bear_code = make_strategy_code(f"{scenario_name}_bear", desc, w_bear)
        bull_module = code_to_module(bull_code)
        bear_module = code_to_module(bear_code)

        # 레짐 기반 stock_selector
        def make_selector(b_mod, e_mod):
            def selector(conn, calc_date, _top_n):
                regime = _get_regime(calc_date)
                module = b_mod if regime == "Bull" else e_mod
                universe_set = get_universe_stocks(conn, calc_date, rebal_type="monthly", min_market_cap=BACKTEST_CONFIG.get("min_market_cap", 0))
                candidates = score_stocks_from_strategy(conn, calc_date, module)
                return [(c, s) for c, s in candidates if c in universe_set][:_top_n]
            return selector

        # 백테스트 설정
        orig = {
            "top_n_stocks": BACKTEST_CONFIG["top_n_stocks"],
            "transaction_cost_bp": BACKTEST_CONFIG["transaction_cost_bp"],
            "weight_cap_pct": BACKTEST_CONFIG.get("weight_cap_pct", 15),
        }
        try:
            BACKTEST_CONFIG["top_n_stocks"] = 30
            BACKTEST_CONFIG["transaction_cost_bp"] = 30
            BACKTEST_CONFIG["weight_cap_pct"] = 10

            pf_conn = get_db()
            prefetch_all_data(pf_conn)
            pf_conn.close()

            result = run_backtest(
                scenario_name,
                stock_selector=make_selector(bull_module, bear_module),
                rebal_type="monthly",
            )

            if result:
                s = calc_stats(result["monthly_returns"])
                all_stats[scenario_name] = s
                print(f"  → 수익률: {s['total']:+.1%}, CAGR: {s['cagr']:+.1%}, "
                      f"Sharpe: {s['sharpe']:.2f}, MDD: {s['mdd']:.1%}, 승률: {s['win_rate']:.0%}")

                # 레짐별 성과도 표시
                rb_dates = result.get("rebalance_dates", [])
                m_rets = result["monthly_returns"]
                bull_rets = []
                bear_rets = []
                for i, r in enumerate(m_rets):
                    if i < len(rb_dates) - 1:
                        regime = _get_regime(rb_dates[i])
                        if regime == "Bull":
                            bull_rets.append(r)
                        else:
                            bear_rets.append(r)
                if bull_rets:
                    print(f"    강세장 ({len(bull_rets)}개월): 월평균 {np.mean(bull_rets)*100:+.2f}%")
                if bear_rets:
                    print(f"    약세장 ({len(bear_rets)}개월): 월평균 {np.mean(bear_rets)*100:+.2f}%")
            else:
                print("  → 백테스트 실패")

        finally:
            BACKTEST_CONFIG["top_n_stocks"] = orig["top_n_stocks"]
            BACKTEST_CONFIG["transaction_cost_bp"] = orig["transaction_cost_bp"]
            BACKTEST_CONFIG["weight_cap_pct"] = orig["weight_cap_pct"]
            clear_prefetch_cache()

    _conn_regime.close()

    # ── 비교 요약 ──
    print("\n" + "=" * 90)
    print("▶ 전체 비교 요약")
    print("=" * 90)
    print(f"{'시나리오':<45} {'수익률':>8} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'승률':>6}")
    print("-" * 87)

    base = all_stats.get("A0 기준 (레짐 무관)", {})
    for name, s in all_stats.items():
        marker = " ★" if s.get("sharpe", 0) > base.get("sharpe", 0) else ""
        print(f"{name:<45} {s['total']:>+7.1%} {s['cagr']:>+7.1%} "
              f"{s['sharpe']:>7.2f} {s['mdd']:>7.1%} {s['win_rate']:>5.0%}{marker}")

    if base:
        print(f"\n기준 A0 Sharpe: {base.get('sharpe', 0):.2f}")
        best_name = max(all_stats, key=lambda k: all_stats[k].get("sharpe", 0))
        best = all_stats[best_name]
        if best_name != "A0 기준 (레짐 무관)":
            print(f"Best: {best_name} (Sharpe {best['sharpe']:.2f}, "
                  f"CAGR {best['cagr']:+.1%})")
        else:
            print("→ 레짐 전환 시나리오 중 A0를 이기는 조합이 없습니다.")

    print("\n완료!")


if __name__ == "__main__":
    main()
