"""
전략 백테스트 비교: A0 원본 vs 간소화 전략들
(대형주, 사분위 0-4 스코어링, Top 30)

step7의 run_backtest()를 활용하되, 커스텀 stock_selector로
A0 파이프라인과 동일한 사분위 채점을 직접 수행한다.

사용법:
  python -m analysis.strategy_backtest
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DB_PATH, BACKTEST_CONFIG, CACHE_DIR
from lib.factor_engine import (
    load_factor_data, clear_factor_cache, run_regressions,
    apply_scoring, calc_weighted_scores, apply_quality_filter,
)
from scripts.step7_backtest import (
    run_backtest, calc_all_benchmarks, get_db, get_monthly_rebalance_dates,
)
from analysis.factor_longshort import REGRESSION_MODELS, OUTLIER_FILTERS

import json


# ─── A0 전략 상수 (factor_engine.py DEFAULT_STRATEGY_CODE와 동일) ───

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

SCORE_MAP = {
    "T_PER": "t_per_score", "F_PER": "f_per_score",
    "T_EVEBITDA": "t_ev_ebitda_score", "F_EVEBITDA": "f_ev_ebitda_score",
    "T_PBR": "pbr_score", "F_PBR": "f_pbr_score", "T_PCF": "t_pcf_score",
    "ATT_PBR": "pbr_roe_attractiveness_score",
    "ATT_EVIC": "evic_roic_attractiveness_score",
    "ATT_PER": "fper_epsg_attractiveness_score",
    "ATT_EVEBIT": "fevebit_ebitg_attractiveness_score",
    "T_SPSG": "t_spsg_score", "F_SPSG": "f_spsg_score",
    "F_EPS_M": "f_eps_m_score",
}

WEIGHTS_A0 = {
    "T_PER": .05, "F_PER": .05, "T_EVEBITDA": .05, "F_EVEBITDA": .05,
    "T_PBR": .05, "F_PBR": .05, "T_PCF": .05,
    "ATT_PBR": .05, "ATT_EVIC": .05, "ATT_PER": .10, "ATT_EVEBIT": .10,
    "T_SPSG": .10, "F_SPSG": .10,
    "F_EPS_M": .15,
}

# ─── 간소화 전략 가중치 ───

# 핵심 5팩터: A0에서 비중 10%+ 팩터만 (회귀2 + 성장2 + 모멘텀1)
WEIGHTS_CORE5 = {
    "ATT_PER": .20, "ATT_EVEBIT": .20,
    "T_SPSG": .15, "F_SPSG": .15,
    "F_EPS_M": .30,
}

# 핵심 7팩터: Core5 + 포워드 밸류 2개
WEIGHTS_CORE7 = {
    "F_PER": .10, "F_PBR": .05,
    "ATT_PER": .15, "ATT_EVEBIT": .15,
    "T_SPSG": .10, "F_SPSG": .10,
    "F_EPS_M": .20,
    "ATT_EVIC": .15,
}

# 밸류+회귀 4팩터: 밸류 1 + 회귀 2 + 모멘텀 1 (성장 제외)
WEIGHTS_VR4 = {
    "F_PER": .20,
    "ATT_PER": .30, "ATT_EVEBIT": .30,
    "F_EPS_M": .20,
}

# 기존 비교용
WEIGHTS_ATT2 = {
    "ATT_PBR": 0.5, "ATT_EVIC": 0.5,
}


def make_quartile_selector(weights_large, flip_factors=None):
    """사분위(0-4) 스코어링 기반 stock_selector 팩토리.

    A0 파이프라인과 동일한 quartile 채점:
      1. load_factor_data → 원시 팩터
      2. run_regressions → 매력도 계산 (필요 시)
      3. apply_scoring → 사분위(0~4) 점수
      4. calc_weighted_scores → 가중합 → value_score (0~100)
      5. value_score 상위 종목 선정 + 유동성 필터

    flip_factors: 값을 부호 반전하여 스코어링 방향을 뒤집을 팩터 컬럼명 리스트.
                  (rule2 팩터: 높을수록 좋음 → 반전 후 낮을수록 좋음)
    """
    if flip_factors is None:
        flip_factors = []

    needs_regression = any("ATT" in wk for wk in weights_large if weights_large[wk] > 0)

    def selector(conn, calc_date, top_n):
        import pandas as pd

        df = load_factor_data(conn, calc_date)
        if df is None:
            return []

        # 대형주만
        df = df[df["size_group"] == "large"].copy()

        if len(df) < 10:
            return []

        # 퀄리티 필터 (대시보드와 동일: OI>0, ROE>0, 거래대금>=5억, SPAC/ETF/REIT 제외)
        df = apply_quality_filter(df, {
            "exclude_spac_etf_reit": True,
            "require_positive_oi": True,
            "require_positive_roe": True,
            "min_avg_volume": 500_000_000,
        })
        universe = df[df["quality_pass"] == 1].copy()

        if len(universe) < 10:
            return []

        # 회귀분석 (매력도 팩터용) — 퀄리티 통과 종목만
        if needs_regression:
            universe, _ = run_regressions(universe, REGRESSION_MODELS, OUTLIER_FILTERS)

        # flip_factors: 값 부호 반전 → quartile_rule2에서 방향이 뒤집힘
        # (높은 성장 → 부호반전 → 낮은 값 → rule2에서 낮은 점수)
        for f in flip_factors:
            if f in universe.columns:
                universe[f] = -universe[f]

        # 사분위 스코어링 (0~4)
        scoring_mode = {"large": "quartile", "small": "quartile"}
        universe = apply_scoring(universe, SCORING_RULES, scoring_mode)

        # 가중합 → value_score (0~100)
        universe = calc_weighted_scores(
            universe, weights_large, {}, SCORE_MAP, scoring_mode
        )

        # 상위 후보 (top_n * 2) → 유동성 필터 → top_n
        candidates = universe.nlargest(top_n * 2, "value_score")

        filtered = []
        for _, row in candidates.iterrows():
            if len(filtered) >= top_n:
                break

            code = row["stock_code"].lstrip("A")

            vol = conn.execute("""
                SELECT AVG(close * volume) FROM daily_price
                WHERE stock_code = ? AND trade_date <= ?
                  AND trade_date >= date(?, '-30 days')
            """, (code, calc_date, calc_date)).fetchone()

            price_exists = conn.execute("""
                SELECT COUNT(*) FROM daily_price
                WHERE stock_code = ? AND trade_date >= date(?, '-5 days')
                  AND trade_date <= ?
            """, (code, calc_date, calc_date)).fetchone()[0]

            if price_exists == 0:
                continue
            if vol and vol[0] and vol[0] >= 100_000_000:
                filtered.append((code, float(row["value_score"])))

        return filtered

    return selector


def run_comparison():
    strategies = [
        ("A0_quartile",
         make_quartile_selector(WEIGHTS_A0),
         "A0 (14팩터)"),
        ("core5",
         make_quartile_selector(WEIGHTS_CORE5),
         "Core5 (회귀+성장+모멘텀)"),
        ("core7",
         make_quartile_selector(WEIGHTS_CORE7),
         "Core7 (밸류+회귀+성장)"),
        ("vr4",
         make_quartile_selector(WEIGHTS_VR4),
         "VR4 (밸류+회귀+모멘텀)"),
        ("att2",
         make_quartile_selector(WEIGHTS_ATT2),
         "회귀2 (PBR+EVIC)"),
    ]

    all_keys = [s[0] for s in strategies]

    results = {}
    for key, selector, desc in strategies:
        print(f"\n{'='*60}")
        print(f"  {desc} 백테스트 중...")
        print(f"{'='*60}")

        result = run_backtest(key, stock_selector=selector)
        clear_factor_cache()

        if result:
            result["strategy"] = desc
            results[key] = result
            print(f"  완료: CAGR={result['cagr']*100:+.1f}%, MDD={result['mdd']*100:.1f}%, Sharpe={result['sharpe']:.3f}")

    # 벤치마크
    conn = get_db()
    rebalance_dates = get_monthly_rebalance_dates(conn)
    bm = calc_all_benchmarks(conn, rebalance_dates)
    results.update(bm)
    conn.close()

    # 비교 테이블
    print(f"\n{'='*70}")
    print(f"  전략 비교 (대형주, 사분위 0-4, Top 30)")
    print(f"  기간: {BACKTEST_CONFIG['start']} ~ {BACKTEST_CONFIG['end']}")
    print(f"  시총비중+15%캡 | 거래비용 {BACKTEST_CONFIG['transaction_cost_bp']}bp")
    print(f"{'='*70}")
    print(f"  {'전략':<28} {'누적':>8} {'CAGR':>7} {'MDD':>7} {'Sharpe':>7} {'월평균':>7} {'턴오버':>7}")
    print(f"  {'─'*70}")

    for key in all_keys + ["KOSPI"]:
        r = results.get(key)
        if not r:
            continue
        total = r.get("total_return", 0) * 100
        cagr = r.get("cagr", 0) * 100
        mdd = r.get("mdd", 0) * 100
        sharpe = r.get("sharpe", 0)
        avg_m = r.get("avg_monthly_return", 0) * 100
        turn = r.get("avg_turnover", 0) * 100
        name = r["strategy"]
        print(f"  {name:<28} {total:>+7.1f}% {cagr:>+6.1f}% {mdd:>6.1f}% {sharpe:>7.3f} {avg_m:>+6.2f}% {turn:>6.1f}%")

    print(f"  {'─'*70}")
    print(f"{'='*70}\n")

    # 캐시 저장
    CACHE_DIR.mkdir(exist_ok=True)
    save_path = CACHE_DIR / "strategy_comparison.json"
    output = {}
    for key, r in results.items():
        output[key] = {k: v for k, v in r.items()
                       if k not in ("monthly_returns", "portfolio_values", "portfolio_sizes", "rebalance_dates")}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"  결과 저장: {save_path}")


if __name__ == "__main__":
    run_comparison()
