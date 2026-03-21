"""
회귀 팩터가 손해 본 달에 어떤 종목을 잘못 골랐는지 분석

방법:
- 기존전략(회귀 포함) vs 회귀제외 전략의 포트폴리오를 월별로 비교
- 회귀 기여도가 마이너스인 달에:
  - 회귀 때문에 들어온 종목 (기존에만 있고, 회귀제외에는 없는 종목)
  - 회귀 때문에 빠진 종목 (회귀제외에만 있고, 기존에는 없는 종목)
  - 각 종목의 해당 월 수익률
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import json
import numpy as np
import pandas as pd

from config.settings import BACKTEST_CONFIG
from lib.db import get_conn
from lib.factor_engine import (
    code_to_module, score_stocks_from_strategy,
    DEFAULT_STRATEGY_CODE,
)

# 회귀제외 전략 코드
NO_REG_CODE = '''"""
Strategy: 회귀 제외
"""
SCORING_MODE = {"large": "quartile"}
WEIGHTS_LARGE = {
    "T_PER": .071, "F_PER": .071, "T_EVEBITDA": .071, "F_EVEBITDA": .071,
    "T_PBR": .071, "F_PBR": .071, "T_PCF": .071,
    "T_SPSG": .143, "F_SPSG": .143,
    "F_EPS_M": .214,
}
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

# 회귀 기여도 마이너스 달 (이전 분석 결과)
BAD_MONTHS = [
    ("2021-04-01", "2021-05-03"),
    ("2021-05-03", "2021-06-01"),
    ("2021-08-02", "2021-09-01"),
    ("2022-02-03", "2022-03-02"),
    ("2022-03-02", "2022-04-01"),
    ("2022-04-01", "2022-05-02"),
    ("2022-06-02", "2022-07-01"),
    ("2022-08-01", "2022-09-01"),
    ("2022-11-01", "2022-12-01"),
    ("2023-06-01", "2023-07-03"),
    ("2023-07-03", "2023-08-01"),
    ("2024-04-01", "2024-05-02"),
    ("2024-05-02", "2024-06-03"),
    ("2025-02-03", "2025-03-04"),
    ("2025-04-01", "2025-05-02"),
]

# 기여도 값 (이전 분석 결과에서)
EXCESS_DATA = json.load(open(ROOT / "analysis" / "regression_factor_results.json"))
excess_dates = EXCESS_DATA["excess_monthly"]["dates"]
excess_values = EXCESS_DATA["excess_monthly"]["values"]
excess_map = dict(zip(excess_dates, excess_values))


def get_stock_return(conn, code, start_date, end_date):
    """종목 월 수익률"""
    sp = conn.execute(
        "SELECT close FROM daily_price WHERE stock_code=? AND trade_date>=? ORDER BY trade_date ASC LIMIT 1",
        (code, start_date),
    ).fetchone()
    ep = conn.execute(
        "SELECT close FROM daily_price WHERE stock_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
        (code, end_date),
    ).fetchone()
    if sp and ep and sp[0] > 0:
        return (ep[0] - sp[0]) / sp[0]
    return None


def get_stock_name(conn, code):
    row = conn.execute(
        "SELECT stock_name FROM stock_master WHERE stock_code=?", (code,)
    ).fetchone()
    return row[0] if row else code


def get_top_stocks(conn, strategy_module, calc_date, top_n=30):
    """전략 모듈로 상위 종목 코드 리스트 반환"""
    candidates = score_stocks_from_strategy(conn, calc_date, strategy_module)
    filtered = []
    for code, score in candidates:
        if len(filtered) >= top_n:
            break
        price_exists = conn.execute(
            "SELECT COUNT(*) FROM daily_price "
            "WHERE stock_code=? AND trade_date >= date(?, '-5 days') AND trade_date <= ?",
            (code, calc_date, calc_date),
        ).fetchone()[0]
        if price_exists == 0:
            continue
        vol_data = conn.execute(
            "SELECT AVG(close * volume) FROM daily_price "
            "WHERE stock_code=? AND trade_date<=? AND trade_date>=date(?, '-30 days')",
            (code, calc_date, calc_date),
        ).fetchone()
        if vol_data and vol_data[0] and vol_data[0] >= 100_000_000:
            filtered.append((code, score))
    return filtered


def main():
    conn = get_conn()
    mod_with = code_to_module(DEFAULT_STRATEGY_CODE)
    mod_without = code_to_module(NO_REG_CODE)

    # 기여도 큰 순서로 정렬 (가장 큰 손해 순)
    bad_months_sorted = []
    for start, end in BAD_MONTHS:
        contrib = excess_map.get(start, 0)
        bad_months_sorted.append((start, end, contrib))
    bad_months_sorted.sort(key=lambda x: x[2])

    print("=" * 70)
    print("회귀 팩터가 손해 본 달 — 종목 수준 분석")
    print("=" * 70)

    all_analysis = []
    # 상위 8개만 상세 분석
    for start, end, contrib in bad_months_sorted[:8]:
        print(f"\n{'─'*70}")
        print(f"📅 {start[:7]} | 회귀 기여도: {contrib*100:+.2f}%p")
        print(f"{'─'*70}")

        stocks_with = get_top_stocks(conn, mod_with, start)
        stocks_without = get_top_stocks(conn, mod_without, start)

        codes_with = set(c for c, _ in stocks_with)
        codes_without = set(c for c, _ in stocks_without)

        # 회귀 때문에 들어온 종목
        only_with_reg = codes_with - codes_without
        # 회귀 때문에 빠진 종목
        only_without_reg = codes_without - codes_with
        # 공통 종목
        common = codes_with & codes_without

        month_data = {
            "month": start[:7],
            "contrib": contrib,
            "added_by_reg": [],
            "removed_by_reg": [],
        }

        if only_with_reg:
            print(f"\n  🔵 회귀 때문에 편입된 종목 ({len(only_with_reg)}개):")
            for code in only_with_reg:
                name = get_stock_name(conn, code)
                ret = get_stock_return(conn, code, start, end)
                ret_str = f"{ret*100:+.1f}%" if ret is not None else "N/A"
                score = next((s for c, s in stocks_with if c == code), 0)
                print(f"    {name:12s} | 수익률: {ret_str:>7s} | 점수: {score:.1f}")
                month_data["added_by_reg"].append({
                    "name": name, "return": ret, "score": score
                })

        if only_without_reg:
            print(f"\n  🔴 회귀 때문에 빠진 종목 ({len(only_without_reg)}개):")
            for code in only_without_reg:
                name = get_stock_name(conn, code)
                ret = get_stock_return(conn, code, start, end)
                ret_str = f"{ret*100:+.1f}%" if ret is not None else "N/A"
                score = next((s for c, s in stocks_without if c == code), 0)
                print(f"    {name:12s} | 수익률: {ret_str:>7s} | 점수: {score:.1f}")
                month_data["removed_by_reg"].append({
                    "name": name, "return": ret, "score": score
                })

        # 요약
        added_rets = [d["return"] for d in month_data["added_by_reg"] if d["return"] is not None]
        removed_rets = [d["return"] for d in month_data["removed_by_reg"] if d["return"] is not None]
        if added_rets and removed_rets:
            avg_added = np.mean(added_rets) * 100
            avg_removed = np.mean(removed_rets) * 100
            print(f"\n  📊 요약: 회귀로 넣은 종목 평균 {avg_added:+.1f}% vs 뺀 종목 평균 {avg_removed:+.1f}%")
            print(f"         → 차이: {avg_added - avg_removed:+.1f}%p")

        all_analysis.append(month_data)

    # ── 패턴 분석 ──
    print(f"\n{'='*70}")
    print("종합 패턴 분석")
    print(f"{'='*70}")

    total_added_bad = 0
    total_added_good = 0
    total_removed_good = 0
    total_removed_bad = 0

    for m in all_analysis:
        for s in m["added_by_reg"]:
            if s["return"] is not None:
                if s["return"] < 0:
                    total_added_bad += 1
                else:
                    total_added_good += 1
        for s in m["removed_by_reg"]:
            if s["return"] is not None:
                if s["return"] > 0:
                    total_removed_good += 1
                else:
                    total_removed_bad += 1

    print(f"\n  회귀가 손해 본 달에서:")
    print(f"    회귀로 넣은 종목 중 하락한 종목: {total_added_bad}개")
    print(f"    회귀로 넣은 종목 중 상승한 종목: {total_added_good}개")
    print(f"    회귀로 뺀 종목 중 상승한 종목 (놓친 기회): {total_removed_good}개")
    print(f"    회귀로 뺀 종목 중 하락한 종목 (잘 뺀 것): {total_removed_bad}개")

    # 자주 잘못 편입된 섹터
    sector_count = {}
    for m in all_analysis:
        for s in m["added_by_reg"]:
            if s["return"] is not None and s["return"] < 0:
                # 종목명으로는 섹터 파악 어려움 → 이름만 수집
                sector_count[s["name"]] = sector_count.get(s["name"], 0) + 1

    if sector_count:
        print(f"\n  회귀가 반복적으로 잘못 편입한 종목:")
        for name, cnt in sorted(sector_count.items(), key=lambda x: -x[1]):
            if cnt >= 2:
                print(f"    {name}: {cnt}회")

    conn.close()


if __name__ == "__main__":
    main()
