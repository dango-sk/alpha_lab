"""
강세장/약세장별 팩터 성과 분석

KOSPI 200 50일 이동평균 기준 레짐 구분 후,
각 팩터의 Long-Short 수익률을 레짐별로 분해하여
어떤 팩터가 어떤 장세에서 효과적인지 분석.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lib.db import get_conn
from analysis.regime_combo_backtest import get_kospi200_50d_signal

CACHE_DIR = ROOT / "cache"

# ── 팩터 카테고리 분류 ──
FACTOR_CATEGORIES = {
    "밸류에이션": [
        "t_per", "pbr", "t_ev_ebitda", "t_pcf",
        "f_per", "f_pbr", "f_ev_ebitda",
    ],
    "매력도(회귀)": [
        "pbr_roe_attractiveness", "evic_roic_attractiveness",
        "fper_epsg_attractiveness", "fevebit_ebitg_attractiveness",
    ],
    "성장": ["t_spsg", "f_spsg", "f_epsg", "f_ebitg"],
    "모멘텀": ["price_m", "f_eps_m"],
    "퀄리티/안정성": ["ndebt_ebitda"],
}


def load_factor_data():
    """캐시된 팩터 리서치 결과를 모두 로드."""
    factors = {}
    for f in CACHE_DIR.glob("factor_research_*.json"):
        key = f.stem.replace("factor_research_", "")
        with open(f) as fp:
            data = json.load(fp)
        factors[key] = data
    return factors


def get_regime_for_dates(conn, dates):
    """팩터 리서치 날짜 목록에 대해 레짐 신호 반환."""
    return get_kospi200_50d_signal(conn, dates)


def analyze():
    conn = get_conn()
    factors = load_factor_data()

    if not factors:
        print("팩터 리서치 캐시가 없습니다. 파이프라인을 먼저 실행하세요.")
        return

    # 모든 팩터에 공통인 날짜만 사용
    date_sets = []
    for data in factors.values():
        date_sets.append(set(m["date"] for m in data["monthly_returns"]))
    common_dates = sorted(set.intersection(*date_sets))
    n = len(common_dates)

    # 레짐 신호
    print("KOSPI 200 50일 MA 레짐 신호 계산 중...")
    signals = get_regime_for_dates(conn, common_dates)
    conn.close()

    bull_idx = [i for i, s in enumerate(signals) if s == "bull"]
    bear_idx = [i for i, s in enumerate(signals) if s == "bear"]
    print(f"기간: {common_dates[0]} ~ {common_dates[-1]} ({n}개월)")
    print(f"강세장: {len(bull_idx)}개월, 약세장: {len(bear_idx)}개월\n")

    # ── 팩터별 레짐 성과 계산 ──
    rows = []
    for key, data in factors.items():
        rets = data["monthly_returns"]
        ls_all = np.array([m["LS"] for m in rets[:n]])
        q1_all = np.array([m["Q1"] for m in rets[:n]])
        q5_all = np.array([m["Q5"] for m in rets[:n]])

        label = data.get("label", key)

        # 카테고리 찾기
        cat = "기타"
        for c, keys in FACTOR_CATEGORIES.items():
            if key in keys:
                cat = c
                break

        def stats(arr):
            if len(arr) == 0:
                return 0, 0, 0, 0
            mean = arr.mean()
            std = arr.std() if len(arr) > 1 else 0
            sharpe = mean / std * np.sqrt(12) if std > 0 else 0
            win = (arr > 0).mean()
            return mean, std, sharpe, win

        # 전체
        m_all, s_all, sh_all, w_all = stats(ls_all)
        # 강세장
        ls_bull = ls_all[bull_idx]
        m_bull, s_bull, sh_bull, w_bull = stats(ls_bull)
        # 약세장
        ls_bear = ls_all[bear_idx]
        m_bear, s_bear, sh_bear, w_bear = stats(ls_bear)

        # Q1 (상위분위) 레짐별 수익률
        q1_bull_m = q1_all[bull_idx].mean() if len(bull_idx) > 0 else 0
        q1_bear_m = q1_all[bear_idx].mean() if len(bear_idx) > 0 else 0

        rows.append({
            "카테고리": cat,
            "팩터": label,
            "key": key,
            # 전체
            "전체_월평균": m_all,
            "전체_Sharpe": sh_all,
            "전체_승률": w_all,
            # 강세장
            "강세_월평균": m_bull,
            "강세_Sharpe": sh_bull,
            "강세_승률": w_bull,
            # 약세장
            "약세_월평균": m_bear,
            "약세_Sharpe": sh_bear,
            "약세_승률": w_bear,
            # 레짐 차이
            "강세-약세_차이": m_bull - m_bear,
            # Q1 수익률
            "Q1_강세_월평균": q1_bull_m,
            "Q1_약세_월평균": q1_bear_m,
        })

    df = pd.DataFrame(rows).sort_values("카테고리")

    # ═══════════════════════════════════════════════
    # 출력 1: 전체 요약
    # ═══════════════════════════════════════════════
    print("=" * 100)
    print("▶ 팩터별 강세장/약세장 L/S 수익률 분석")
    print("=" * 100)
    print(f"\n{'카테고리':<14} {'팩터':<28} {'전체':>8} {'강세장':>8} {'약세장':>8} {'차이':>8}  {'강세Sh':>7} {'약세Sh':>7}")
    print("-" * 100)

    for _, r in df.iterrows():
        print(f"{r['카테고리']:<14} {r['팩터']:<28} "
              f"{r['전체_월평균']*100:>+7.2f}% {r['강세_월평균']*100:>+7.2f}% "
              f"{r['약세_월평균']*100:>+7.2f}% {r['강세-약세_차이']*100:>+7.2f}%  "
              f"{r['강세_Sharpe']:>6.2f} {r['약세_Sharpe']:>6.2f}")

    # ═══════════════════════════════════════════════
    # 출력 2: 강세장 TOP 팩터
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("▶ 강세장에서 가장 효과적인 팩터 (월평균 L/S 수익률 기준)")
    print("=" * 80)
    top_bull = df.nlargest(7, "강세_월평균")
    for i, (_, r) in enumerate(top_bull.iterrows(), 1):
        print(f"  {i}. {r['팩터']:<28} 월평균 {r['강세_월평균']*100:>+.2f}%, "
              f"Sharpe {r['강세_Sharpe']:.2f}, 승률 {r['강세_승률']:.0%}")

    # ═══════════════════════════════════════════════
    # 출력 3: 약세장 TOP 팩터
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("▶ 약세장에서 가장 효과적인 팩터 (월평균 L/S 수익률 기준)")
    print("=" * 80)
    top_bear = df.nlargest(7, "약세_월평균")
    for i, (_, r) in enumerate(top_bear.iterrows(), 1):
        print(f"  {i}. {r['팩터']:<28} 월평균 {r['약세_월평균']*100:>+.2f}%, "
              f"Sharpe {r['약세_Sharpe']:.2f}, 승률 {r['약세_승률']:.0%}")

    # ═══════════════════════════════════════════════
    # 출력 4: 레짐별 차이가 큰 팩터 (레짐 타이밍 효과가 큰 팩터)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("▶ 레짐 민감도 (강세-약세 차이가 큰 팩터 → 레짐 전환 시 조절 대상)")
    print("=" * 80)
    regime_sens = df.reindex(df["강세-약세_차이"].abs().sort_values(ascending=False).index)
    for i, (_, r) in enumerate(regime_sens.head(7).iterrows(), 1):
        direction = "강세장 유리" if r["강세-약세_차이"] > 0 else "약세장 유리"
        print(f"  {i}. {r['팩터']:<28} 차이 {r['강세-약세_차이']*100:>+.2f}%p ({direction})")

    # ═══════════════════════════════════════════════
    # 출력 5: Q1(상위분위) 절대수익률 레짐별 비교
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("▶ Q1(상위 20%) 포트폴리오 절대수익률 레짐별 비교")
    print("=" * 80)
    print(f"{'팩터':<28} {'Q1 강세장':>10} {'Q1 약세장':>10} {'차이':>10}")
    print("-" * 62)
    for _, r in df.sort_values("Q1_약세_월평균", ascending=False).iterrows():
        print(f"{r['팩터']:<28} {r['Q1_강세_월평균']*100:>+9.2f}% "
              f"{r['Q1_약세_월평균']*100:>+9.2f}% "
              f"{(r['Q1_강세_월평균']-r['Q1_약세_월평균'])*100:>+9.2f}%")

    # ═══════════════════════════════════════════════
    # 출력 6: 제안
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("▶ 레짐별 팩터 가중치 제안")
    print("=" * 80)

    # 강세장: Sharpe > 0인 팩터 중 상위
    bull_candidates = df[df["강세_Sharpe"] > 0].nlargest(5, "강세_Sharpe")
    bear_candidates = df[df["약세_Sharpe"] > 0].nlargest(5, "약세_Sharpe")

    print("\n[강세장 추천 팩터]")
    for _, r in bull_candidates.iterrows():
        print(f"  ✓ {r['팩터']:<28} (Sharpe {r['강세_Sharpe']:.2f}, 승률 {r['강세_승률']:.0%})")

    print("\n[약세장 추천 팩터]")
    for _, r in bear_candidates.iterrows():
        print(f"  ✓ {r['팩터']:<28} (Sharpe {r['약세_Sharpe']:.2f}, 승률 {r['약세_승률']:.0%})")

    # 양쪽 모두 효과적인 팩터 (전천후)
    allweather = df[(df["강세_Sharpe"] > 0.2) & (df["약세_Sharpe"] > 0.2)]
    if len(allweather) > 0:
        print("\n[전천후 팩터 (강세/약세 모두 Sharpe > 0.2)]")
        for _, r in allweather.iterrows():
            print(f"  ★ {r['팩터']:<28} (강세 {r['강세_Sharpe']:.2f} / 약세 {r['약세_Sharpe']:.2f})")

    print("\n완료!")


if __name__ == "__main__":
    analyze()
