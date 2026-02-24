"""
팩터 리서치: 단일 팩터 롱숏 백테스트 (대형주)

각 팩터에 대해 5분위 포트폴리오를 구성하고,
Q1(롱) - Q5(숏) 스프레드의 수익률을 측정한다.

사용법:
  python -m analysis.factor_longshort              # F_PER 기본
  python -m analysis.factor_longshort --factor t_per
"""
import json
import sqlite3
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DB_PATH, BACKTEST_CONFIG, CACHE_DIR
from lib.factor_engine import load_factor_data, clear_factor_cache, run_regressions

# ─── 팩터 정의 ───
# (팩터명, 정렬방향): "asc" = 낮을수록 좋음 (PER, PBR 등), "desc" = 높을수록 좋음 (성장률 등)
FACTOR_DEFS = {
    # 밸류 멀티플 (낮을수록 좋음)
    "f_per":       {"sort": "asc",  "label": "Forward PER"},
    "t_per":       {"sort": "asc",  "label": "Trailing PER"},
    "pbr":         {"sort": "asc",  "label": "PBR"},
    "f_pbr":       {"sort": "asc",  "label": "Forward PBR"},
    "t_ev_ebitda": {"sort": "asc",  "label": "Trailing EV/EBITDA"},
    "f_ev_ebitda": {"sort": "asc",  "label": "Forward EV/EBITDA"},
    "t_pcf":       {"sort": "asc",  "label": "Trailing PCF"},
    "ev_ic":       {"sort": "asc",  "label": "EV/IC"},
    # 성장 (높을수록 좋음)
    "f_epsg":      {"sort": "desc", "label": "Forward EPS Growth"},
    "f_ebitg":     {"sort": "desc", "label": "Forward EBIT Growth"},
    "t_spsg":      {"sort": "desc", "label": "Trailing 매출 성장률"},
    "f_spsg":      {"sort": "desc", "label": "Forward 매출 성장률"},
    # 모멘텀 (높을수록 좋음)
    "f_eps_m":     {"sort": "desc", "label": "Forward EPS 모멘텀 (3M)"},
    "price_m":     {"sort": "desc", "label": "가격 모멘텀 (3M)"},
    # 레버리지 (낮을수록 좋음)
    "ndebt_ebitda": {"sort": "asc", "label": "순부채/EBITDA"},
    # 회귀 매력도 (높을수록 좋음 = 펀더멘털 대비 저평가)
    "pbr_roe_attractiveness":       {"sort": "desc", "label": "ATT: PBR~ROE", "needs_regression": True},
    "evic_roic_attractiveness":     {"sort": "desc", "label": "ATT: EV/IC~ROIC", "needs_regression": True},
    "fper_epsg_attractiveness":     {"sort": "desc", "label": "ATT: F_PER~EPSG", "needs_regression": True},
    "fevebit_ebitg_attractiveness": {"sort": "desc", "label": "ATT: F_EV/EBIT~EBITG", "needs_regression": True},
}

# step3와 동일한 회귀 모델 설정
REGRESSION_MODELS = [
    ("pbr_roe",       "roe",    "pbr",      "ratio"),
    ("evic_roic",     "roic",   "ev_ic",    "ev_equity"),
    ("fper_epsg",     "f_epsg", "f_per",    "ratio"),
    ("fevebit_ebitg", "f_ebitg","f_ev_ebit","ev_equity_ebit"),
]
OUTLIER_FILTERS = {
    "pbr_roe":       {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 20},
    "evic_roic":     {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 51},
    "fper_epsg":     {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60},
    "fevebit_ebitg": {"x_min": 0, "x_max": 500, "y_min": 0, "y_max": 60},
}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_monthly_rebalance_dates(conn):
    """매월 첫 거래일 리스트"""
    trade_dates = conn.execute("""
        SELECT DISTINCT trade_date FROM daily_price
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, (BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"])).fetchall()

    monthly = []
    current_month = ""
    for (td,) in trade_dates:
        month = td[:7]
        if month != current_month:
            monthly.append(td)
            current_month = month

    if trade_dates:
        last_trade = trade_dates[-1][0]
        if monthly and last_trade > monthly[-1]:
            monthly.append(last_trade)

    return monthly


def get_stock_return(conn, stock_code, start_date, end_date):
    """종목의 기간 수익률 계산"""
    # daily_price의 stock_code는 'A' prefix 없음
    code = stock_code.lstrip("A")

    start_price = conn.execute("""
        SELECT close FROM daily_price
        WHERE stock_code = ? AND trade_date >= ?
        ORDER BY trade_date ASC LIMIT 1
    """, (code, start_date)).fetchone()

    end_price = conn.execute("""
        SELECT close FROM daily_price
        WHERE stock_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT 1
    """, (code, end_date)).fetchone()

    if not start_price or not end_price or start_price[0] <= 0:
        return None

    return (end_price[0] - start_price[0]) / start_price[0]


def run_factor_longshort(factor_name="f_per", n_quantiles=5):
    """단일 팩터 롱숏 백테스트 실행"""

    if factor_name not in FACTOR_DEFS:
        print(f"  지원하지 않는 팩터: {factor_name}")
        print(f"  사용 가능: {', '.join(FACTOR_DEFS.keys())}")
        return None

    fdef = FACTOR_DEFS[factor_name]
    ascending = fdef["sort"] == "asc"  # asc면 낮은 값이 Q1 (롱)

    conn = get_db()
    dates = get_monthly_rebalance_dates(conn)
    total_periods = len(dates) - 1

    print(f"\n{'='*60}")
    print(f"  팩터 롱숏 백테스트: {fdef['label']} ({factor_name})")
    print(f"  기간: {dates[0]} ~ {dates[-1]} ({total_periods}개월)")
    print(f"  대형주 | 동일가중 | {n_quantiles}분위 | 거래비용 없음")
    print(f"{'='*60}\n")

    # 월별 결과 저장
    monthly_results = []

    for i in range(total_periods):
        calc_date = dates[i]
        next_date = dates[i + 1]

        # 팩터 데이터 로드
        df = load_factor_data(conn, calc_date)
        if df is None:
            print(f"  [{calc_date}] 데이터 없음, 스킵")
            continue

        # 회귀 매력도 팩터인 경우 회귀분석 실행
        if fdef.get("needs_regression"):
            df, _ = run_regressions(df, REGRESSION_MODELS, OUTLIER_FILTERS)

        # 대형주 + 팩터값 존재 필터
        large = df[
            (df["size_group"] == "large") &
            (df[factor_name].notna()) &
            (df["operating_income"].notna()) & (df["operating_income"] > 0) &
            (df["roe"].notna()) & (df["roe"] > 0)
        ].copy()

        if len(large) < n_quantiles * 5:
            print(f"  [{calc_date}] 종목 수 부족 ({len(large)}), 스킵")
            continue

        # 팩터 기준 정렬 → 분위 배정
        large = large.sort_values(factor_name, ascending=ascending).reset_index(drop=True)
        large["quantile"] = pd.qcut(
            range(len(large)), n_quantiles, labels=range(1, n_quantiles + 1)
        )

        # 각 분위별 동일가중 수익률
        q_returns = {}
        for q in range(1, n_quantiles + 1):
            q_stocks = large[large["quantile"] == q]["stock_code"].tolist()
            rets = []
            for code in q_stocks:
                r = get_stock_return(conn, code, calc_date, next_date)
                if r is not None:
                    rets.append(r)

            q_returns[f"Q{q}"] = np.mean(rets) if rets else 0.0

        # 롱숏 스프레드
        ls_return = q_returns["Q1"] - q_returns[f"Q{n_quantiles}"]

        monthly_results.append({
            "date": calc_date,
            "n_stocks": len(large),
            **q_returns,
            "LS": ls_return,
        })

        q1_pct = q_returns["Q1"] * 100
        q5_pct = q_returns[f"Q{n_quantiles}"] * 100
        ls_pct = ls_return * 100
        print(f"  [{calc_date}] N={len(large):3d}  Q1={q1_pct:+6.2f}%  Q5={q5_pct:+6.2f}%  L/S={ls_pct:+6.2f}%")

    conn.close()
    clear_factor_cache()

    if not monthly_results:
        print("  결과 없음")
        return None

    # ─── 통계 집계 ───
    results_df = pd.DataFrame(monthly_results)
    summary = compute_statistics(results_df, n_quantiles)
    summary["factor"] = factor_name
    summary["label"] = fdef["label"]
    summary["monthly_results"] = monthly_results

    print_summary(summary, n_quantiles)
    save_results(summary, factor_name)

    return summary


def compute_statistics(df, n_quantiles):
    """롱숏 수익률 통계 계산"""
    ls = df["LS"].values
    n = len(ls)

    mean_monthly = np.mean(ls)
    std_monthly = np.std(ls, ddof=1)
    t_stat, p_value = stats.ttest_1samp(ls, 0)
    sharpe = (mean_monthly / std_monthly * np.sqrt(12)) if std_monthly > 0 else 0

    # 누적 수익률
    cumulative_ls = np.cumprod(1 + ls) - 1
    cumulative_q = {}
    for q in range(1, n_quantiles + 1):
        q_rets = df[f"Q{q}"].values
        cumulative_q[f"Q{q}"] = float(np.cumprod(1 + q_rets)[-1] - 1)

    # 최대 연속 손실
    max_consecutive_loss = 0
    current_loss = 0
    for r in ls:
        if r < 0:
            current_loss += 1
            max_consecutive_loss = max(max_consecutive_loss, current_loss)
        else:
            current_loss = 0

    return {
        "n_months": n,
        "mean_monthly": float(mean_monthly),
        "annualized": float((1 + mean_monthly) ** 12 - 1),
        "std_monthly": float(std_monthly),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "sharpe": float(sharpe),
        "cumulative_ls": float(cumulative_ls[-1]),
        "cumulative_q": cumulative_q,
        "win_rate": float(np.mean(ls > 0)),
        "max_consecutive_loss": max_consecutive_loss,
        "avg_stocks_per_month": float(df["n_stocks"].mean()),
    }


def print_summary(s, n_quantiles):
    """결과 요약 출력"""
    print(f"\n{'='*60}")
    print(f"  {s['label']} ({s['factor']}) 롱숏 결과")
    print(f"{'='*60}")
    print(f"  기간: {s['n_months']}개월 | 평균 종목 수: {s['avg_stocks_per_month']:.0f}")
    print()

    # 분위별 누적 수익률
    print(f"  분위별 누적 수익률:")
    for q in range(1, n_quantiles + 1):
        cum = s["cumulative_q"][f"Q{q}"] * 100
        bar = "█" * int(abs(cum) / 5)
        sign = "+" if cum >= 0 else ""
        print(f"    Q{q}: {sign}{cum:6.1f}%  {bar}")

    print()
    print(f"  롱숏 (Q1-Q5):")
    print(f"    월평균 수익률: {s['mean_monthly']*100:+.2f}%")
    print(f"    연환산 수익률: {s['annualized']*100:+.1f}%")
    print(f"    월 변동성:     {s['std_monthly']*100:.2f}%")
    print(f"    Sharpe Ratio:  {s['sharpe']:.3f}")
    print(f"    t-stat:        {s['t_stat']:.3f}")
    print(f"    p-value:       {s['p_value']:.4f}")
    sig = "유의 (p < 0.05)" if s["p_value"] < 0.05 else "유의하지 않음"
    print(f"    통계적 유의성: {sig}")
    print(f"    누적 수익률:   {s['cumulative_ls']*100:+.1f}%")
    print(f"    승률:          {s['win_rate']*100:.1f}%")
    print(f"    최대 연속 손실: {s['max_consecutive_loss']}개월")
    print(f"{'='*60}\n")


def save_results(summary, factor_name):
    """결과 JSON 저장"""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"factor_research_{factor_name}.json"

    output = {k: v for k, v in summary.items() if k != "monthly_results"}
    output["monthly_returns"] = [
        {"date": r["date"], "LS": r["LS"], "Q1": r["Q1"],
         f"Q{len(r)-3}": r[f"Q{len(r)-3}"] if f"Q{len(r)-3}" in r else None}
        for r in summary["monthly_results"]
    ]
    # 전체 월별 데이터 저장
    output["monthly_detail"] = summary["monthly_results"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장: {path}")


# 대형주 전략(A0)에 실제 사용되는 팩터만
LARGE_CAP_FACTORS = [
    "f_per", "t_per", "pbr", "f_pbr",
    "t_ev_ebitda", "f_ev_ebitda", "t_pcf", "ev_ic",
    "f_epsg", "t_spsg", "f_spsg",
    "f_eps_m", "price_m",
    "ndebt_ebitda",
    "pbr_roe_attractiveness", "evic_roic_attractiveness",
    "fper_epsg_attractiveness", "fevebit_ebitg_attractiveness",
]


def run_all_factors(n_quantiles=5):
    """대형주 팩터 전체 롱숏 백테스트 → 요약 테이블"""
    results = []

    for factor in LARGE_CAP_FACTORS:
        print(f"\n{'─'*60}")
        print(f"  >>> {FACTOR_DEFS[factor]['label']} ({factor})")
        print(f"{'─'*60}")
        s = run_factor_longshort(factor, n_quantiles)
        if s:
            results.append(s)

    if not results:
        print("  결과 없음")
        return

    # ─── 요약 테이블 ───
    print(f"\n{'='*90}")
    print(f"  대형주 팩터 롱숏 요약 (Q1-Q5, 동일가중, {results[0]['n_months']}개월)")
    print(f"{'='*90}")
    print(f"  {'팩터':<20} {'월평균':>7} {'연환산':>7} {'누적':>7} {'Sharpe':>7} {'t-stat':>7} {'p-value':>8} {'승률':>6} {'유의':>4}")
    print(f"  {'─'*84}")

    # p-value 기준 정렬
    results.sort(key=lambda x: x["p_value"])

    for s in results:
        sig = "**" if s["p_value"] < 0.05 else "* " if s["p_value"] < 0.10 else "  "
        print(
            f"  {s['label']:<20} "
            f"{s['mean_monthly']*100:>+6.2f}% "
            f"{s['annualized']*100:>+6.1f}% "
            f"{s['cumulative_ls']*100:>+6.1f}% "
            f"{s['sharpe']:>7.3f} "
            f"{s['t_stat']:>7.3f} "
            f"{s['p_value']:>8.4f} "
            f"{s['win_rate']*100:>5.1f}% "
            f"{sig}"
        )

    print(f"  {'─'*84}")
    print(f"  ** p<0.05 (유의)  * p<0.10 (약한 유의)")
    print(f"{'='*90}\n")

    # 전체 결과 저장
    CACHE_DIR.mkdir(exist_ok=True)
    all_summary = [
        {k: v for k, v in s.items() if k != "monthly_results"}
        for s in results
    ]
    path = CACHE_DIR / "factor_research_all.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)
    print(f"  전체 결과 저장: {path}")


def run_combo_longshort(factors, weights=None, combo_name="combo", n_quantiles=5, reverse=False, flip_factors=None):
    """복합 팩터 롱숏 백테스트.
    각 팩터의 percentile rank를 구한 뒤 가중 합산하여 복합 시그널 생성.
    flip_factors: 개별 팩터의 정렬 방향을 뒤집을 팩터 리스트
    """
    if flip_factors is None:
        flip_factors = []
    if weights is None:
        weights = [1.0 / len(factors)] * len(factors)

    # 팩터 유효성 확인
    for f in factors:
        if f not in FACTOR_DEFS:
            print(f"  지원하지 않는 팩터: {f}")
            return None

    needs_regression = any(FACTOR_DEFS[f].get("needs_regression") for f in factors)
    labels = [FACTOR_DEFS[f]["label"] for f in factors]
    weight_strs = [f"{w*100:.0f}%" for w in weights]

    conn = get_db()
    dates = get_monthly_rebalance_dates(conn)
    total_periods = len(dates) - 1

    print(f"\n{'='*60}")
    print(f"  복합 팩터 롱숏 백테스트: {combo_name}")
    for lbl, f, ws in zip(labels, factors, weight_strs):
        flipped = " [FLIPPED]" if f in flip_factors else ""
        print(f"    {ws} {lbl} ({f}){flipped}")
    direction = "REVERSE (낮은 점수 = 롱)" if reverse else "정방향 (높은 점수 = 롱)"
    print(f"  기간: {dates[0]} ~ {dates[-1]} ({total_periods}개월)")
    print(f"  대형주 | 동일가중 | {n_quantiles}분위 | 거래비용 없음")
    print(f"  방향: {direction}")
    print(f"{'='*60}\n")

    monthly_results = []

    for i in range(total_periods):
        calc_date = dates[i]
        next_date = dates[i + 1]

        df = load_factor_data(conn, calc_date)
        if df is None:
            continue

        if needs_regression:
            df, _ = run_regressions(df, REGRESSION_MODELS, OUTLIER_FILTERS)

        # 대형주 + 퀄리티 필터
        large = df[
            (df["size_group"] == "large") &
            (df["operating_income"].notna()) & (df["operating_income"] > 0) &
            (df["roe"].notna()) & (df["roe"] > 0)
        ].copy()

        if len(large) < n_quantiles * 5:
            print(f"  [{calc_date}] 종목 수 부족 ({len(large)}), 스킵")
            continue

        # 팩터별 percentile rank → 가중 합산 (결측은 0.5=neutral로 채움)
        large["combo_score"] = 0.0
        for f, w in zip(factors, weights):
            ascending = FACTOR_DEFS[f]["sort"] == "asc"
            if f in flip_factors:
                ascending = not ascending  # 개별 팩터 방향 반전
            ranks = large[f].rank(ascending=ascending, pct=True)
            large[f"_rank_{f}"] = ranks.fillna(0.5)
            large["combo_score"] += large[f"_rank_{f}"] * w

        # combo_score 기준 정렬 (reverse=True면 낮은 점수가 Q1)
        large = large.sort_values("combo_score", ascending=reverse).reset_index(drop=True)
        large["quantile"] = pd.qcut(
            range(len(large)), n_quantiles, labels=range(1, n_quantiles + 1)
        )

        q_returns = {}
        for q in range(1, n_quantiles + 1):
            q_stocks = large[large["quantile"] == q]["stock_code"].tolist()
            rets = []
            for code in q_stocks:
                r = get_stock_return(conn, code, calc_date, next_date)
                if r is not None:
                    rets.append(r)
            q_returns[f"Q{q}"] = np.mean(rets) if rets else 0.0

        ls_return = q_returns["Q1"] - q_returns[f"Q{n_quantiles}"]

        monthly_results.append({
            "date": calc_date,
            "n_stocks": len(large),
            **q_returns,
            "LS": ls_return,
        })

        q1_pct = q_returns["Q1"] * 100
        q5_pct = q_returns[f"Q{n_quantiles}"] * 100
        ls_pct = ls_return * 100
        print(f"  [{calc_date}] N={len(large):3d}  Q1={q1_pct:+6.2f}%  Q5={q5_pct:+6.2f}%  L/S={ls_pct:+6.2f}%")

    conn.close()
    clear_factor_cache()

    if not monthly_results:
        print("  결과 없음")
        return None

    results_df = pd.DataFrame(monthly_results)
    summary = compute_statistics(results_df, n_quantiles)
    summary["factor"] = combo_name
    summary["label"] = combo_name
    summary["monthly_results"] = monthly_results

    print_summary(summary, n_quantiles)
    save_results(summary, combo_name)

    return summary


# ─── A0 전략 팩터/가중치 매핑 ───
# WEIGHTS_LARGE (step3) → factor_longshort.py 팩터명
A0_FACTORS = [
    "t_per", "f_per", "t_ev_ebitda", "f_ev_ebitda",
    "pbr", "f_pbr", "t_pcf",
    "pbr_roe_attractiveness", "evic_roic_attractiveness",
    "fper_epsg_attractiveness", "fevebit_ebitg_attractiveness",
    "t_spsg", "f_spsg",
    "f_eps_m",
]
A0_WEIGHTS = [
    0.05, 0.05, 0.05, 0.05,
    0.05, 0.05, 0.05,
    0.05, 0.05, 0.10, 0.10,
    0.10, 0.10,
    0.15,
]
GROWTH_FACTORS = ["f_eps_m", "t_spsg", "f_spsg"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="팩터 롱숏 백테스트")
    parser.add_argument("--factor", default="f_per", help=f"팩터명 ({', '.join(FACTOR_DEFS.keys())})")
    parser.add_argument("--quantiles", type=int, default=5, help="분위 수 (기본: 5)")
    parser.add_argument("--all", action="store_true", help="대형주 팩터 전체 실행")
    parser.add_argument("--combo", type=str, help="복합 팩터 (쉼표 구분, 예: f_eps_m,t_spsg,f_spsg)")
    parser.add_argument("--reverse", action="store_true", help="콤보 정렬 방향 반전 (낮은 점수 = 롱)")
    parser.add_argument("--a0", action="store_true", help="A0 전략 14개 팩터 콤보 (원래 가중치)")
    parser.add_argument("--flip-growth", action="store_true", help="성장 팩터 3개 방향 반전 (저성장 = 좋음)")
    args = parser.parse_args()

    if args.all:
        run_all_factors(args.quantiles)
    elif args.a0:
        flip = GROWTH_FACTORS if args.flip_growth else []
        name = "A0_flip_growth" if args.flip_growth else "A0_original"
        run_combo_longshort(
            A0_FACTORS, A0_WEIGHTS,
            combo_name=name, n_quantiles=args.quantiles, flip_factors=flip,
        )
    elif args.combo:
        combo_factors = args.combo.split(",")
        name = args.combo.replace(",", "+")
        flip = GROWTH_FACTORS if args.flip_growth else []
        run_combo_longshort(
            combo_factors, combo_name=name,
            n_quantiles=args.quantiles, reverse=args.reverse, flip_factors=flip,
        )
    else:
        run_factor_longshort(args.factor, args.quantiles)
