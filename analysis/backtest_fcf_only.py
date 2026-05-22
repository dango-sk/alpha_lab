"""
FCF-Bear 전략 단독 백테스트

Bear 구간: F_PBR 3% + FCF_YIELD 2% (cap30%_손절율15%(고점) 기반)
Bull 구간: 수정전략_코스피_cap30%_top30_tx30bp_월간 (기존 동일)

실행: python analysis/backtest_fcf_only.py
"""
import json
import os
import re
import sys
from datetime import datetime as _dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.db import get_conn, read_sql
from lib.factor_engine import (
    score_stocks_from_strategy, code_to_module, clear_factor_cache,
    load_factor_data, apply_quality_filter, run_regressions,
    apply_scoring, calc_weighted_scores,
)

BULL_KEY = "수정전략_코스피_cap30%_top30_tx30bp_월간"
BEAR_KEY = "cap30%_손절율15%(고점)"


# ═══════════════════════════════════════════════════════
# AI 레짐 로드
# ═══════════════════════════════════════════════════════

def _load_ai_regime() -> dict:
    """AI 레짐 결과 파일 로드 → {YYYY-MM: "Bull"/"Bear"}."""
    _env = os.environ.get("AI_REGIME_RESULTS_PATH", "")
    ai_path = Path(_env) if _env else Path(__file__).parent / "regime_agent_results.json"
    if not ai_path.exists():
        print("[REGIME] AI 레짐 파일 없음 → 전 구간 Bull로 처리")
        return {}

    er_map = {}
    with open(ai_path, encoding="utf-8") as f:
        for r in json.load(f):
            ym = r.get("as_of", "")[:7]
            er_map[ym] = r.get("expected_return", 0)

    regime_map: dict[str, str] = {}
    prev = "Bull"
    for ym in sorted(er_map):
        er = er_map[ym]
        if prev == "Bull":
            regime_map[ym] = "Bear" if er <= -2 else "Bull"
        else:
            regime_map[ym] = "Bull" if er >= 1 else "Bear"
        prev = regime_map[ym]

    bull_cnt = sum(1 for v in regime_map.values() if v == "Bull")
    print(f"[REGIME] {len(regime_map)}개월 로드 — Bull {bull_cnt} / Bear {len(regime_map)-bull_cnt}")
    return regime_map


# ═══════════════════════════════════════════════════════
# FCF 직접 로드 (DB)
# ═══════════════════════════════════════════════════════

def _fetch_all_fcf_from_db(conn) -> dict[str, dict[int, float]]:
    """
    fnspace_finance에서 Annual FCF 전체 로드.
    Returns: {stock_code: {fiscal_year: fcf(원)}}
    DB는 천원 단위로 저장, *1000 하여 원 단위로 변환.
    """
    df = read_sql("""
        SELECT stock_code, fiscal_year, fcf
        FROM fnspace_finance
        WHERE fiscal_quarter = 'Annual' AND fcf IS NOT NULL
        ORDER BY stock_code, fiscal_year
    """, conn)

    result: dict[str, dict[int, float]] = {}
    for _, row in df.iterrows():
        code    = str(row["stock_code"])
        yr      = int(row["fiscal_year"])
        fcf_won = float(row["fcf"]) * 1000  # 천원 → 원
        result.setdefault(code, {})[yr] = fcf_won

    print(f"[FCF] DB에서 {len(result)}종목 Annual FCF 로드")
    return result


# ═══════════════════════════════════════════════════════
# Bear 스코어링 (FCF_YIELD 직접 보완 포함)
# ═══════════════════════════════════════════════════════

def _score_bear_with_direct_fcf(
    conn,
    calc_date: str,
    bear_module,
    fcf_annual: dict[str, dict[int, float]],
) -> list[tuple[str, float]]:
    """
    Bear 모듈 스코어링.
    load_factor_data 결과에서 fcf_yield가 NaN인 종목에 대해
    fnspace_finance Annual FCF 직접 로드값으로 보완 후 재계산.
    """
    weights_large     = getattr(bear_module, "WEIGHTS_LARGE", {})
    weights_small     = getattr(bear_module, "WEIGHTS_SMALL", {})
    regression_models = getattr(bear_module, "REGRESSION_MODELS", [])
    outlier_filters   = getattr(bear_module, "OUTLIER_FILTERS", {})
    score_map         = getattr(bear_module, "SCORE_MAP", {})
    scoring_rules     = getattr(bear_module, "SCORING_RULES", {})
    scoring_mode      = getattr(bear_module, "SCORING_MODE", {"large": "quartile", "small": "decile"})
    quality_filter    = getattr(bear_module, "QUALITY_FILTER", {
        "exclude_spac_etf_reit": True,
        "require_positive_oi":   True,
        "require_positive_roe":  True,
        "min_avg_volume":        500_000_000,
    })
    params     = getattr(bear_module, "PARAMS", {})
    ma_rev_win = params.get("ma_reversion_window", None)

    # 1. 기존 파이프라인으로 팩터 데이터 로드
    df = load_factor_data(conn, calc_date, ma_reversion_window=ma_rev_win)
    if df is None or df.empty:
        return []

    # 2. look-ahead bias 기준연도 (파이프라인과 동일 로직)
    d      = _dt.strptime(calc_date, "%Y-%m-%d")
    max_yr = d.year - 1 if d.month >= 4 else d.year - 2

    # 3. fcf_yield NaN 보완: DB Annual FCF / market_cap
    if "market_cap" in df.columns:
        if "fcf_yield" not in df.columns:
            df["fcf_yield"] = float("nan")
        if "fcf" not in df.columns:
            df["fcf"] = float("nan")

        nan_mask = df["fcf_yield"].isna()
        filled = 0
        for idx in df.index[nan_mask]:
            code = df.at[idx, "stock_code"]
            mc   = df.at[idx, "market_cap"]
            if not isinstance(mc, (int, float)) or mc <= 0:
                continue
            code_fcf  = fcf_annual.get(code, {})
            valid_yrs = [yr for yr in code_fcf if yr <= max_yr]
            if not valid_yrs:
                continue
            fcf_won = code_fcf[max(valid_yrs)]
            df.at[idx, "fcf"]       = fcf_won
            df.at[idx, "fcf_yield"] = fcf_won / mc
            filled += 1

        if filled:
            print(f"    [FCF보완] {calc_date}: {filled}종목 fcf_yield 직접 계산으로 채움")

    # 4. 대형주 전용 (weights_small이 비어 있으면)
    if not weights_small:
        df = df[df["size_group"] == "large"].copy()

    # 5. 퀄리티 필터
    df = apply_quality_filter(df, quality_filter)
    df = df[df["quality_pass"] == 1].copy()

    if len(df) < 10:
        return []

    # 6. 회귀분석
    df, _ = run_regressions(df, regression_models, outlier_filters)

    # 7. 점수화
    df = apply_scoring(df, scoring_rules, scoring_mode)

    # 8. 가중합 → value_score
    df = calc_weighted_scores(df, weights_large, weights_small, score_map, scoring_mode)

    if "value_score" not in df.columns:
        return []

    df = df.sort_values("value_score", ascending=False).reset_index(drop=True)
    return list(zip(df["stock_code"], df["value_score"]))


# ═══════════════════════════════════════════════════════
# FCF-Bear 백테스트
# ═══════════════════════════════════════════════════════

def run_fcf_bear_backtest() -> dict:
    """
    FCF-Bear 백테스트 실행.
    Bull: 기존 Bull 전략 / Bear: F_PBR 3% + FCF_YIELD 2%
    """
    from lib.data import load_strategy
    from config.settings import BACKTEST_CONFIG
    from step7_backtest import run_backtest, get_universe_stocks

    bull_data = load_strategy(BULL_KEY)
    bear_data = load_strategy(BEAR_KEY)
    bull_module = code_to_module(bull_data["code"])

    # Bear 모듈: F_PBR 5% → 3%, FCF_YIELD 2% 추가 (가중치 합 1.0 유지)
    bear_modified_code = bear_data["code"].replace('"F_PBR": .05,', '"F_PBR": .03,')
    if '"FCF_YIELD"' not in bear_modified_code:
        bear_modified_code = re.sub(
            r'(WEIGHTS_LARGE\s*=\s*\{[^}]*)\}',
            r'\1    "FCF_YIELD": .02,\n}',
            bear_modified_code,
            count=1,
            flags=re.DOTALL,
        )
    else:
        bear_modified_code = bear_modified_code.replace('"FCF_YIELD": 0,', '"FCF_YIELD": .02,')
    bear_module = code_to_module(bear_modified_code)

    bull_params  = getattr(bull_module, "PARAMS", {})
    bear_params  = getattr(bear_module, "PARAMS", {})
    top_n        = bull_params.get("top_n", 30)
    tx_cost_bp   = int((bull_params.get("tx_cost_bp", 30) + bear_params.get("tx_cost_bp", 30)) / 2)
    bull_cap     = bull_params.get("weight_cap_pct", 30)
    bear_cap     = bear_params.get("weight_cap_pct", 30)
    bear_sl      = bear_params.get("stop_loss_enabled", False)
    bear_sl_pct  = bear_params.get("stop_loss_pct", 15)
    bear_sl_mode = bear_params.get("stop_loss_mode", "sell")

    ai_regime_map = _load_ai_regime()

    from lib.factor_engine import prefetch_all_data
    from step7_backtest import get_db as _get_db
    _pf_conn = _get_db()
    prefetch_all_data(_pf_conn)
    # Annual FCF를 DB에서 직접 로드 (prefetch cache 의존 없이)
    fcf_annual = _fetch_all_fcf_from_db(_pf_conn)
    _pf_conn.close()

    orig = {k: BACKTEST_CONFIG.get(k) for k in [
        "top_n_stocks", "transaction_cost_bp", "weight_cap_pct",
        "stop_loss_enabled", "stop_loss_pct", "stop_loss_mode",
        "universe", "rebal_type",
    ]}
    try:
        BACKTEST_CONFIG["top_n_stocks"]        = top_n
        BACKTEST_CONFIG["transaction_cost_bp"] = tx_cost_bp
        BACKTEST_CONFIG["universe"]            = "KOSPI"
        BACKTEST_CONFIG["rebal_type"]          = "monthly"
        BACKTEST_CONFIG["weight_cap_pct"]      = bull_cap
        BACKTEST_CONFIG["stop_loss_enabled"]   = False

        def selector(conn, calc_date, _top_n):
            regime      = ai_regime_map.get(calc_date[:7], "Bull")
            universe_set = get_universe_stocks(conn, calc_date, "monthly", 0)

            if regime == "Bull":
                BACKTEST_CONFIG["weight_cap_pct"]    = bull_cap
                BACKTEST_CONFIG["stop_loss_enabled"] = False
                cands = score_stocks_from_strategy(conn, calc_date, bull_module)
            else:
                BACKTEST_CONFIG["weight_cap_pct"]    = bear_cap
                BACKTEST_CONFIG["stop_loss_enabled"] = bear_sl
                BACKTEST_CONFIG["stop_loss_pct"]     = bear_sl_pct
                BACKTEST_CONFIG["stop_loss_mode"]    = bear_sl_mode
                # FCF_YIELD NaN 보완 포함된 Bear 스코어링 사용
                cands = _score_bear_with_direct_fcf(conn, calc_date, bear_module, fcf_annual)

            return [(c, s) for c, s in cands if c in universe_set][:_top_n]

        result = run_backtest("FCB_ONLY", stock_selector=selector, rebal_type="monthly")
        if result:
            result["strategy"] = "FCF-Bear"
        return result or {}

    finally:
        for k, v in orig.items():
            if v is not None:
                BACKTEST_CONFIG[k] = v


# ═══════════════════════════════════════════════════════
# 수익률 그래프
# ═══════════════════════════════════════════════════════

def plot_cumulative_returns(labeled_results: list[tuple[str, dict | None]]):
    """누적 수익률 그래프 저장 및 표시."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    _, ax = plt.subplots(figsize=(13, 6))

    styles = [
        ("FCF-Bear",  "darkorange", "-",  2.0),
        ("KODEX 200", "dimgray",    ":",  1.4),
    ]
    style_map = {s[0]: s[1:] for s in styles}

    for label, r in labeled_results:
        if not r:
            continue
        dates  = r.get("rebalance_dates", [])
        values = r.get("portfolio_values", [])
        if len(dates) != len(values) or not dates:
            continue
        color, ls, lw = style_map.get(label, ("black", "-", 1.2))
        ax.plot(dates, values, label=label, color=color, linestyle=ls, linewidth=lw)

    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (base = 1)")
    ax.set_title("FCF-Bear Strategy Performance")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fcf_only.png"
    plt.savefig(out_path, dpi=150)
    print(f"[CHART] Saved: {out_path}")
    plt.show()


# ═══════════════════════════════════════════════════════
# 결과 출력 헬퍼
# ═══════════════════════════════════════════════════════

def _fmt(r: dict | None, label: str):
    if not r:
        print(f"  {label:<36}  결과 없음")
        return
    print(
        f"  {label:<36}  "
        f"누적 {r['total_return']:>+7.1%}  "
        f"CAGR {r['cagr']:>+7.1%}  "
        f"MDD {r['mdd']:>6.1%}  "
        f"Sharpe {r['sharpe']:>5.2f}"
    )


# ═══════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════

def main():
    from step7_backtest import get_rebalance_dates, calc_all_benchmarks

    conn = get_conn()

    # ── FCF-Bear 단독 실행 ─────────────────────────────────
    print("\n" + "=" * 60)
    print("[FCF-Bear] Bear: F_PBR 3% + FCF_YIELD 2%")
    print("=" * 60)
    result_fcf = run_fcf_bear_backtest()
    clear_factor_cache()

    # ── 벤치마크 ────────────────────────────────────────────
    rebal_dates = get_rebalance_dates(conn, "monthly")
    bm = calc_all_benchmarks(conn, rebal_dates) if len(rebal_dates) >= 2 else {}
    conn.close()

    # ── 결과 출력 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("결과")
    print("=" * 60)
    print(f"  {'전략':<36}  {'누적':>8}  {'CAGR':>8}  {'MDD':>7}  {'Sharpe':>7}")
    print("  " + "-" * 70)
    _fmt(result_fcf, "FCF-Bear (F_PBR 3% + FCF_YIELD 2%)")
    bm_r = bm.get("KOSPI")
    if bm_r:
        print("  " + "-" * 70)
        _fmt(bm_r, "KODEX 200 (벤치마크)")
    print()

    # ── 그래프 ──────────────────────────────────────────────
    plot_cumulative_returns([
        ("FCF-Bear",  result_fcf),
        ("KODEX 200", bm_r),
    ])


if __name__ == "__main__":
    main()
