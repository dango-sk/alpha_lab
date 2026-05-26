"""
Bear 구간 첫 두 달 — 상위 5종목 FCF-Bear value_score 산출과정 진단

실행: python analysis/debug_fcf_scores.py
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.db import get_conn, read_sql
from lib.factor_engine import (
    load_factor_data, apply_quality_filter, run_regressions,
    apply_scoring, calc_weighted_scores, code_to_module,
    _prefetch_cache, _check_fcf_column,
)
from lib.data import load_strategy

BULL_KEY = "수정전략_코스피_cap30%_top30_tx30bp_월간"
BEAR_KEY = "cap30%_손절율15%(고점)"
TOP_N    = 5


# ── AI 레짐 로드 ──────────────────────────────────────────────────────────────

def _load_ai_regime() -> dict:
    _env = os.environ.get("AI_REGIME_RESULTS_PATH", "")
    ai_path = Path(_env) if _env else Path(__file__).parent / "regime_agent_results.json"
    if not ai_path.exists():
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
    return regime_map


# ── Bear 모듈 준비 ────────────────────────────────────────────────────────────

def _build_bear_module():
    bear_data = load_strategy(BEAR_KEY)
    code = bear_data["code"]

    code = code.replace('"F_PBR": .05,', '"F_PBR": .03,')
    if '"FCF_YIELD"' not in code:
        code = re.sub(
            r'(WEIGHTS_LARGE\s*=\s*\{[^}]*)\}',
            r'\1    "FCF_YIELD": .02,\n}',
            code, count=1, flags=re.DOTALL,
        )
    else:
        code = code.replace('"FCF_YIELD": 0,', '"FCF_YIELD": .02,')

    # SCORE_MAP에 FCF_YIELD → fcf_yield_score 매핑 추가
    if '"fcf_yield_score"' not in code:
        code = re.sub(
            r'(SCORE_MAP\s*=\s*\{[^}]*)\}',
            r'\1    "FCF_YIELD": "fcf_yield_score",\n}',
            code, count=1, flags=re.DOTALL,
        )

    # SCORING_RULES에 fcf_yield 점수화 규칙 추가 (높을수록 좋음 → rule2)
    if '"fcf_yield"' not in code:
        code = re.sub(
            r'(SCORING_RULES\s*=\s*\{[^}]*)\}',
            r'\1    "fcf_yield": "rule2",\n}',
            code, count=1, flags=re.DOTALL,
        )

    mod = code_to_module(code)

    # 실제 WEIGHTS_LARGE 확인
    print("▶ Bear 모듈 WEIGHTS_LARGE:")
    for k, v in getattr(mod, "WEIGHTS_LARGE", {}).items():
        if v > 0:
            print(f"   {k:<20} {v:.0%}")
    print()
    return mod


# ── DB에서 Annual FCF 직접 로드 ───────────────────────────────────────────────

def _fetch_all_fcf_from_db(conn) -> dict[str, dict[int, float]]:
    """fnspace_finance Annual FCF 전체 로드 → {stock_code: {fiscal_year: fcf(원)}}"""
    df = read_sql("""
        SELECT stock_code, fiscal_year, fcf
        FROM fnspace_finance
        WHERE fiscal_quarter = 'Annual' AND fcf IS NOT NULL
        ORDER BY stock_code, fiscal_year
    """, conn)
    result: dict[str, dict[int, float]] = {}
    for _, row in df.iterrows():
        result.setdefault(str(row["stock_code"]), {})[int(row["fiscal_year"])] = float(row["fcf"]) * 1000
    print(f"[FCF] {len(result)}종목 Annual FCF 로드")
    return result


# ── 단계별 파이프라인 (score_stocks_from_strategy 내부를 그대로 재현) ────────

def _score_with_breakdown(conn, calc_date: str, bear_module,
                          fcf_annual: dict[str, dict[int, float]]) -> pd.DataFrame:
    weights_large   = getattr(bear_module, "WEIGHTS_LARGE", {})
    weights_small   = getattr(bear_module, "WEIGHTS_SMALL", {})
    regression_models = getattr(bear_module, "REGRESSION_MODELS", [])
    outlier_filters = getattr(bear_module, "OUTLIER_FILTERS", {})
    score_map       = getattr(bear_module, "SCORE_MAP", {})
    scoring_rules   = getattr(bear_module, "SCORING_RULES", {})
    scoring_mode    = getattr(bear_module, "SCORING_MODE", {"large": "quartile", "small": "decile"})
    quality_filter  = getattr(bear_module, "QUALITY_FILTER", {
        "exclude_spac_etf_reit": True,
        "require_positive_oi": True,
        "require_positive_roe": True,
        "min_avg_volume": 500_000_000,
    })
    params      = getattr(bear_module, "PARAMS", {})
    ma_rev_win  = params.get("ma_reversion_window", None)

    # 1. 팩터 데이터 로드
    df = load_factor_data(conn, calc_date, ma_reversion_window=ma_rev_win)
    if df is None or df.empty:
        return pd.DataFrame()

    # 1-b. fcf_yield NaN 보완 (DB Annual FCF 직접 사용)
    dt = pd.Timestamp(calc_date)
    max_yr = dt.year - 1 if dt.month >= 4 else dt.year - 2
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
            valid_yrs = [yr for yr in fcf_annual.get(code, {}) if yr <= max_yr]
            if not valid_yrs:
                continue
            fcf_won = fcf_annual[code][max(valid_yrs)]
            df.at[idx, "fcf"]       = fcf_won
            df.at[idx, "fcf_yield"] = fcf_won / mc
            filled += 1
        if filled:
            print(f"  [FCF보완] {filled}종목 fcf_yield 직접 계산으로 채움")

    # ── 진단 출력 ──
    total = len(df)
    fcf_notnull  = df["fcf"].notna().sum()       if "fcf"       in df.columns else "컬럼없음"
    mktcap_ok    = (df["market_cap"] > 0).sum()  if "market_cap" in df.columns else "컬럼없음"
    yield_notnull= df["fcf_yield"].notna().sum() if "fcf_yield"  in df.columns else "컬럼없음"
    print(f"  [진단] 전체 {total}종목 | fcf 비null={fcf_notnull} | market_cap>0={mktcap_ok} | fcf_yield 비null={yield_notnull}")

    # 특정 종목 fcf/fcf_yield 직접 확인
    check_codes = ["A000270", "A004170", "A069960"]
    if "stock_code" in df.columns:
        sample = df[df["stock_code"].isin(check_codes)][["stock_code", "fcf", "market_cap", "fcf_yield"]]
        if not sample.empty:
            print(f"  [진단] 샘플 종목 fcf/fcf_yield:")
            print(sample.to_string(index=False))

    # 2. 대형주 전용
    if not weights_small:
        df = df[df["size_group"] == "large"].copy()
    fcf_large = df["fcf_yield"].notna().sum() if "fcf_yield" in df.columns else "N/A"
    print(f"  [진단] 대형주 필터 후: {len(df)}종목 | fcf_yield 비null={fcf_large}")

    # 3. 퀄리티 필터
    df = apply_quality_filter(df, quality_filter)
    df = df[df["quality_pass"] == 1].copy()
    fcf_qual = df["fcf_yield"].notna().sum() if "fcf_yield" in df.columns else "N/A"
    print(f"  [진단] 퀄리티 필터 후: {len(df)}종목 | fcf_yield 비null={fcf_qual}")

    if len(df) < 10:
        return pd.DataFrame()

    # 4. 회귀분석
    df, _ = run_regressions(df, regression_models, outlier_filters)

    # 5. 점수화
    df = apply_scoring(df, scoring_rules, scoring_mode)

    # 6. 가중합 → value_score
    df = calc_weighted_scores(df, weights_large, weights_small, score_map, scoring_mode)

    return df


# ── 결과 출력 ─────────────────────────────────────────────────────────────────

def _print_breakdown(calc_date: str, df: pd.DataFrame, bear_module):
    weights_large = getattr(bear_module, "WEIGHTS_LARGE", {})
    score_map     = getattr(bear_module, "SCORE_MAP", {})
    scoring_mode  = getattr(bear_module, "SCORING_MODE", {"large": "quartile", "small": "decile"})

    # 사용 중인 팩터만 (가중치 > 0)
    active = {k: v for k, v in weights_large.items() if v > 0}

    top5 = df.nlargest(TOP_N, "value_score").copy()
    top5["stock_code"] = top5["stock_code"].str.lstrip("A")

    mode_large = scoring_mode.get("large", "quartile")
    max_raw = 4.0 if mode_large == "quartile" else 10.0

    print(f"\n{'─'*80}")
    print(f"  날짜: {calc_date}  |  대형주 스코어링 모드: {mode_large}  |  점수 최대값: {max_raw}")
    print(f"{'─'*80}")

    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        code = row["stock_code"]
        name = row.get("stock_name", "")
        print(f"\n  [{rank}위]  {code}  {name}  →  value_score: {row['value_score']:.1f}")
        print(f"  {'팩터':<20} {'원시값':>10}  {'점수':>6}  {'가중치':>6}  {'기여도':>8}")
        print(f"  {'─'*58}")

        total_check = 0.0
        for factor_key, weight in active.items():
            score_col = score_map.get(factor_key, "")
            score_val = row.get(score_col, 0) if score_col else 0

            # 원시 팩터값 (score_col에서 _score 제거 → 원시 컬럼명 추정)
            raw_col = score_col.replace("_score", "") if score_col else ""
            raw_val = row.get(raw_col, float("nan"))
            raw_str = f"{raw_val:.4f}" if pd.notna(raw_val) else "NaN"

            contribution = score_val * weight
            total_check += contribution

            fcf_mark = "  ◀ FCF" if factor_key == "FCF_YIELD" else ""
            print(f"  {factor_key:<20} {raw_str:>10}  {score_val:>6.1f}  {weight:>6.1%}  {contribution:>8.4f}{fcf_mark}")

        print(f"  {'─'*58}")
        print(f"  {'합계 (total_raw)':<20} {'':>10}  {'':>6}  {'':>6}  {total_check:>8.4f}")
        print(f"  {'value_score 계산':<20} {'':>10}  {'':>6}  {'':>6}  {total_check/max_raw*100:>7.1f}  (÷{max_raw}×100)")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def _do_prefetch(conn, fwd_start: str, fwd_end: str):
    """fnspace_forward를 분석 기간만 로드하는 경량 prefetch.
    전체 로드 시 Railway 연결이 끊기는 문제 방지."""
    _fcf_sel = ", fcf" if _check_fcf_column(conn) else ""
    print("[PREFETCH] finance 로드 중...", flush=True)
    _prefetch_cache["finance"] = read_sql(f"""
        SELECT stock_code, fiscal_year, fiscal_quarter,
               pbr, roe, roic, ev, ic, ev_ebit, ebit, ebitda,
               net_debt, interest_debt, total_equity,
               eps, bps, per, psr, ev_ebitda,
               revenue, operating_income, net_income,
               oi_margin, div_yield, pcf{_fcf_sel}
        FROM fnspace_finance
        WHERE fiscal_quarter IN ('Annual', 'TTM_1Q', 'TTM_2Q', 'TTM_3Q', 'TTM_4Q')
    """, conn)
    print(f"[PREFETCH] forward 로드 중 ({fwd_start} ~ {fwd_end})...", flush=True)
    _prefetch_cache["forward"] = read_sql(f"""
        SELECT stock_code, trade_date, fwd_eps, fwd_per, fwd_ebit, fwd_ebitda,
               fwd_ev_ebitda, fwd_revenue, fwd_oi, fwd_ni, fwd_roe, fwd_bps
        FROM fnspace_forward
        WHERE trade_date BETWEEN '{fwd_start}' AND '{fwd_end}'
    """, conn)
    print(f"[PREFETCH] price 로드 중 ({fwd_start} ~ {fwd_end})...", flush=True)
    _prefetch_cache["price"] = read_sql(f"""
        SELECT 'A' || stock_code as stock_code, trade_date,
               close, high, low, volume, market_cap, trade_amount
        FROM daily_price
        WHERE trade_date BETWEEN '{fwd_start}' AND '{fwd_end}'
    """, conn)
    print("[PREFETCH] master 로드 중...", flush=True)
    _prefetch_cache["master"] = read_sql("""
        SELECT stock_code, stock_name, market, sec_cd_nm, finacc_typ, snapshot_date
        FROM fnspace_master
    """, conn)
    print("[PREFETCH] 완료", flush=True)


def main():
    # regime 먼저 로드해서 분석 대상 날짜 파악 (DB 불필요)
    regime_map = _load_ai_regime()
    bear_months = sorted(ym for ym, r in regime_map.items() if r == "Bear")

    if not bear_months:
        print("Bear 구간 없음")
        return

    bear_months_2020 = [ym for ym in bear_months if ym >= "2020-01"]
    first_two = bear_months_2020[:2]
    print(f"전체 Bear 구간: {bear_months}")
    print(f"2020년 이후 Bear 구간: {bear_months_2020}")
    print(f"분석 대상 첫 두 달: {first_two}")

    # forward 로드 범위: 분석 달 전후 3개월
    fwd_start = f"{int(first_two[0][:4]) - 1}-10-01" if first_two else "2019-10-01"
    fwd_end   = f"{first_two[-1][:7]}-28"             if first_two else "2020-06-30"

    conn = get_conn()
    _do_prefetch(conn, fwd_start=fwd_start, fwd_end=fwd_end)
    fcf_annual = _fetch_all_fcf_from_db(conn)

    bear_module = _build_bear_module()

    for ym in first_two:
        calc_date = f"{ym}-01"
        print(f"\n{'='*80}")
        print(f"  Bear 구간: {ym}  (calc_date={calc_date})")
        print(f"{'='*80}")

        df = _score_with_breakdown(conn, calc_date, bear_module, fcf_annual)
        if df.empty:
            print("  데이터 없음")
            continue

        _print_breakdown(calc_date, df, bear_module)

    conn.close()


if __name__ == "__main__":
    main()
