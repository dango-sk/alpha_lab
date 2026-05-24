"""
analysis/verify_indicators_db.py

alpha_lab.stock_indicators 테이블 기반 indicator (FE_USE_INDICATORS_DB=1)
결과가 기존 함수와 정확히 동일한지 검증.

3단계:
1. 함수 단위 — _calc_ma_reversion / _calc_mfi 결과 동일성
2. score_stocks 단위 — 임시 전략으로 종목 선정 동일성
3. (--full) 전체 백테스트 결과 동일성

사전 조건:
  - alpha_lab.stock_indicators 테이블 적재 완료
    python scripts/build_stock_indicators.py
  - 적재 안 됐으면 1, 2단계에서 에러

사용:
  python analysis/verify_indicators_db.py
  python analysis/verify_indicators_db.py --full
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
import numpy as np


TEST_STRATEGY_CODE = '''"""검증용: MA/MFI 100% 비중 (OBV 제외, DB 적재 대상 아님)."""

SCORING_MODE = {"large": "quartile"}

WEIGHTS_LARGE = {
    "PRICE_MA_REV": 0.70,
    "MFI": 0.30,
}
WEIGHTS_SMALL = {}

REGRESSION_MODELS = []
OUTLIER_FILTERS = {}

SCORE_MAP = {
    "PRICE_MA_REV": "price_ma_rev_score",
    "MFI": "mfi_score",
}

SCORING_RULES = {
    "price_ma_rev": "rule2",
    "mfi": "rule2",
}

PARAMS = {
    "top_n": 30,
    "tx_cost_bp": 30,
    "weight_cap_pct": 10,
    "stop_loss_enabled": False,
}

QUALITY_FILTER = {
    "exclude_spac_etf_reit": True,
    "require_positive_oi": True,
    "require_positive_roe": True,
    "min_avg_volume": 500_000_000,
}
'''

TEST_DATES = [
    "2018-04-02",
    "2020-04-01",
    "2022-04-01",
    "2024-04-01",
    "2026-05-15",
]


def _check_table_exists(conn) -> bool:
    raw = conn._conn.cursor()
    raw.execute("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema='alpha_lab' AND table_name='stock_indicators'
    """)
    if raw.fetchone()[0] == 0:
        return False
    raw.execute("SELECT COUNT(*) FROM alpha_lab.stock_indicators")
    rows = raw.fetchone()[0]
    print(f"  alpha_lab.stock_indicators: {rows:,} rows")
    return rows > 0


def _load_price_cache(conn):
    from lib.factor_engine import prefetch_all_data, _prefetch_cache
    if "price" not in _prefetch_cache:
        print("[setup] prefetch_all_data 호출 중...")
        prefetch_all_data(conn)
    return _prefetch_cache["price"]


def step1_function_unit():
    """1단계: ma_reversion / mfi DB 버전 vs 기존 결과 동일성."""
    from lib.db import get_conn
    from lib.factor_engine import (
        _calc_ma_reversion, _calc_ma_reversion_from_db,
        _calc_mfi, _calc_mfi_from_db,
    )

    conn = get_conn()

    print("\n" + "=" * 60)
    print("0단계: alpha_lab.stock_indicators 테이블 확인")
    print("=" * 60)
    if not _check_table_exists(conn):
        print("\n  ❌ stock_indicators 테이블이 없거나 비어있음.")
        print("  먼저 다음 명령 실행:")
        print("    python scripts/build_stock_indicators.py")
        return False

    price = _load_price_cache(conn)

    print("\n" + "=" * 60)
    print("1단계: 함수 단위 결과 동일성 검증")
    print("=" * 60 + "\n")

    all_ok = True
    for calc_date in TEST_DATES:
        try:
            old = _calc_ma_reversion(price, calc_date).sort_values("stock_code").reset_index(drop=True)
            new = _calc_ma_reversion_from_db(conn, calc_date).sort_values("stock_code").reset_index(drop=True)
            ma_ok = _compare_df(old, new, ["price_ma_rev", "below_ma"])
            print(f"  {calc_date} ma_reversion : {'✅' if ma_ok else '❌'} (rows old={len(old)} new={len(new)})")
            if not ma_ok:
                all_ok = False
                _diff_report(old, new, calc_date, "ma_reversion")

            old = _calc_mfi(price, calc_date).sort_values("stock_code").reset_index(drop=True)
            new = _calc_mfi_from_db(conn, calc_date).sort_values("stock_code").reset_index(drop=True)
            mfi_ok = _compare_df(old, new, ["mfi"])
            print(f"  {calc_date} mfi          : {'✅' if mfi_ok else '❌'} (rows old={len(old)} new={len(new)})")
            if not mfi_ok:
                all_ok = False
                _diff_report(old, new, calc_date, "mfi")
        except Exception as e:
            print(f"  {calc_date} ❌ 오류: {e}")
            all_ok = False

    conn.close()
    print(f"\n{'✅ 1단계 PASS' if all_ok else '❌ 1단계 FAIL'}")
    return all_ok


def _compare_df(old: pd.DataFrame, new: pd.DataFrame, cols: list, tol: float = 1e-6) -> bool:
    if len(old) != len(new):
        return False
    if not (old["stock_code"].values == new["stock_code"].values).all():
        return False
    for col in cols:
        if col not in old.columns or col not in new.columns:
            return False
        if pd.api.types.is_numeric_dtype(old[col]):
            diff = (old[col].fillna(-99999) - new[col].fillna(-99999)).abs().max()
            if diff > tol:
                return False
        else:
            if not (old[col].fillna("") == new[col].fillna("")).all():
                return False
    return True


def _diff_report(old: pd.DataFrame, new: pd.DataFrame, calc_date: str, name: str):
    print(f"      차이 상세 [{calc_date} {name}]:")
    if len(old) != len(new):
        print(f"        rows: old={len(old)} vs new={len(new)}")
    merged = old.merge(new, on="stock_code", suffixes=("_old", "_new"))
    for col in old.columns:
        if col == "stock_code" or not pd.api.types.is_numeric_dtype(old[col]):
            continue
        a, b = f"{col}_old", f"{col}_new"
        if a in merged.columns and b in merged.columns:
            merged["_diff"] = (merged[a].fillna(0) - merged[b].fillna(0)).abs()
            top = merged.nlargest(5, "_diff")[["stock_code", a, b, "_diff"]]
            if top["_diff"].max() > 1e-6:
                print(f"        {col} 차이 TOP 5:")
                print(top.to_string(index=False))


def step2_score_stocks():
    from lib.db import get_conn
    from lib.factor_engine import score_stocks_from_strategy, code_to_module, clear_factor_cache, clear_indicators_db_cache

    calc_date = "2024-04-01"
    print(f"\n{'=' * 60}")
    print(f"2단계: score_stocks_from_strategy 결과 동일성 ({calc_date})")
    print(f"{'=' * 60}\n")

    strategy = code_to_module(TEST_STRATEGY_CODE)
    conn = get_conn()

    os.environ.pop("FE_USE_INDICATORS_DB", None)
    clear_factor_cache()
    result_off = score_stocks_from_strategy(conn, calc_date, strategy)

    os.environ["FE_USE_INDICATORS_DB"] = "1"
    clear_factor_cache()
    clear_indicators_db_cache()
    result_on = score_stocks_from_strategy(conn, calc_date, strategy)
    conn.close()

    codes_off = [c for c, _ in result_off]
    codes_on = [c for c, _ in result_on]
    scores_off = {c: s for c, s in result_off}
    scores_on = {c: s for c, s in result_on}

    same_codes = codes_off == codes_on
    score_diff_max = max((abs(scores_off[c] - scores_on.get(c, 0)) for c in codes_off if c in scores_on), default=0)

    print(f"  종목 순서 동일: {'✅' if same_codes else '❌'}")
    print(f"  점수 최대 차이: {score_diff_max:.6f}  {'✅' if score_diff_max < 1e-6 else '❌'}")
    if not same_codes:
        print(f"\n  OFF top10: {codes_off[:10]}")
        print(f"  ON  top10: {codes_on[:10]}")
    ok = same_codes and score_diff_max < 1e-6
    print(f"\n{'✅ 2단계 PASS' if ok else '❌ 2단계 FAIL'}")
    return ok


def step3_full_backtest():
    from lib.data import run_strategy_backtest
    from lib.factor_engine import clear_factor_cache, clear_prefetch_cache, clear_indicators_db_cache

    print(f"\n{'=' * 60}")
    print("3단계: 전체 백테스트 OFF/ON 결과 비교 (~10분)")
    print(f"{'=' * 60}\n")

    print("  OFF 백테스트 시작...")
    os.environ.pop("FE_USE_INDICATORS_DB", None)
    clear_prefetch_cache()
    clear_factor_cache()
    r_off = run_strategy_backtest(strategy_code=TEST_STRATEGY_CODE, universe="KOSPI", rebal_type="monthly")
    if not r_off or "error" in r_off:
        print(f"  ❌ OFF 실패: {r_off}")
        return False

    print("  ON 백테스트 시작...")
    os.environ["FE_USE_INDICATORS_DB"] = "1"
    clear_prefetch_cache()
    clear_factor_cache()
    clear_indicators_db_cache()
    r_on = run_strategy_backtest(strategy_code=TEST_STRATEGY_CODE, universe="KOSPI", rebal_type="monthly")
    if not r_on or "error" in r_on:
        print(f"  ❌ ON 실패: {r_on}")
        return False

    custom_off = r_off.get("CUSTOM", r_off)
    custom_on = r_on.get("CUSTOM", r_on)
    keys = ["total_return", "cagr", "mdd", "sharpe", "monthly_std", "avg_turnover"]
    all_ok = True
    for k in keys:
        v_off = custom_off.get(k)
        v_on = custom_on.get(k)
        if v_off is None or v_on is None:
            print(f"  {k:14s}: ⚠️  missing (off={v_off}, on={v_on})")
            continue
        diff = abs(v_off - v_on)
        ok = diff < 1e-6
        all_ok = all_ok and ok
        print(f"  {k:14s}: off={v_off:.6f}  on={v_on:.6f}  diff={diff:.2e}  {'✅' if ok else '❌'}")

    print(f"\n{'✅ 3단계 PASS' if all_ok else '❌ 3단계 FAIL'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="3단계 전체 백테스트 (~10분)")
    args = parser.parse_args()

    print("=" * 60)
    print("Indicators DB (alpha_lab.stock_indicators) 결과 동일성 검증")
    print("=" * 60)

    ok1 = step1_function_unit()
    if not ok1:
        print("\n  → 1단계 실패. 코드 수정 또는 DB 적재 다시 필요.")
        sys.exit(1)

    ok2 = step2_score_stocks()
    ok3 = step3_full_backtest() if args.full else None

    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"  1단계 함수 단위:    {'✅ PASS' if ok1 else '❌ FAIL'}")
    print(f"  2단계 score_stocks: {'✅ PASS' if ok2 else '❌ FAIL'}")
    if ok3 is None:
        print(f"  3단계 전체 백테스트: ⏭  스킵 (--full 옵션으로 실행)")
    else:
        print(f"  3단계 전체 백테스트: {'✅ PASS' if ok3 else '❌ FAIL'}")

    all_ok = ok1 and ok2 and (ok3 if ok3 is not None else True)
    if all_ok:
        print("\n  ✅ 검증 통과 — production 적용 안전")
        print("     Railway 환경변수 추가: FE_USE_INDICATORS_DB=1")
    else:
        print("\n  ❌ 일부 검증 실패")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
