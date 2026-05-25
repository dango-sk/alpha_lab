"""
analysis/audit_factor_coverage.py

팩터 계산에 쓰이는 모든 컬럼의 coverage (NaN%) 를 fiscal/시간 축 별로 audit.

목적:
  fnspace 4Q BS 누락 같은 데이터 quirk 를 한눈에 노출.
  특정 시점 / 컬럼의 NaN 비율이 다른 시점 대비 비정상적으로 높으면 회귀 의심.

검사 범위:
  1. fnspace_finance     — 22+ 컬럼 (TTM/Annual 별, fiscal_year × fiscal_quarter)
  2. fnspace_forward     — 10+ 컬럼 (year-month 별)
  3. stock_indicators    — ma_120, mfi_val 등 (year-month 별)
  4. daily_price         — close, market_cap (year-month 별)

실행:
  python analysis/audit_factor_coverage.py                       # 전체 출력
  python analysis/audit_factor_coverage.py --since 2024          # 2024년 이후만
  python analysis/audit_factor_coverage.py --csv out_dir/        # CSV 저장
  python analysis/audit_factor_coverage.py --section finance     # 특정 섹션만
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
from lib.db import get_conn


# 팩터 사용 컬럼 정의 (factor_engine.py 의 SELECT 와 동기화)
FINANCE_COLS = [
    # FLOW
    "ebit", "ebitda", "eps", "revenue", "operating_income", "net_income",
    # STOCK (BS)
    "bps", "net_debt", "interest_debt", "total_equity", "ic",
    # EXTRA
    "div_yield", "pcf",
    # Derived
    "pbr", "per", "psr", "ev_ebitda", "ev_ebit", "ev", "roe", "roic", "oi_margin",
]

FORWARD_COLS = [
    "fwd_eps", "fwd_per", "fwd_ebit", "fwd_ebitda", "fwd_ev_ebitda",
    "fwd_revenue", "fwd_oi", "fwd_ni", "fwd_roe", "fwd_bps",
]

INDICATOR_COLS = ["ma_120", "deviation_120", "mfi_val", "pos_sum_14", "neg_sum_14"]

PRICE_COLS = ["close", "market_cap"]


def nan_pct_pivot(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    """group_cols 별로 각 value_col 의 NaN% 와 total count 를 계산."""
    out = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = {gc: (keys if not isinstance(keys, tuple) else keys[i])
               for i, gc in enumerate(group_cols if isinstance(keys, tuple) else [group_cols[0]])}
        # tuple unpack 안전 처리
        if isinstance(keys, tuple):
            for i, gc in enumerate(group_cols):
                row[gc] = keys[i]
        else:
            row[group_cols[0]] = keys
        row["n"] = len(g)
        for col in value_cols:
            if col not in g.columns:
                row[col] = "—"
            else:
                nan_n = g[col].isna().sum()
                row[col] = f"{nan_n / len(g) * 100:.0f}%" if len(g) > 0 else "—"
        out.append(row)
    result = pd.DataFrame(out).sort_values(group_cols).reset_index(drop=True)
    return result


def audit_finance(conn, since_year: int | None = None) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("  [1/4] fnspace_finance — TTM/Annual coverage by (fiscal_year, fiscal_quarter)")
    print("=" * 80)

    cols_select = ", ".join(FINANCE_COLS)
    where = f"WHERE fiscal_year >= {since_year}" if since_year else ""
    sql = f"""
        SELECT stock_code, fiscal_year, fiscal_quarter, {cols_select}
        FROM fnspace_finance
        {where}
        ORDER BY fiscal_year DESC, fiscal_quarter
    """
    df = pd.read_sql(sql, conn._conn)
    if len(df) == 0:
        print("  (데이터 없음)")
        return pd.DataFrame()

    pivot = nan_pct_pivot(df, ["fiscal_year", "fiscal_quarter"], FINANCE_COLS)
    print(pivot.to_string(index=False))
    return pivot


def audit_forward(conn, since_year: int | None = None) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("  [2/4] fnspace_forward — consensus coverage by year-month")
    print("=" * 80)

    cols_select = ", ".join(FORWARD_COLS)
    where = f"WHERE trade_date >= '{since_year}-01-01'" if since_year else ""
    sql = f"""
        SELECT stock_code, trade_date, {cols_select}
        FROM fnspace_forward
        {where}
        ORDER BY trade_date DESC
    """
    df = pd.read_sql(sql, conn._conn)
    if len(df) == 0:
        print("  (데이터 없음)")
        return pd.DataFrame()

    df["ym"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m")
    pivot = nan_pct_pivot(df, ["ym"], FORWARD_COLS)
    print(pivot.to_string(index=False))
    return pivot


def audit_indicators(conn, since_year: int | None = None) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("  [3/4] stock_indicators — MA/MFI coverage by year-month")
    print("=" * 80)

    cols_select = ", ".join(INDICATOR_COLS)
    where = f"WHERE trade_date >= '{since_year}-01-01'" if since_year else ""
    sql = f"""
        SELECT stock_code, trade_date, {cols_select}
        FROM alpha_lab.stock_indicators
        {where}
    """
    try:
        df = pd.read_sql(sql, conn._conn)
    except Exception as e:
        print(f"  (조회 실패: {e})")
        return pd.DataFrame()
    if len(df) == 0:
        print("  (데이터 없음)")
        return pd.DataFrame()

    df["ym"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m")
    pivot = nan_pct_pivot(df, ["ym"], INDICATOR_COLS)
    print(pivot.to_string(index=False))
    return pivot


def audit_price(conn, since_year: int | None = None) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("  [4/4] daily_price — close/market_cap coverage by year-month")
    print("=" * 80)

    cols_select = ", ".join(PRICE_COLS)
    where = f"WHERE trade_date >= '{since_year}-01-01'" if since_year else ""
    sql = f"""
        SELECT stock_code, trade_date, {cols_select}
        FROM daily_price
        {where}
    """
    df = pd.read_sql(sql, conn._conn)
    if len(df) == 0:
        print("  (데이터 없음)")
        return pd.DataFrame()

    df["ym"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m")
    pivot = nan_pct_pivot(df, ["ym"], PRICE_COLS)
    print(pivot.to_string(index=False))
    return pivot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=2023,
                        help="시작 연도 (기본 2023). 전체는 0 또는 음수.")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV 저장 디렉토리 (예: out/coverage/)")
    parser.add_argument("--section", type=str, default="all",
                        choices=["all", "finance", "forward", "indicators", "price"])
    args = parser.parse_args()

    since = args.since if args.since and args.since > 0 else None
    conn = get_conn()

    results = {}
    if args.section in ("all", "finance"):
        results["finance"] = audit_finance(conn, since)
    if args.section in ("all", "forward"):
        results["forward"] = audit_forward(conn, since)
    if args.section in ("all", "indicators"):
        results["indicators"] = audit_indicators(conn, since)
    if args.section in ("all", "price"):
        results["price"] = audit_price(conn, since)

    if args.csv:
        out_dir = Path(args.csv)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in results.items():
            if df is None or len(df) == 0:
                continue
            path = out_dir / f"coverage_{name}.csv"
            df.to_csv(path, index=False)
            print(f"  → {path}")

    print("\n" + "=" * 80)
    print("  팁:")
    print("  - 같은 컬럼이 특정 (year, quarter) 에서만 NaN% 급등 → fnspace 데이터 quirk")
    print("  - 모든 quarter 에서 일관되게 높은 NaN% → 그 컬럼은 쓰지 말아야 할 dead 컬럼")
    print("  - TTM_4Q 의 BS (bps/net_debt/interest_debt/total_equity/ic) 가 ~70% 면")
    print("    Annual fallback (PR #31) 이 아직 데이터에 반영 안 됨 — step5b 재실행 필요")
    print("=" * 80)


if __name__ == "__main__":
    main()
