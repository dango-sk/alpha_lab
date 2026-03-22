"""
Step 5b: PostgreSQL 분기 재무 데이터로 Rolling TTM 계산 → fnspace_finance에 저장

- 각 분기 종료 시점 기준 직전 4개 분기 합산 (Rolling TTM)
- TTM_1Q / TTM_2Q / TTM_3Q / TTM_4Q 로 저장
- Flow 항목(매출, 이익 등): 4분기 합산
- Stock 항목(자산, 부채 등): 마지막 분기 값
- EV = 분기 종료일 기준 시가총액 + net_debt
- 비율 항목(ROE, ROIC 등): TTM 기준 재계산
"""
import sys
import os
import calendar
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

sys.path.append(str(Path(__file__).parent.parent))

from lib.db import get_conn

FLOW_COLS = ["ebit", "ebitda", "eps", "revenue", "operating_income", "net_income"]
STOCK_COLS = ["bps", "net_debt", "interest_debt", "total_equity", "ic"]
EXTRA_COLS = ["div_yield", "pcf"]

ALL_FIN_COLS = [
    "pbr", "roe", "roic", "ev", "ic", "ev_ebit", "ebit", "net_debt",
    "interest_debt", "total_equity", "eps", "bps", "per", "psr",
    "ev_ebitda", "ebitda", "revenue", "operating_income", "net_income",
    "oi_margin", "div_yield", "pcf",
]

QTR_NUM = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4}
QTR_LABEL = {1: "TTM_1Q", 2: "TTM_2Q", 3: "TTM_3Q", 4: "TTM_4Q"}
# 분기 종료월
QTR_END_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}
BASE_YEAR = 2017


def qtr_index(year: int, qtr_num: int) -> int:
    return (year - BASE_YEAR) * 4 + (qtr_num - 1)


def load_mcap_map(conn) -> dict:
    """분기 종료일 기준 시가총액 맵 {(stock_code, year, qtr_num): market_cap}"""
    print("  시가총액 데이터 로드 중...")
    rows = conn.execute("""
        SELECT stock_code, trade_date, market_cap
        FROM daily_price
        WHERE market_cap IS NOT NULL AND market_cap > 0
    """).fetchall()

    from collections import defaultdict
    by_code = defaultdict(dict)
    for code, td, mc in rows:
        by_code[code][td] = mc

    mcap_map = {}
    for code, date_mc in by_code.items():
        sorted_dates = sorted(date_mc.keys())
        for year in range(2017, 2027):
            for qtr_num, end_month in QTR_END_MONTH.items():
                end_day = calendar.monthrange(year, end_month)[1]
                target = f"{year}-{end_month:02d}-{end_day}"
                candidates = [d for d in sorted_dates if d <= target]
                if candidates:
                    mcap_map[(code, year, qtr_num)] = date_mc[candidates[-1]]
    print(f"  시가총액 맵: {len(mcap_map):,}건")
    return mcap_map


def calc_ttm():
    conn = get_conn()

    print("  분기 재무 데이터 로드 중...")
    rows = conn.execute("""
        SELECT stock_code, fiscal_year, fiscal_quarter,
               ebit, ebitda, eps, revenue, operating_income, net_income,
               bps, net_debt, interest_debt, total_equity, ic,
               div_yield, pcf
        FROM fnspace_finance
        WHERE fiscal_quarter IN ('1Q', '2Q', '3Q', '4Q')
        ORDER BY stock_code, fiscal_year, fiscal_quarter
    """).fetchall()

    if not rows:
        print("분기 데이터 없음 — TTM 계산 스킵")
        conn.close()
        return

    cols = ["stock_code", "fiscal_year", "fiscal_quarter",
            "ebit", "ebitda", "eps", "revenue", "operating_income", "net_income",
            "bps", "net_debt", "interest_debt", "total_equity", "ic",
            "div_yield", "pcf"]
    df = pd.DataFrame(rows, columns=cols)
    df["fiscal_year"] = df["fiscal_year"].apply(lambda v: int(v))
    df["qtr_num"] = df["fiscal_quarter"].map(QTR_NUM)
    df["qtr_idx"] = df.apply(lambda r: qtr_index(r["fiscal_year"], r["qtr_num"]), axis=1)
    df = df.sort_values(["stock_code", "qtr_idx"]).reset_index(drop=True)

    # 시가총액 로드
    mcap_map = load_mcap_map(conn)

    total_saved = 0
    batch = []

    for code, grp in df.groupby("stock_code", sort=False):
        grp = grp.reset_index(drop=True)
        idxs = grp["qtr_idx"].tolist()

        for i in range(3, len(grp)):
            window_idxs = idxs[i - 3: i + 1]
            if window_idxs != list(range(window_idxs[0], window_idxs[0] + 4)):
                continue

            window = grp.iloc[i - 3: i + 1]
            last = grp.iloc[i]
            end_year = int(last["fiscal_year"])
            qtr_num = int(last["qtr_num"])
            ttm_qtr = QTR_LABEL[qtr_num]

            values = {}
            for c in FLOW_COLS:
                s = window[c].dropna()
                values[c] = float(s.sum()) if len(s) > 0 else None
            for c in STOCK_COLS + EXTRA_COLS:
                v = last[c]
                values[c] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

            te = values.get("total_equity")
            ni = values.get("net_income")
            values["roe"] = (ni / te * 100) if (te and te > 0 and ni is not None) else None

            ic = values.get("ic")
            ebit = values.get("ebit")
            values["roic"] = (ebit / ic * 100) if (ic and ic > 0 and ebit is not None) else None

            rev = values.get("revenue")
            oi = values.get("operating_income")
            values["oi_margin"] = (oi / rev * 100) if (rev and rev > 0 and oi is not None) else None

            # EV = 시가총액 + net_debt
            mcap = mcap_map.get((code, end_year, qtr_num))
            nd = values.get("net_debt")
            ev = (mcap + nd) if (mcap is not None and nd is not None) else None
            values["ev"] = ev
            values["ev_ebit"] = (ev / ebit) if (ev and ebit and ebit > 0) else None
            ebitda = values.get("ebitda")
            values["ev_ebitda"] = (ev / ebitda) if (ev and ebitda and ebitda > 0) else None

            # pbr, per, psr은 주가 필요 → factor_engine에서 재계산
            for c in ["pbr", "per", "psr"]:
                values[c] = None

            batch.append((code, end_year, ttm_qtr, values))

    # PG 저장
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pg_rows = []
    for code, year, ttm_qtr, values in batch:
        row = [code, year, ttm_qtr] + [values.get(c) for c in ALL_FIN_COLS] + [now_str]
        pg_rows.append(tuple(row))
        total_saved += 1

    if pg_rows:
        col_list = ", ".join(ALL_FIN_COLS)
        set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in ALL_FIN_COLS)
        raw_conn = conn._conn if hasattr(conn, '_conn') else conn
        cur = raw_conn.cursor()
        execute_values(cur, f"""
            INSERT INTO alpha_lab.fnspace_finance (
                stock_code, fiscal_year, fiscal_quarter,
                {col_list}, updated_at
            ) VALUES %s
            ON CONFLICT (stock_code, fiscal_year, fiscal_quarter) DO UPDATE SET
                {set_clause}, updated_at=EXCLUDED.updated_at
        """, pg_rows)
        cur.close()
        conn.commit()

    conn.close()
    print(f"\n=== TTM 계산 완료: {total_saved:,}건 저장 (TTM_1Q~4Q) ===")


def main():
    print(f"=== Step 5b: Rolling TTM 계산 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    calc_ttm()


if __name__ == "__main__":
    main()
