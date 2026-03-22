"""
Step 5b: 분기 재무 데이터로 Rolling TTM 계산 → fnspace_finance에 저장
실행: python scripts/step5b_calc_ttm.py

- 각 분기 종료 시점 기준 직전 4개 분기 합산 (Rolling TTM)
- TTM_1Q / TTM_2Q / TTM_3Q / TTM_4Q 로 저장
  예) 2025년 TTM_3Q = 2024 4Q + 2025 1Q + 2Q + 3Q
- Flow 항목(매출, 이익 등): 4분기 합산
- Stock 항목(자산, 부채 등): 마지막 분기 값
- 비율 항목(ROE, ROIC 등): TTM 기준 재계산

소요 시간: 수십 초
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

LAB_DB = Path(__file__).parent.parent / "data" / "alpha_lab.db"

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
BASE_YEAR = 2017


def qtr_index(year: int, qtr_num: int) -> int:
    return (year - BASE_YEAR) * 4 + (qtr_num - 1)


def calc_ttm():
    conn = sqlite3.connect(str(LAB_DB))
    conn.execute("PRAGMA journal_mode=WAL")

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
    df["fiscal_year"] = df["fiscal_year"].apply(
        lambda v: int.from_bytes(v, "little") if isinstance(v, bytes) else int(v)
    )
    df["qtr_num"] = df["fiscal_quarter"].map(QTR_NUM)
    df["qtr_idx"] = df.apply(lambda r: qtr_index(r["fiscal_year"], r["qtr_num"]), axis=1)
    df = df.sort_values(["stock_code", "qtr_idx"]).reset_index(drop=True)

    total_saved = 0
    batch = []

    for code, grp in df.groupby("stock_code", sort=False):
        grp = grp.reset_index(drop=True)
        idxs = grp["qtr_idx"].tolist()

        for i in range(3, len(grp)):
            window_idxs = idxs[i - 3: i + 1]
            # 4개 분기가 연속인지 확인
            if window_idxs != list(range(window_idxs[0], window_idxs[0] + 4)):
                continue

            window = grp.iloc[i - 3: i + 1]
            last = grp.iloc[i]
            end_year = int(last["fiscal_year"])
            ttm_qtr = QTR_LABEL[int(last["qtr_num"])]

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

            for c in ["pbr", "ev", "ev_ebit", "per", "psr", "ev_ebitda"]:
                values[c] = None

            batch.append((code, end_year, ttm_qtr, values))

    # DB 저장
    for code, year, ttm_qtr, values in batch:
        valid_cols = [c for c in ALL_FIN_COLS if values.get(c) is not None]
        if not valid_cols:
            continue
        set_clause = ", ".join(f"{c} = ?" for c in valid_cols)
        vals = [values[c] for c in valid_cols]
        conn.execute(f"""
            INSERT INTO fnspace_finance (stock_code, fiscal_year, fiscal_quarter, {', '.join(valid_cols)}, updated_at)
            VALUES (?, ?, ?, {', '.join('?' for _ in valid_cols)}, datetime('now','localtime'))
            ON CONFLICT(stock_code, fiscal_year, fiscal_quarter) DO UPDATE SET
            {set_clause}, updated_at = datetime('now','localtime')
        """, (code, year, ttm_qtr, *vals, *vals))
        total_saved += 1

    conn.commit()
    conn.close()
    print(f"\n=== TTM 계산 완료: {total_saved}건 저장 (TTM_1Q~4Q) ===")


def upload_ttm_to_pg():
    """TTM_1Q~4Q 데이터를 Railway PostgreSQL에 업로드"""
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        print("[SKIP] psycopg2 없음, PostgreSQL 업로드 스킵")
        return

    import os
    for env_path in [
        Path(__file__).parent.parent / ".env",
        Path.home() / "Downloads" / "alpha_radar" / ".env",
    ]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"'))

    PG_URL = os.environ.get("PG_URL", "postgresql://postgres:NgHDMsgiGwbvMpWHLUgTQguaedbGoxvv@metro.proxy.rlwy.net:50087/railway")

    conn = sqlite3.connect(str(LAB_DB))
    pg = psycopg2.connect(PG_URL)
    cur = pg.cursor()

    import struct
    def to_int(v):
        if isinstance(v, bytes):
            return struct.unpack('<q', v)[0]
        return int(v)

    rows = conn.execute("""
        SELECT stock_code, fiscal_year, fiscal_quarter,
               pbr, roe, roic, ev, ic, ev_ebit, ebit, net_debt,
               interest_debt, total_equity, eps, bps, per, psr,
               ev_ebitda, ebitda, revenue, operating_income, net_income,
               oi_margin, div_yield, pcf, updated_at
        FROM fnspace_finance
        WHERE fiscal_quarter IN ('TTM_1Q', 'TTM_2Q', 'TTM_3Q', 'TTM_4Q')
    """).fetchall()

    if rows:
        rows = [(r[0], to_int(r[1]), r[2], *r[3:]) for r in rows]
        execute_values(cur, """
            INSERT INTO alpha_lab.fnspace_finance (
                stock_code, fiscal_year, fiscal_quarter,
                pbr, roe, roic, ev, ic, ev_ebit, ebit, net_debt,
                interest_debt, total_equity, eps, bps, per, psr,
                ev_ebitda, ebitda, revenue, operating_income, net_income,
                oi_margin, div_yield, pcf, updated_at
            ) VALUES %s
            ON CONFLICT (stock_code, fiscal_year, fiscal_quarter) DO UPDATE SET
                pbr=EXCLUDED.pbr, roe=EXCLUDED.roe, roic=EXCLUDED.roic,
                ev=EXCLUDED.ev, ic=EXCLUDED.ic, ev_ebit=EXCLUDED.ev_ebit,
                ebit=EXCLUDED.ebit, net_debt=EXCLUDED.net_debt,
                interest_debt=EXCLUDED.interest_debt, total_equity=EXCLUDED.total_equity,
                eps=EXCLUDED.eps, bps=EXCLUDED.bps, per=EXCLUDED.per, psr=EXCLUDED.psr,
                ev_ebitda=EXCLUDED.ev_ebitda, ebitda=EXCLUDED.ebitda,
                revenue=EXCLUDED.revenue, operating_income=EXCLUDED.operating_income,
                net_income=EXCLUDED.net_income, oi_margin=EXCLUDED.oi_margin,
                div_yield=EXCLUDED.div_yield, pcf=EXCLUDED.pcf,
                updated_at=EXCLUDED.updated_at
        """, rows)
        pg.commit()

    print(f"PostgreSQL TTM 업로드: {len(rows):,}건 (TTM_1Q~4Q)")
    pg.close()
    conn.close()


def main():
    from datetime import datetime
    print(f"=== Step 5b: Rolling TTM 계산 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    calc_ttm()
    upload_ttm_to_pg()


if __name__ == "__main__":
    main()
