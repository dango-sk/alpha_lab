"""
Step 5b: 분기 재무 데이터로 TTM 계산 → fnspace_finance에 저장
실행: python scripts/step5b_calc_ttm.py

- fnspace_finance의 1Q~4Q 데이터를 읽어 연도별 TTM 계산
- fiscal_quarter='TTM'으로 fnspace_finance에 저장
- 4분기 모두 있는 종목만 TTM 생성 (없으면 Annual fallback은 factor_engine에서 처리)
- Flow 항목(매출, 이익 등): 4분기 합산
- Stock 항목(자산, 부채 등): 4Q 값 사용
- 비율 항목(ROE, ROIC 등): TTM 기준 재계산

소요 시간: 수초
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

LAB_DB = Path(__file__).parent.parent / "data" / "alpha_lab.db"

# Flow 항목: 4분기 합산
FLOW_COLS = ["ebit", "ebitda", "eps", "revenue", "operating_income", "net_income"]
# Stock 항목: 4Q 값 사용
STOCK_COLS = ["bps", "net_debt", "interest_debt", "total_equity", "ic"]
# FnSpace에서 오는 비율/밸류 항목 (TTM에서는 무시, 재계산)
RATIO_COLS = ["pbr", "roe", "roic", "ev", "ev_ebit", "per", "psr", "ev_ebitda",
              "oi_margin", "div_yield", "pcf"]

ALL_FIN_COLS = [
    "pbr", "roe", "roic", "ev", "ic", "ev_ebit", "ebit", "net_debt",
    "interest_debt", "total_equity", "eps", "bps", "per", "psr",
    "ev_ebitda", "ebitda", "revenue", "operating_income", "net_income",
    "oi_margin", "div_yield", "pcf",
]


def calc_ttm():
    conn = sqlite3.connect(str(LAB_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    # 분기 데이터 로드
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

    import pandas as pd
    cols = ["stock_code", "fiscal_year", "fiscal_quarter",
            "ebit", "ebitda", "eps", "revenue", "operating_income", "net_income",
            "bps", "net_debt", "interest_debt", "total_equity", "ic",
            "div_yield", "pcf"]
    df = pd.DataFrame(rows, columns=cols)

    total_saved = 0

    for year in sorted(df["fiscal_year"].unique()):
        yr = df[df["fiscal_year"] == year]
        # 4분기 모두 있는 종목만
        qtr_counts = yr.groupby("stock_code")["fiscal_quarter"].nunique()
        full_codes = qtr_counts[qtr_counts == 4].index
        if len(full_codes) == 0:
            continue

        yr_full = yr[yr["stock_code"].isin(full_codes)]

        # Flow: 합산
        flow_agg = yr_full.groupby("stock_code")[FLOW_COLS].sum()
        # Stock: 4Q 값
        q4 = yr_full[yr_full["fiscal_quarter"] == "4Q"].set_index("stock_code")
        stock_vals = q4[STOCK_COLS]
        # div_yield, pcf: 4Q 값 사용 (연간 기준)
        extra_vals = q4[["div_yield", "pcf"]]

        ttm = flow_agg.join(stock_vals).join(extra_vals).reset_index()

        # 비율 재계산
        ttm["roe"] = np.where(
            ttm["total_equity"].notna() & (ttm["total_equity"] > 0),
            ttm["net_income"] / ttm["total_equity"] * 100, None)
        ttm["roic"] = np.where(
            ttm["ic"].notna() & (ttm["ic"] > 0),
            ttm["ebit"] / ttm["ic"] * 100, None)
        ttm["oi_margin"] = np.where(
            ttm["revenue"].notna() & (ttm["revenue"] > 0),
            ttm["operating_income"] / ttm["revenue"] * 100, None)

        # 주가 기반 비율은 None (factor_engine에서 현재 주가로 재계산)
        ttm["pbr"] = None
        ttm["ev"] = None
        ttm["ev_ebit"] = None
        ttm["per"] = None
        ttm["psr"] = None
        ttm["ev_ebitda"] = None

        # DB에 저장
        for _, row in ttm.iterrows():
            values = {c: (None if (isinstance(row.get(c), float) and np.isnan(row.get(c))) else row.get(c))
                      for c in ALL_FIN_COLS if c in row.index}
            valid_cols = [c for c, v in values.items() if v is not None]
            if not valid_cols:
                continue

            set_clause = ", ".join(f"{c} = ?" for c in valid_cols)
            vals = [values[c] for c in valid_cols]
            conn.execute(f"""
                INSERT INTO fnspace_finance (stock_code, fiscal_year, fiscal_quarter, {', '.join(valid_cols)}, updated_at)
                VALUES (?, ?, 'TTM', {', '.join('?' for _ in valid_cols)}, datetime('now','localtime'))
                ON CONFLICT(stock_code, fiscal_year, fiscal_quarter) DO UPDATE SET
                {set_clause}, updated_at = datetime('now','localtime')
            """, (row["stock_code"], year, *vals, *vals))
            total_saved += 1

        print(f"  {year}년: {len(full_codes)}종목 TTM 저장")

    conn.commit()
    conn.close()
    print(f"\n=== TTM 계산 완료: {total_saved}건 저장 ===")


def upload_ttm_to_pg():
    """TTM 데이터를 Railway PostgreSQL에 업로드"""
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

    rows = conn.execute("""
        SELECT stock_code, fiscal_year, fiscal_quarter,
               pbr, roe, roic, ev, ic, ev_ebit, ebit, net_debt,
               interest_debt, total_equity, eps, bps, per, psr,
               ev_ebitda, ebitda, revenue, operating_income, net_income,
               oi_margin, div_yield, pcf, updated_at
        FROM fnspace_finance
        WHERE fiscal_quarter = 'TTM'
    """).fetchall()

    if rows:
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

    print(f"PostgreSQL TTM 업로드: {len(rows):,}건")
    pg.close()
    conn.close()


def main():
    from datetime import datetime
    print(f"=== Step 5b: TTM 계산 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    calc_ttm()
    upload_ttm_to_pg()


if __name__ == "__main__":
    main()
