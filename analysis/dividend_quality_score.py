"""
배당-퀄리티-리스크 스코어링 결과 출력

실행: python analysis/dividend_quality_score.py [YYYY-MM-DD]
기준일 생략 시 오늘 날짜 사용
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.factor_engine import (
    load_factor_data, apply_quality_filter,
    apply_dividend_quality_score, prefetch_all_data,
)
from lib.db import get_conn

QUALITY_FILTER = {
    "exclude_spac_etf_reit": True,
    "require_positive_oi": True,
    "require_positive_roe": True,
    "min_avg_volume": 500_000_000,
}
MIN_MARKET_CAP = 500_000_000_000


def main():
    calc_date = sys.argv[1] if len(sys.argv) > 1 else str(date.today())
    print(f"\n기준일: {calc_date}")

    conn = get_conn()
    prefetch_all_data(conn)

    df = load_factor_data(conn, calc_date)
    conn.close()

    if df is None or df.empty:
        print("데이터 없음")
        return

    # 기존 필터
    df_large = df[df["size_group"] == "large"].copy()
    df_large = df_large[df_large["market_cap"] >= MIN_MARKET_CAP].copy()
    df_large = apply_quality_filter(df_large, QUALITY_FILTER)

    # 배당-퀄리티 스코어링
    df_large = apply_dividend_quality_score(df_large)

    passed = df_large[df_large["quality_pass"] == 1].copy()
    score2 = passed[passed["div_quality_score"] == 2]
    score1 = passed[passed["div_quality_score"] == 1]
    score0 = passed[passed["div_quality_score"] == 0]

    print(f"\n전체 대형주(시총 5천억↑): {len(df_large)}개")
    print(f"퀄리티 필터 통과:          {len(passed)}개")
    print(f"  → 2점 (배당OK + Beta≤1.2 + GPA상위): {len(score2)}개")
    print(f"  → 1점 (배당OK + Beta≤1.2 + GPA하위): {len(score1)}개")
    print(f"  → 0점 (배당미달 또는 Junk):           {len(score0)}개")

    # 유니버스 배당수익률 중앙값
    div_med = passed["div_yield"].median()
    print(f"\n유니버스 배당수익률 중앙값: {div_med:.2%}" if div_med else "")

    for score_val, label, subset in [
        (2, "★★ 2점 종목", score2),
        (1, "★  1점 종목", score1),
    ]:
        if subset.empty:
            continue
        print(f"\n{'─'*60}")
        print(f"[{label}]  배당>중앙값, Beta≤1.2, GPA {'상위' if score_val==2 else '하위'}50%")
        print(f"{'#':>4}  {'종목코드':<10}  {'종목명':<16}  {'배당수익률':>8}  {'Beta':>6}  {'GPA':>6}")
        print(f"{'─'*60}")
        subset_sorted = subset.sort_values("gpa", ascending=False)
        for i, (_, row) in enumerate(subset_sorted.iterrows(), 1):
            div = f"{row['div_yield']:.2%}" if hasattr(row['div_yield'], '__float__') and row['div_yield'] == row['div_yield'] else "N/A"
            beta = f"{row['beta']:.2f}" if row['beta'] == row['beta'] else "N/A"
            gpa = f"{row['gpa']:.3f}" if row['gpa'] == row['gpa'] else "N/A"
            print(f"{i:>4}  {row['stock_code']:<10}  {row['stock_name']:<16}  {div:>8}  {beta:>6}  {gpa:>6}")

    print(f"\n{'─'*60}")
    print(f"[0점 사유별]")
    if score0.empty:
        print("  없음")
    else:
        no_div = (score0["div_yield"].isna() | (score0["div_yield"] <= div_med)).sum()
        junk = (~score0["div_yield"].isna() & (score0["div_yield"] > div_med) & (score0["beta"] > 1.2)).sum()
        no_gpa = (~score0["div_yield"].isna() & (score0["div_yield"] > div_med) & (score0["beta"] <= 1.2) & score0["gpa"].isna()).sum()
        print(f"  배당 미달 (하위50% 이하): {no_div}개")
        print(f"  Junk (Beta > 1.2):      {junk}개")
        print(f"  GPA 데이터 없음:         {no_gpa}개")


if __name__ == "__main__":
    main()
