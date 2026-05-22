"""
analysis/verify_forward_fix.py

forward fix(PR #13)가 실제로 적용되어 fwd_* 컬럼이 join되고
F_PER / F_PBR / F_EVEBITDA / F_SPSG / F_EPS_M raw 값이 채워지는지
**단일 리밸 날짜**에 대해 직접 검증한다 (백테스트 안 돌림, ~30초).

실행:
  python analysis/verify_forward_fix.py                # 가장 최근 리밸일
  python analysis/verify_forward_fix.py --date 2026-05-15
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.db import get_conn
from lib.factor_engine import load_factor_data, clear_factor_cache


def _pct_non_null(series) -> str:
    n = len(series)
    if n == 0:
        return "0/0 (-)"
    nn = series.notna().sum()
    return f"{nn}/{n} ({nn/n*100:.1f}%)"


def _pct_positive(series) -> str:
    n = len(series)
    if n == 0:
        return "0/0 (-)"
    pos = ((series.notna()) & (series > 0)).sum()
    return f"{pos}/{n} ({pos/n*100:.1f}%)"


def get_latest_rebal_date(conn) -> str:
    """가장 최근 KOSPI/monthly 리밸 날짜."""
    cur = conn._conn.cursor() if hasattr(conn, "_conn") else conn.cursor()
    cur.execute("""
        SELECT MAX(trade_date) FROM alpha_lab.universe
        WHERE rebal_type='monthly'
    """)
    return cur.fetchone()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="검증할 리밸 날짜 (YYYY-MM-DD)")
    args = parser.parse_args()

    conn = get_conn()
    calc_date = args.date or get_latest_rebal_date(conn)
    print(f"\n검증 날짜: {calc_date}")
    print("=" * 60)

    clear_factor_cache()
    df = load_factor_data(conn, calc_date)
    if df is None or df.empty:
        print("  load_factor_data 결과 비어있음")
        return

    print(f"  load_factor_data 종목수: {len(df)}")
    print()

    # 1) fwd_* 원본 컬럼 채움률 (join이 성공했는지)
    print("── 1단계: fnspace_forward join 성공 여부 (fwd_* 원본 컬럼) ──")
    for col in ["fwd_eps", "fwd_bps", "fwd_ebitda", "fwd_ebit", "fwd_revenue", "fwd_eps_3m"]:
        if col in df.columns:
            print(f"  {col:15s}: notna {_pct_non_null(df[col])}")
        else:
            print(f"  {col:15s}: 컬럼 없음 (계산 안 됨)")

    # 2) F_* 파생 컬럼 (실제 점수 계산에 들어가는 값)
    print()
    print("── 2단계: F_* 파생 팩터 raw 값 (점수 계산에 들어가는 값) ──")
    for col in ["f_per", "f_pbr", "f_ev_ebitda", "f_ev_ebit", "f_epsg", "f_ebitg", "f_spsg", "f_eps_m"]:
        if col in df.columns:
            ser = df[col]
            pos = _pct_positive(ser) if col in ("f_per", "f_pbr", "f_ev_ebitda", "f_ev_ebit") else _pct_non_null(ser)
            label = ">0" if col in ("f_per", "f_pbr", "f_ev_ebitda", "f_ev_ebit") else "notna"
            mean = ser[ser.notna()].mean()
            print(f"  {col:15s}: {label} {pos}   mean={mean:.3f}" if mean == mean else f"  {col:15s}: {label} {pos}")
        else:
            print(f"  {col:15s}: 컬럼 없음")

    # 3) 회귀 입력 컬럼 검증 (ATT_PER, ATT_EVEBIT)
    print()
    print("── 3단계: 회귀 입력 검증 (ATT_PER, ATT_EVEBIT) ──")
    for x_col, y_col, name in [("f_epsg", "f_per", "fper_epsg(ATT_PER)"),
                                ("f_ebitg", "f_ev_ebit", "fevebit_ebitg(ATT_EVEBIT)")]:
        if x_col in df.columns and y_col in df.columns:
            valid = ((df[x_col].notna()) & (df[y_col].notna()) &
                     (df[x_col] > 0) & (df[y_col] > 0))
            n_valid = valid.sum()
            status = "회귀 가능 ✓" if n_valid >= 20 else f"회귀 불가능 (valid<20) ✗"
            print(f"  {name:30s}: valid={n_valid}/{len(df)}  → {status}")
        else:
            print(f"  {name:30s}: 컬럼 없음")

    # 4) 샘플 종목 (POSCO홀딩스, 삼성전자 등) raw 값 출력
    print()
    print("── 4단계: 샘플 종목 raw 값 ──")
    samples = ["A005490", "A005930", "A000660", "A035420"]  # POSCO, 삼성전자, SK하이닉스, NAVER
    cols_show = ["stock_code", "close", "eps", "fwd_eps", "t_per", "f_per", "bps", "fwd_bps", "pbr", "f_pbr"]
    cols_avail = [c for c in cols_show if c in df.columns]
    sub = df[df["stock_code"].isin(samples)][cols_avail]
    if sub.empty:
        print("  샘플 종목 없음 (universe 다를 수 있음)")
    else:
        print(sub.to_string(index=False))

    print()
    print("=" * 60)
    # 최종 판정
    has_fwd_eps = "fwd_eps" in df.columns and df["fwd_eps"].notna().sum() > 0
    has_f_per = "f_per" in df.columns and (df["f_per"] > 0).sum() > 0
    if has_fwd_eps and has_f_per:
        print("  ✓ forward fix 적용됨: fwd_* join 성공 + F_* raw 값 정상 계산")
    elif has_fwd_eps and not has_f_per:
        print("  ⚠ fwd_* join은 성공했으나 F_* 파생 계산 어딘가 막힘 — 추가 조사 필요")
    else:
        print("  ✗ fwd_* 컬럼이 비어있음 → forward fix 미적용 또는 데이터 없음")
    print("=" * 60)


if __name__ == "__main__":
    main()
