"""
analysis/backtest_forward_fix_compare.py

forward fix(PR #13) 적용 효과 확인용 핀포인트 백테스트.
A0(기본 전략)과 수정전략_코스피_cap30%_top30_tx30bp_월간 두 개만 돌려서
지표(CAGR/MDD/Sharpe/total_return) + 최근 리밸 편입 종목을 비교한다.

실행:
  python analysis/backtest_forward_fix_compare.py                # 결과 출력만 (DB 저장 X)
  python analysis/backtest_forward_fix_compare.py --save         # backtest_cache에 결과도 갱신
  python analysis/backtest_forward_fix_compare.py --only 수정    # 수정전략만
  python analysis/backtest_forward_fix_compare.py --only A0      # A0만
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.db import get_conn
from lib.data import run_strategy_backtest, save_strategy
from lib.factor_engine import DEFAULT_STRATEGY_CODE, clear_factor_cache, clear_prefetch_cache


TARGETS = {
    "A0": {
        "code_source": "DEFAULT_STRATEGY_CODE",
        "universe": "KOSPI",
        "rebal_type": "monthly",
    },
    "수정전략_코스피_cap30%_top30_tx30bp_월간": {
        "code_source": "DB",
        "universe": "KOSPI",
        "rebal_type": "monthly",
    },
}


def load_strategy_code(name: str) -> str | None:
    """A0는 DEFAULT_STRATEGY_CODE, 그 외는 backtest_cache.strategy_code에서 로드."""
    spec = TARGETS[name]
    if spec["code_source"] == "DEFAULT_STRATEGY_CODE":
        return DEFAULT_STRATEGY_CODE
    conn = get_conn()
    raw = conn._conn.cursor() if hasattr(conn, "_conn") else conn.cursor()
    raw.execute(
        "SELECT strategy_code FROM alpha_lab.backtest_cache "
        "WHERE name=%s AND universe=%s AND rebal_type=%s",
        (name, spec["universe"], spec["rebal_type"]),
    )
    row = raw.fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    return row[0]


def _pick(result: dict, key: str):
    """result["CUSTOM"][key] 또는 result[key]에서 안전하게 추출."""
    if not isinstance(result, dict):
        return None
    if "CUSTOM" in result and key in result["CUSTOM"]:
        return result["CUSTOM"][key]
    return result.get(key)


def _fmt_pct(v):
    return f"{v*100:+.2f}%" if isinstance(v, (int, float)) else "n/a"


def _fmt_num(v, prec=2):
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "n/a"


def run_one(name: str) -> dict | None:
    """단일 전략 실행."""
    spec = TARGETS[name]
    code = load_strategy_code(name)
    if not code:
        print(f"  ✗ {name}: strategy_code 못 찾음")
        return None

    print(f"\n  ▶ {name} ({spec['universe']}/{spec['rebal_type']}) 백테스트 시작")
    clear_factor_cache()
    clear_prefetch_cache()

    result = run_strategy_backtest(
        strategy_code=code,
        universe=spec["universe"],
        rebal_type=spec["rebal_type"],
    )

    if not result or "error" in result:
        err = result.get("error") if result else "no result"
        print(f"    ✗ 실패: {err}")
        return None

    return result


def print_summary(name: str, result: dict):
    """단일 전략 결과 요약."""
    tr = _pick(result, "total_return")
    cagr = _pick(result, "cagr")
    mdd = _pick(result, "mdd")
    sharpe = _pick(result, "sharpe")
    mstd = _pick(result, "monthly_std")
    turnover = _pick(result, "avg_turnover")
    months = _pick(result, "months")
    avg_size = _pick(result, "avg_portfolio_size")

    print(f"\n  ── {name} ──")
    print(f"    기간(개월):     {months}")
    print(f"    평균 종목수:    {_fmt_num(avg_size, 1)}")
    print(f"    총수익률:       {_fmt_pct(tr)}")
    print(f"    CAGR:           {_fmt_pct(cagr)}")
    print(f"    MDD:            {_fmt_pct(mdd)}")
    print(f"    Sharpe:         {_fmt_num(sharpe, 2)}")
    print(f"    월간 변동성:    {_fmt_pct(mstd)}")
    print(f"    평균 회전율:    {_fmt_pct(turnover)}")


def print_latest_holdings(name: str, result: dict, top: int = 30):
    """가장 최근 리밸 날짜의 편입 종목과 비중."""
    custom = result.get("CUSTOM", result) if isinstance(result, dict) else {}
    dates = custom.get("rebalance_dates", [])
    holdings = custom.get("holdings", None) or custom.get("holdings_by_date", None)
    if not dates or not holdings:
        print(f"    (holdings 정보 없음 — name={name})")
        return

    last_date = dates[-1]
    last_holdings = holdings.get(last_date) if isinstance(holdings, dict) else None
    if not last_holdings:
        print(f"    (마지막 리밸 {last_date} holdings 없음)")
        return

    print(f"\n  ── {name} 최근 리밸({last_date}) 편입 종목 ──")
    if isinstance(last_holdings, list):
        rows = last_holdings
    elif isinstance(last_holdings, dict):
        rows = [{"code": k, **(v if isinstance(v, dict) else {"weight": v})}
                for k, v in last_holdings.items()]
    else:
        print(f"    (holdings 형식 미지원: {type(last_holdings).__name__})")
        return

    def _w(r):
        if not isinstance(r, dict):
            return 0
        return r.get("weight") or r.get("w") or 0

    rows = sorted(rows, key=_w, reverse=True)[:top]
    print(f"    {'#':>3}  {'코드':<8}  {'종목명':<20}  {'비중':>8}")
    for i, r in enumerate(rows, 1):
        code = r.get("code") or r.get("stock_code") or "?"
        nm = (r.get("name") or r.get("종목명") or "")[:20]
        w = _w(r)
        w_str = f"{w*100:.2f}%" if isinstance(w, (int, float)) else str(w)
        print(f"    {i:>3}  {code:<8}  {nm:<20}  {w_str:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="backtest_cache에 결과 저장")
    parser.add_argument("--only", type=str, default=None,
                        choices=list(TARGETS.keys()) + [None],
                        help="특정 전략만 실행")
    parser.add_argument("--no-holdings", action="store_true", help="최근 holdings 출력 생략")
    args = parser.parse_args()

    print("=" * 60)
    print("  forward fix 적용 효과 핀포인트 백테스트")
    print("  대상: A0 + 수정전략_코스피_cap30%_top30_tx30bp_월간")
    print("=" * 60)

    targets = [args.only] if args.only else list(TARGETS.keys())
    results = {}

    for name in targets:
        result = run_one(name)
        if result is None:
            continue
        results[name] = result
        print_summary(name, result)
        if not args.no_holdings:
            print_latest_holdings(name, result)

        if args.save:
            code = load_strategy_code(name)
            spec = TARGETS[name]
            try:
                save_strategy(
                    name=name,
                    code=code,
                    results=result,
                    universe=spec["universe"],
                    rebal_type=spec["rebal_type"],
                )
                print(f"    ✓ backtest_cache 저장 완료")
            except Exception as e:
                print(f"    ✗ 저장 실패: {e}")

    print("\n" + "=" * 60)
    print("  완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
