"""
analysis/inspect_factor_scores.py

backtest_cache.factor_scores_json에 저장된 종목별 팩터 raw/score/weight/contrib
값을 표 형태로 출력. 새 백테스트(`--only-backtest`) 후 어느 종목이 어떤 팩터로
선정됐는지, F_*가 실제로 raw 값을 가지고 들어갔는지 한눈에 확인용.

실행:
  python analysis/inspect_factor_scores.py --strategy A0                                    # 최근 리밸일
  python analysis/inspect_factor_scores.py --strategy "수정전략_코스피_cap30%_top30_tx30bp_월간"
  python analysis/inspect_factor_scores.py --strategy A0 --date 2026-05-15
  python analysis/inspect_factor_scores.py --strategy A0 --top 10                            # 상위 10개만
  python analysis/inspect_factor_scores.py --strategy A0 --csv out.csv                       # CSV로 저장
  python analysis/inspect_factor_scores.py --list                                            # 사용 가능한 전략 목록
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.db import get_conn


def list_strategies(conn):
    raw = conn._conn.cursor()
    raw.execute("""
        SELECT name, universe, rebal_type,
               (factor_scores_json IS NOT NULL) AS has_fs,
               updated_at
        FROM alpha_lab.backtest_cache
        ORDER BY (factor_scores_json IS NOT NULL) DESC, updated_at DESC
    """)
    rows = raw.fetchall()
    print(f"{'전략명':<60} {'universe':<14} {'rebal':<9} {'factor_scores':<14} {'updated_at'}")
    for name, uni, rt, has_fs, ts in rows:
        flag = "✓" if has_fs else "✗"
        print(f"{name[:60]:<60} {uni:<14} {rt:<9} {flag:<14} {ts}")


def load_factor_scores(conn, name: str, universe: str, rebal_type: str):
    raw = conn._conn.cursor()
    raw.execute("""
        SELECT factor_scores_json, results_json FROM alpha_lab.backtest_cache
        WHERE name=%s AND universe=%s AND rebal_type=%s
    """, (name, universe, rebal_type))
    row = raw.fetchone()
    if not row:
        return None, None
    fs, rj = row
    if isinstance(fs, str): fs = json.loads(fs)
    if isinstance(rj, str): rj = json.loads(rj)
    return fs, rj


def pick_date(fs: dict, date: str | None) -> str | None:
    """최신 리밸 날짜 또는 입력 날짜."""
    if not fs:
        return None
    keys = sorted(fs.keys())
    if date:
        if date not in fs:
            print(f"  날짜 {date} 없음. 가능한 날짜 (마지막 5개): {keys[-5:]}")
            return None
        return date
    return keys[-1]


def print_table(rows: list, date: str, top: int):
    """rows: [{code, name, factors:[{key,raw,score,weight,contrib}, ...]}, ...]"""
    if not rows:
        print("  종목 없음")
        return

    # 활성 팩터 키 추출 (weight > 0)
    sample = rows[0]
    factor_keys = [f["key"] for f in sample.get("factors", []) if (f.get("weight") or 0) > 0]
    if not factor_keys:
        factor_keys = [f["key"] for f in sample.get("factors", [])]

    print(f"\n  ── 리밸일 {date} | 상위 {min(top, len(rows))}/{len(rows)}종목 ──\n")

    # 헤더
    hdr = f"{'#':>3}  {'코드':<8}  {'종목명':<14}  {'final':>7}"
    for k in factor_keys:
        hdr += f"  {k:>11}"
    print(hdr)
    print(f"  {'':>3}  {'':<8}  {'':<14}  {'':>7}", end="")
    for _ in factor_keys:
        print(f"  {'raw|score':>11}", end="")
    print()

    for i, r in enumerate(rows[:top], 1):
        code = r.get("code", "?")
        nm = (r.get("name") or "")[:14]
        final = r.get("final_score") or sum((f.get("contrib") or 0) for f in r.get("factors", []))
        line = f"  {i:>3}  {code:<8}  {nm:<14}  {final:>7.2f}"
        f_map = {f["key"]: f for f in r.get("factors", [])}
        for k in factor_keys:
            f = f_map.get(k, {})
            raw = f.get("raw")
            score = f.get("score")
            raw_str = "null" if raw is None else (f"{raw:.2f}" if isinstance(raw, (int, float)) else str(raw)[:5])
            score_str = "-" if score is None else f"{score}"
            line += f"  {raw_str:>6}|{score_str:<4}"
        print(line)


def to_csv(rows: list, date: str, path: str):
    import csv
    if not rows:
        return
    factor_keys = [f["key"] for f in rows[0].get("factors", [])]
    fieldnames = ["rank", "code", "name", "final_score"]
    for k in factor_keys:
        fieldnames.extend([f"{k}_raw", f"{k}_score", f"{k}_weight", f"{k}_contrib"])

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            row = {
                "rank": i,
                "code": r.get("code", ""),
                "name": r.get("name", ""),
                "final_score": r.get("final_score") or sum((fc.get("contrib") or 0) for fc in r.get("factors", [])),
            }
            f_map = {fc["key"]: fc for fc in r.get("factors", [])}
            for k in factor_keys:
                fc = f_map.get(k, {})
                row[f"{k}_raw"] = fc.get("raw")
                row[f"{k}_score"] = fc.get("score")
                row[f"{k}_weight"] = fc.get("weight")
                row[f"{k}_contrib"] = fc.get("contrib")
            w.writerow(row)
    print(f"\n  ✓ CSV 저장: {path} (리밸일 {date}, {len(rows)}종목)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, help="전략명 (예: 'A0' 또는 '수정전략_...')")
    parser.add_argument("--universe", type=str, default="KOSPI")
    parser.add_argument("--rebal-type", type=str, default="monthly")
    parser.add_argument("--date", type=str, default=None, help="리밸 날짜 (미지정시 가장 최근)")
    parser.add_argument("--top", type=int, default=30, help="출력 종목 수")
    parser.add_argument("--csv", type=str, default=None, help="CSV 파일로 저장")
    parser.add_argument("--list", action="store_true", help="저장된 전략 목록만 보기")
    args = parser.parse_args()

    conn = get_conn()

    if args.list or not args.strategy:
        list_strategies(conn)
        return

    fs, _ = load_factor_scores(conn, args.strategy, args.universe, args.rebal_type)
    if fs is None:
        print(f"  '{args.strategy}' 캐시 없음. --list로 확인하세요.")
        return
    if not fs:
        print(f"  '{args.strategy}' 의 factor_scores_json이 비어있음. 백테스트를 다시 돌려야 합니다.")
        return

    date = pick_date(fs, args.date)
    if not date:
        return

    rows = fs[date]
    if not isinstance(rows, list):
        print(f"  factor_scores[{date}] 형식이 list가 아님: {type(rows).__name__}")
        return

    print(f"\n  전략: {args.strategy} ({args.universe}/{args.rebal_type})")
    print_table(rows, date, args.top)

    if args.csv:
        to_csv(rows, date, args.csv)


if __name__ == "__main__":
    main()
