"""analysis/profile_slice.py

slice 최적화 (prefetch date 인덱스 미리 계산) PoC.

- 옛 로직 (매 호출 unique() + sort + boolean indexing) vs 새 로직 (searchsorted + iloc)
- 호출 시간 비교 (price/forward/master)
- 결과 동일성 검증
- 메모리 추가 사용량 측정

사용:
    python analysis/profile_slice.py
    python analysis/profile_slice.py --calc-date 2024-12-01 --repeat 10

주의:
- Railway PG 에 접속 (.env 의 DATABASE_URL)
- prefetch_all_data 가 disk cache 사용 (있으면 빠름)
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
import pandas as pd

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from lib.db import get_conn
from lib.factor_engine import (
    prefetch_all_data,
    _prefetch_cache,
    _get_price_for_date,
    _get_fwd_for_date,
    _get_master_for_date,
)


def mem_mb() -> float:
    if not _HAS_PSUTIL:
        return -1.0
    return psutil.Process().memory_info().rss / 1024 / 1024


# ═══════════════════════════════════════════════════════
# 옛 로직 (PR #25 시점) — 비교용 인라인 복사
# ═══════════════════════════════════════════════════════

def old_get_price(calc_date: str):
    price_all = _prefetch_cache["price"]
    dates = price_all["trade_date"].unique()
    dates_before = sorted(d for d in dates if d < calc_date)
    price_date = dates_before[-1] if dates_before else None
    price_df = (
        price_all[price_all["trade_date"] == price_date][
            ["stock_code", "close", "market_cap", "trade_amount"]
        ]
        if price_date else pd.DataFrame()
    )
    three_m_ago = (datetime.strptime(calc_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    dates_3m = sorted(d for d in dates if d <= three_m_ago)
    price_date_3m = dates_3m[-1] if dates_3m else None
    price_3m_df = pd.DataFrame()
    if price_date_3m:
        price_3m_df = price_all[price_all["trade_date"] == price_date_3m][
            ["stock_code", "close"]
        ].copy()
        price_3m_df = price_3m_df.rename(columns={"close": "close_3m"})
    return price_df, price_3m_df


def old_get_fwd(calc_date: str):
    fwd_all = _prefetch_cache["forward"]
    dates = fwd_all["trade_date"].unique()
    dates_before = sorted(d for d in dates if d < calc_date)
    fwd_date = dates_before[-1] if dates_before else (sorted(dates)[0] if len(dates) else None)
    fwd_df = (
        fwd_all[fwd_all["trade_date"] == fwd_date].drop(columns=["trade_date"])
        if fwd_date else pd.DataFrame()
    )
    three_m_ago = (datetime.strptime(calc_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    dates_3m = sorted(d for d in dates if d <= three_m_ago)
    fwd_date_3m = dates_3m[-1] if dates_3m else None
    fwd_3m_df = pd.DataFrame()
    if fwd_date_3m:
        fwd_3m_df = fwd_all[fwd_all["trade_date"] == fwd_date_3m][["stock_code", "fwd_eps"]].copy()
        fwd_3m_df = fwd_3m_df.rename(columns={"fwd_eps": "fwd_eps_3m"})
    return fwd_df, fwd_3m_df


def old_get_master(calc_date: str):
    master_all = _prefetch_cache["master"]
    snap_month = calc_date[:7]
    snaps = sorted(master_all["snapshot_date"].unique())
    valid_snaps = [s for s in snaps if s <= snap_month]
    snap = valid_snaps[-1] if valid_snaps else (snaps[0] if snaps else snap_month)
    return master_all[master_all["snapshot_date"] == snap][
        ["stock_code", "stock_name", "market", "sec_cd_nm", "finacc_typ"]
    ]


# ═══════════════════════════════════════════════════════
# 측정 유틸
# ═══════════════════════════════════════════════════════

def time_fn(fn, args, repeat: int) -> list[float]:
    times = []
    for _ in range(repeat):
        t = time.time()
        fn(*args)
        times.append(time.time() - t)
    return times


def fmt(times):
    arr = np.array(times) * 1000
    return f"avg {arr.mean():7.1f}ms, min {arr.min():7.1f}ms, max {arr.max():7.1f}ms"


def df_equal(a: pd.DataFrame, b: pd.DataFrame, key_col: str = "stock_code") -> tuple[bool, str]:
    """동일성 검사. (정렬 무관, 인덱스 무관) — key_col 기준 정렬 후 값 비교."""
    if len(a) != len(b):
        return False, f"length 다름: {len(a)} vs {len(b)}"
    if set(a.columns) != set(b.columns):
        return False, f"컬럼 다름: {sorted(a.columns)} vs {sorted(b.columns)}"
    a_sorted = a.sort_values(key_col).reset_index(drop=True)
    b_sorted = b.sort_values(key_col).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(a_sorted, b_sorted, check_dtype=False, check_like=True)
        return True, "동일"
    except AssertionError as e:
        return False, str(e).split("\n")[0][:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc-date", default="2024-12-01")
    ap.add_argument("--repeat", type=int, default=10,
                    help="호출 반복 횟수 (warm-up 1 추가)")
    args = ap.parse_args()

    print("=" * 70)
    print(f"  Profile slice  calc_date={args.calc_date}  repeat={args.repeat}")
    print("=" * 70)

    mem0 = mem_mb()
    if mem0 >= 0:
        print(f"\nMemory baseline: {mem0:.0f} MB")

    conn = get_conn()

    print("\n[1] prefetch_all_data — 캐시 로딩 + date 인덱스 빌드")
    t0 = time.time()
    prefetch_all_data(conn, use_local_cache=True)
    print(f"  total: {time.time()-t0:.1f}s")
    mem1 = mem_mb()
    if mem1 >= 0:
        print(f"  memory: {mem1:.0f} MB  (+{mem1-mem0:.0f} MB vs baseline)")

    # 인덱스 캐시 확인
    print("\n  date 인덱스 확인:")
    for name in ("price", "forward", "master"):
        ud = _prefetch_cache.get(f"_{name}_unique_dates")
        if ud is not None:
            print(f"    {name}: unique dates {len(ud)}, "
                  f"date_ranges {len(_prefetch_cache.get(f'_{name}_date_ranges', {}))} entries")
        else:
            print(f"    {name}: 인덱스 미생성")

    # ──── [2] 호출 시간 비교 ────
    print(f"\n[2] 호출 시간 비교 (warm-up 1회 + 측정 {args.repeat}회)\n")

    # warm-up
    old_get_price(args.calc_date)
    old_get_fwd(args.calc_date)
    old_get_master(args.calc_date)
    _get_price_for_date(args.calc_date)
    _get_fwd_for_date(args.calc_date)
    _get_master_for_date(args.calc_date)

    # 측정
    funcs = [
        ("price",   old_get_price,      _get_price_for_date),
        ("forward", old_get_fwd,        _get_fwd_for_date),
        ("master",  old_get_master,     _get_master_for_date),
    ]
    results = []
    for name, old_fn, new_fn in funcs:
        old_times = time_fn(old_fn, (args.calc_date,), args.repeat)
        new_times = time_fn(new_fn, (args.calc_date,), args.repeat)
        old_avg = np.mean(old_times) * 1000
        new_avg = np.mean(new_times) * 1000
        speedup = old_avg / new_avg if new_avg > 0 else float("inf")
        print(f"  [{name:7}]  옛: {fmt(old_times)}")
        print(f"             새: {fmt(new_times)}")
        print(f"             속도 향상: {speedup:.1f}x\n")
        results.append((name, old_avg, new_avg, speedup))

    # ──── [3] 결과 동일성 검증 ────
    print("[3] 결과 동일성 검증")
    # price: tuple (df, df_3m)
    op = old_get_price(args.calc_date)
    np_ = _get_price_for_date(args.calc_date)
    ok1, msg1 = df_equal(op[0], np_[0])
    ok2, msg2 = df_equal(op[1], np_[1]) if not op[1].empty or not np_[1].empty else (True, "둘 다 empty")
    print(f"  price       df : {'OK' if ok1 else 'DIFF'} ({msg1})")
    print(f"  price_3m    df : {'OK' if ok2 else 'DIFF'} ({msg2})")

    of = old_get_fwd(args.calc_date)
    nf = _get_fwd_for_date(args.calc_date)
    ok3, msg3 = df_equal(of[0], nf[0])
    ok4, msg4 = df_equal(of[1], nf[1]) if not of[1].empty or not nf[1].empty else (True, "둘 다 empty")
    print(f"  forward     df : {'OK' if ok3 else 'DIFF'} ({msg3})")
    print(f"  forward_3m  df : {'OK' if ok4 else 'DIFF'} ({msg4})")

    om = old_get_master(args.calc_date)
    nm = _get_master_for_date(args.calc_date)
    ok5, msg5 = df_equal(om, nm)
    print(f"  master      df : {'OK' if ok5 else 'DIFF'} ({msg5})")

    # ──── [4] 요약 ────
    print("\n" + "=" * 70)
    print("  요약")
    print("=" * 70)
    total_old = sum(r[1] for r in results)
    total_new = sum(r[2] for r in results)
    speedup_total = total_old / total_new if total_new > 0 else float("inf")
    print(f"\n호출당 합계 (3개 함수):")
    print(f"  옛:  {total_old:7.1f}ms")
    print(f"  새:  {total_new:7.1f}ms")
    print(f"  속도 향상: {speedup_total:.1f}x")

    print(f"\n예상 누적 (98회 리밸):")
    print(f"  옛:  {total_old * 98 / 1000:.1f}s")
    print(f"  새:  {total_new * 98 / 1000:.1f}s")
    print(f"  절감: {(total_old - total_new) * 98 / 1000:.1f}s")

    if mem1 >= 0:
        print(f"\n메모리 (slice 인덱스 추가 부담):")
        print(f"  baseline + cache: +{mem1 - mem0:.0f}MB")
        print(f"  (date 인덱스 자체는 ~수 MB 수준 — 별도 측정 어려움)")

    all_ok = ok1 and ok2 and ok3 and ok4 and ok5
    print(f"\n결과 동일성: {'✅ 모두 OK' if all_ok else '❌ 차이 발견'}")
    print()


if __name__ == "__main__":
    main()
