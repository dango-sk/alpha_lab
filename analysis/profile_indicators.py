"""analysis/profile_indicators.py

옵션 C (precompute rolling + drop_duplicates) vs 현재 (PR #22 slice+groupby) 비교 PoC.

- 단일 calc_date 에서 호출 시간 측정 (warm-up 후 repeat)
- 결과 동일성 검증 (price_ma_rev, below_ma, mfi)
- 메모리 사용량 비교
- 예상 백테스트 누적 시간 계산 (98회 호출 기준)

사용:
    python analysis/profile_indicators.py
    python analysis/profile_indicators.py --calc-date 2024-12-01 --repeat 5

주의:
- Railway PG 에 접속 (.env 의 DATABASE_URL)
- alpha_lab.stock_indicators 테이블 필요 (6.4M 행)
- 메모리 ~1GB 사용 가능
"""
import argparse
import os
import sys
import time
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

from lib.db import get_conn, read_sql
from lib.factor_engine import (
    _calc_ma_reversion_from_db,
    _calc_mfi_from_db,
    _ensure_indicators_db_cache,
    clear_indicators_db_cache,
)


def get_memory_mb() -> float:
    """psutil 있으면 RSS 메모리 (MB), 없으면 -1 (스킵 표시)."""
    if not _HAS_PSUTIL:
        return -1.0
    return psutil.Process().memory_info().rss / 1024 / 1024


def _mem_line(label: str, mem: float, base: float | None = None) -> str:
    if mem < 0:
        return f"  memory: (psutil 없음 — 건너뜀)"
    delta = f"  (+{mem - base:.0f} MB)" if base is not None and base >= 0 else ""
    return f"  memory: {mem:.0f} MB{delta}"


# ═══════════════════════════════════════════════════════
# 옵션 C: precompute rolling + drop_duplicates
# ═══════════════════════════════════════════════════════

def load_and_precompute(conn) -> pd.DataFrame:
    """alpha_lab.stock_indicators 로드 + ma_reversion / mfi 결과를 컬럼으로 precompute."""
    t0 = time.time()
    df = read_sql("""
        SELECT stock_code, trade_date, ma_120, deviation_120,
               mfi_val, pos_sum_14, neg_sum_14
        FROM alpha_lab.stock_indicators
        ORDER BY stock_code, trade_date
    """, conn)
    df = df.reset_index(drop=True)
    print(f"  [precompute] load: {time.time()-t0:.1f}s, {len(df):,} rows")

    # ma_reversion: 종목별 deviation_120 의 rolling(250).min() — 절대값
    t1 = time.time()
    df["_price_ma_rev"] = (
        df.groupby("stock_code", sort=False)["deviation_120"]
        .rolling(250, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
        .abs()
    )
    print(f"  [precompute] ma_reversion rolling.min: {time.time()-t1:.1f}s")

    # below_ma: deviation_120 <= 0
    t2 = time.time()
    df["_below_ma"] = (df["deviation_120"] <= 0).astype(int)
    print(f"  [precompute] below_ma: {time.time()-t2:.1f}s")

    # MFI: (current_mfi - mfi_19_ago) + adjustment
    # 기존 _calc_mfi 의 tail(lookback=20) 의 first/last 와 동일 의미
    # (lookback=20 의 첫 값 = 19일 전 = shift(19))
    t3 = time.time()
    df["_mfi_lag19"] = df.groupby("stock_code", sort=False)["mfi_val"].shift(19)
    df["_mfi"] = (df["mfi_val"] - df["_mfi_lag19"]) + np.where(
        df["mfi_val"] < 50,
        (50.0 - df["mfi_val"]) * 0.2,
        0.0,
    )
    print(f"  [precompute] mfi: {time.time()-t3:.1f}s")

    print(f"  [precompute] TOTAL: {time.time()-t0:.1f}s")
    return df


def calc_ma_reversion_precompute(df_pre: pd.DataFrame, calc_date: str) -> pd.DataFrame:
    """옵션 C: precompute 컬럼에서 종목별 마지막 행 lookup."""
    sub = df_pre[df_pre["trade_date"] < calc_date]
    if sub.empty:
        return pd.DataFrame(columns=["stock_code", "price_ma_rev", "below_ma"])
    last_per_stock = sub.drop_duplicates(subset="stock_code", keep="last")
    valid = last_per_stock[last_per_stock["ma_120"].notna()]
    return (
        valid[["stock_code", "_price_ma_rev", "_below_ma"]]
        .rename(columns={"_price_ma_rev": "price_ma_rev", "_below_ma": "below_ma"})
        .reset_index(drop=True)
    )


def calc_mfi_precompute(df_pre: pd.DataFrame, calc_date: str) -> pd.DataFrame:
    """옵션 C: precompute 컬럼에서 종목별 마지막 행 lookup."""
    sub = df_pre[df_pre["trade_date"] < calc_date]
    if sub.empty:
        return pd.DataFrame(columns=["stock_code", "mfi"])
    last_per_stock = sub.drop_duplicates(subset="stock_code", keep="last")
    valid = last_per_stock[last_per_stock["_mfi"].notna()]
    return (
        valid[["stock_code", "_mfi"]]
        .rename(columns={"_mfi": "mfi"})
        .reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════
# 비교 실행
# ═══════════════════════════════════════════════════════

def time_calls(fn, args, repeat: int) -> list[float]:
    times = []
    for _ in range(repeat):
        t = time.time()
        fn(*args)
        times.append(time.time() - t)
    return times


def fmt(ms_list):
    arr = np.array(ms_list) * 1000
    return f"avg {arr.mean():.0f}ms, min {arr.min():.0f}ms, max {arr.max():.0f}ms"


def diff_df(a: pd.DataFrame, b: pd.DataFrame, key: str, val_cols: list[str]) -> dict:
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    only_a = set(a[key]) - set(b[key])
    only_b = set(b[key]) - set(a[key])
    out = {
        "count_a": len(a),
        "count_b": len(b),
        "common": len(merged),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "only_a_sample": list(only_a)[:3],
        "only_b_sample": list(only_b)[:3],
    }
    for col in val_cols:
        if f"{col}_a" in merged.columns:
            d = (merged[f"{col}_a"] - merged[f"{col}_b"]).abs()
            out[f"{col}_diff_max"] = float(d.max()) if len(d) else 0.0
            out[f"{col}_diff_mean"] = float(d.mean()) if len(d) else 0.0
            out[f"{col}_diff_nonzero"] = int((d > 1e-9).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc-date", default="2024-12-01",
                    help="단일 calc_date 측정 기준 (default 2024-12-01)")
    ap.add_argument("--repeat", type=int, default=5,
                    help="호출 반복 횟수 (warm-up 1 추가 + 통계용)")
    args = ap.parse_args()

    print("=" * 70)
    print(f"  Profile indicators  calc_date={args.calc_date}  repeat={args.repeat}")
    print("=" * 70)

    mem_baseline = get_memory_mb()
    if mem_baseline >= 0:
        print(f"\nMemory baseline: {mem_baseline:.0f} MB")
    else:
        print("\nMemory: psutil 미설치 — 메모리 측정 건너뜀 (시간 측정만 진행)")

    conn = get_conn()

    # ──── [1] 현재 방식 (PR #22 slice+groupby) ────
    print("\n[1] 현재 방식 (slice+groupby) — cache load")
    t0 = time.time()
    _ensure_indicators_db_cache(conn)
    print(f"  load: {time.time()-t0:.1f}s")
    mem_curr = get_memory_mb()
    print(_mem_line("curr", mem_curr, mem_baseline))

    # warm-up 1회
    _calc_ma_reversion_from_db(conn, args.calc_date)
    _calc_mfi_from_db(conn, args.calc_date)

    print(f"\n[1] 현재 방식 호출 시간 ({args.repeat}회):")
    ma_curr_times = time_calls(_calc_ma_reversion_from_db, (conn, args.calc_date), args.repeat)
    mfi_curr_times = time_calls(_calc_mfi_from_db, (conn, args.calc_date), args.repeat)
    print(f"  ma_reversion: {fmt(ma_curr_times)}")
    print(f"  mfi:          {fmt(mfi_curr_times)}")

    # 결과 저장
    ma_curr = _calc_ma_reversion_from_db(conn, args.calc_date)
    mfi_curr = _calc_mfi_from_db(conn, args.calc_date)

    # ──── [2] 옵션 C (precompute) ────
    clear_indicators_db_cache()
    mem_after_clear = get_memory_mb()
    print(f"\n[2] 옵션 C (precompute) — cache build")
    if mem_after_clear >= 0:
        print(f"  memory after clear: {mem_after_clear:.0f} MB")
    t0 = time.time()
    df_pre = load_and_precompute(conn)
    print(f"  TOTAL build: {time.time()-t0:.1f}s")
    mem_pre = get_memory_mb()
    print(_mem_line("pre", mem_pre, mem_after_clear))

    # warm-up 1회
    calc_ma_reversion_precompute(df_pre, args.calc_date)
    calc_mfi_precompute(df_pre, args.calc_date)

    print(f"\n[2] 옵션 C 호출 시간 ({args.repeat}회):")
    ma_pre_times = time_calls(calc_ma_reversion_precompute, (df_pre, args.calc_date), args.repeat)
    mfi_pre_times = time_calls(calc_mfi_precompute, (df_pre, args.calc_date), args.repeat)
    print(f"  ma_reversion: {fmt(ma_pre_times)}")
    print(f"  mfi:          {fmt(mfi_pre_times)}")

    ma_pre = calc_ma_reversion_precompute(df_pre, args.calc_date)
    mfi_pre = calc_mfi_precompute(df_pre, args.calc_date)

    # ──── [3] 결과 동일성 검증 ────
    print("\n[3] 결과 동일성 — ma_reversion")
    d = diff_df(ma_curr, ma_pre, "stock_code", ["price_ma_rev", "below_ma"])
    print(f"  종목 수 — 현재: {d['count_a']}, 옵션 C: {d['count_b']}, 공통: {d['common']}")
    print(f"  only_현재: {d['only_a']} (e.g., {d['only_a_sample']})")
    print(f"  only_옵션C: {d['only_b']} (e.g., {d['only_b_sample']})")
    print(f"  price_ma_rev: max diff {d['price_ma_rev_diff_max']:.6f}, "
          f"mean diff {d['price_ma_rev_diff_mean']:.6f}, "
          f"nonzero diff {d['price_ma_rev_diff_nonzero']}")
    print(f"  below_ma:     mismatch {d['below_ma_diff_nonzero']}")

    print("\n[3] 결과 동일성 — mfi")
    d = diff_df(mfi_curr, mfi_pre, "stock_code", ["mfi"])
    print(f"  종목 수 — 현재: {d['count_a']}, 옵션 C: {d['count_b']}, 공통: {d['common']}")
    print(f"  only_현재: {d['only_a']} (e.g., {d['only_a_sample']})")
    print(f"  only_옵션C: {d['only_b']} (e.g., {d['only_b_sample']})")
    print(f"  mfi: max diff {d['mfi_diff_max']:.6f}, "
          f"mean diff {d['mfi_diff_mean']:.6f}, "
          f"nonzero diff {d['mfi_diff_nonzero']}")

    # ──── [4] 요약 ────
    curr_ma_ms = np.mean(ma_curr_times) * 1000
    curr_mfi_ms = np.mean(mfi_curr_times) * 1000
    pre_ma_ms = np.mean(ma_pre_times) * 1000
    pre_mfi_ms = np.mean(mfi_pre_times) * 1000

    print("\n" + "=" * 70)
    print("  요약")
    print("=" * 70)
    print(f"\n호출당 (ma + mfi):")
    print(f"  현재:    {curr_ma_ms + curr_mfi_ms:.0f}ms  "
          f"(ma {curr_ma_ms:.0f}ms + mfi {curr_mfi_ms:.0f}ms)")
    print(f"  옵션 C:  {pre_ma_ms + pre_mfi_ms:.0f}ms  "
          f"(ma {pre_ma_ms:.0f}ms + mfi {pre_mfi_ms:.0f}ms)")
    if pre_ma_ms + pre_mfi_ms > 0:
        speedup = (curr_ma_ms + curr_mfi_ms) / (pre_ma_ms + pre_mfi_ms)
        print(f"  속도 향상: {speedup:.1f}x")

    print(f"\n예상 누적 (98회 리밸):")
    print(f"  현재:    {(curr_ma_ms + curr_mfi_ms) * 98 / 1000:.0f}s")
    print(f"  옵션 C:  {(pre_ma_ms + pre_mfi_ms) * 98 / 1000:.0f}s")

    if mem_curr >= 0 and mem_pre >= 0:
        print(f"\n메모리:")
        print(f"  현재 cache:    +{mem_curr - mem_baseline:.0f}MB (DataFrame 1개)")
        print(f"  옵션 C cache:  +{mem_pre - mem_after_clear:.0f}MB "
              f"(DataFrame + precompute 컬럼 4개)")
    else:
        print(f"\n메모리: psutil 미설치 — 측정 건너뜀")
        print(f"  설치하려면: pip install psutil")

    print()


if __name__ == "__main__":
    main()
