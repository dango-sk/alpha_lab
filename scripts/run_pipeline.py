"""
Alpha Lab 전체 파이프라인 실행
실행: python scripts/run_pipeline.py

순서:
  1. Step 1: 주가 업데이트 (alpha_radar DB)
  2. Step 3: 밸류 팩터 계산
  3. Step 6: 시그널 생성
  4. Step 7: 백테스트
  5. Step 8: 강건성 검증
  6. 캐시 빌드 (대시보드용)

소요 시간: 약 20~40분 (step1 주가 수집이 대부분)
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))


def log(msg):
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Alpha Lab 전체 파이프라인")
    print("=" * 60)

    # ── Step 1: 주가 업데이트 ──
    log("[1/5] Step 1: 주가 업데이트...")
    from step1_update_prices import update_prices, show_summary
    update_prices()
    show_summary()
    log("  -> Step 1 완료")

    # ── Step 3: 밸류 팩터 계산 ──
    log("[2/5] Step 3: 밸류 팩터 계산...")
    from step3_calc_value_factors import calc_valuation_for_date
    from config.settings import DB_PATH, BACKTEST_CONFIG
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH))
    # 백테스트 기간의 월별 첫 거래일 목록
    trade_dates = conn.execute("""
        SELECT DISTINCT trade_date FROM daily_price
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, (BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"])).fetchall()

    monthly_dates = []
    current_month = ""
    for (td,) in trade_dates:
        month = td[:7]
        if month != current_month:
            monthly_dates.append(td)
            current_month = month
    conn.close()

    # 이미 계산된 날짜는 스킵
    conn = sqlite3.connect(str(DB_PATH))
    existing_dates = set(
        row[0] for row in conn.execute(
            "SELECT DISTINCT calc_date FROM valuation_factors"
        ).fetchall()
    )
    conn.close()

    new_dates = [d for d in monthly_dates if d not in existing_dates]
    if new_dates:
        print(f"  신규 계산 필요: {len(new_dates)}개월")
        for date in new_dates:
            calc_valuation_for_date(target_date=date)
    else:
        print(f"  모든 {len(monthly_dates)}개월 이미 계산됨 -> 스킵")
    log("  -> Step 3 완료")

    # ── Step 6: 시그널 생성 ──
    log("[3/5] Step 6: 시그널 생성...")
    from step6_generate_signals import generate_signals_for_date

    conn = sqlite3.connect(str(DB_PATH))
    existing_signals = set(
        row[0] for row in conn.execute(
            "SELECT DISTINCT calc_date FROM signals"
        ).fetchall()
    )
    conn.close()

    new_signal_dates = [d for d in monthly_dates if d not in existing_signals]
    if new_signal_dates:
        print(f"  신규 시그널 필요: {len(new_signal_dates)}개월")
        for date in new_signal_dates:
            generate_signals_for_date(date)
    else:
        print(f"  모든 {len(monthly_dates)}개월 시그널 있음 -> 스킵")
    log("  -> Step 6 완료")

    # ── Step 7 + 8: 백테스트 + 강건성 + 캐시 ──
    log("[4/5] Step 7: 백테스트...")
    from step7_backtest import run_all_backtests, save_backtest_cache
    results = run_all_backtests()
    if results:
        save_backtest_cache(results)
    log("  -> Step 7 완료")

    log("[5/5] Step 8: 강건성 검증...")
    from step8_robustness import (
        test_is_oos_split,
        test_statistical_significance,
        test_rolling_window,
        save_robustness_cache,
    )
    is_oos = test_is_oos_split()
    stat = test_statistical_significance()
    rolling = test_rolling_window(stat["full_results"])
    save_robustness_cache(is_oos, stat, rolling)
    log("  -> Step 8 완료")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  전체 파이프라인 완료! ({elapsed / 60:.1f}분)")
    print("  streamlit run app.py 로 대시보드를 실행하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
