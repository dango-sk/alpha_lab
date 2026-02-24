"""
대시보드 캐시 빌드: step7 → step8 결과를 JSON으로 사전 생성.
대시보드(streamlit) 실행 전에 한 번 돌리면 즉시 로딩됩니다.

사용법:
    python scripts/build_cache.py
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from step7_backtest import run_all_backtests, save_backtest_cache
from step8_robustness import (
    test_is_oos_split,
    test_statistical_significance,
    test_rolling_window,
    save_robustness_cache,
)

def main():
    t0 = time.time()

    # ── Step 7: 백테스트 ──
    print("\n[1/2] 백테스트 실행 중...")
    results = run_all_backtests()
    if results:
        save_backtest_cache(results)

    # ── Step 8: 강건성 검증 ──
    print("\n[2/2] 강건성 검증 실행 중...")
    is_oos = test_is_oos_split()
    stat = test_statistical_significance()
    rolling = test_rolling_window(stat["full_results"])
    save_robustness_cache(is_oos, stat, rolling)

    elapsed = time.time() - t0
    print(f"\n완료! ({elapsed:.0f}초)")
    print("이제 streamlit run app.py 로 대시보드를 즉시 열 수 있습니다.")


if __name__ == "__main__":
    main()
