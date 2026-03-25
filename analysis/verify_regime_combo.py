"""
레짐 조합 백테스트 검증 스크립트

검증 방법:
  각 날짜별로 레짐을 판단하고,
  "레짐이 유지된 달"에서 combo 수익률 == 해당 전략 수익률인지 확인.
  전환 달은 거래비용 차이가 있으므로 별도 표시.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lib.data import run_regime_combo_backtest, load_all_results, compute_regime_analysis

# ─── 설정 ───────────────────────────────────────────
BULL_KEY    = "A0"          # 실제 전략 키로 변경
BEAR_KEY    = "A0"          # 실제 전략 키로 변경
UNIVERSE    = "KOSPI"
REBAL_TYPE  = "monthly"
MA_WINDOW   = 50
TX_COST_THRESHOLD = 0.03    # 전환 달 거래비용 허용 범위
# ────────────────────────────────────────────────────

print("=" * 70)
print(f"레짐 조합 검증: Bull={BULL_KEY}  Bear={BEAR_KEY}")
print(f"MA={MA_WINDOW}일  Universe={UNIVERSE}  Rebal={REBAL_TYPE}")
print("=" * 70)

# 1. 레짐 조합 백테스트 실행
print("\n[1/3] 레짐 조합 백테스트 실행 중...")
result = run_regime_combo_backtest(BULL_KEY, BEAR_KEY, universe=UNIVERSE,
                                   rebal_type=REBAL_TYPE, ma_window=MA_WINDOW)

if not result or "REGIME_COMBO" not in result:
    print("ERROR: 백테스트 결과 없음")
    sys.exit(1)

combo = result["REGIME_COMBO"]
regime_by_date = combo.get("regime_by_date", {})
combo_dates    = combo.get("rebalance_dates", [])
combo_rets     = combo.get("monthly_returns", [])

print(f"  → 완료: {len(combo_rets)}개 기간, 총 수익률 {combo.get('total_return', 0)*100:.1f}%")

# 2. Bull/Bear 개별 전략 수익률 로드
print("\n[2/3] 개별 전략 수익률 로드...")
cached = load_all_results(universe=UNIVERSE, rebal_type=REBAL_TYPE)

bull_data = result.get(BULL_KEY) or cached.get(BULL_KEY)
bear_data = result.get(BEAR_KEY) or cached.get(BEAR_KEY)

if not bull_data or not bear_data:
    print("ERROR: 개별 전략 결과를 찾을 수 없습니다.")
    sys.exit(1)

# 날짜 → 수익률 딕셔너리
def make_ret_map(data):
    dates = data.get("rebalance_dates", [])
    rets  = data.get("monthly_returns", [])
    return {dates[i+1]: rets[i] for i in range(len(rets)) if i+1 < len(dates)}

bull_ret_map = make_ret_map(bull_data)
bear_ret_map = make_ret_map(bear_data)

print(f"  → Bull 전략: {len(bull_ret_map)}개월  Bear 전략: {len(bear_ret_map)}개월")

# 3. 월별 대조
print("\n[3/3] 월별 수익률 대조\n")
print(f"{'날짜':<12} {'레짐':<6} {'전환':<4} {'Combo':<8} {'예상':<8} {'차이':<8} {'판정'}")
print("-" * 70)

n = len(combo_rets)
ret_dates = combo_dates[1:n+1]   # 각 기간의 종료일 (수익률이 실현되는 날)
start_dates = combo_dates[:n]     # 각 기간의 시작일 (레짐 판단 기준)

prev_regime = None
mismatch_count = 0
transition_count = 0
stable_mismatch = 0

rows = []
for i, (start_d, end_d) in enumerate(zip(start_dates, ret_dates)):
    regime = regime_by_date.get(end_d)
    if not regime:
        continue

    is_transition = (prev_regime is not None and regime != prev_regime)
    if is_transition:
        transition_count += 1

    combo_ret = combo_rets[i]
    expected_map = bull_ret_map if regime == "Bull" else bear_ret_map
    expected_ret = expected_map.get(end_d)

    if expected_ret is None:
        prev_regime = regime
        continue

    diff = combo_ret - expected_ret
    abs_diff = abs(diff)

    if is_transition:
        verdict = "전환달(비용↑)"
    elif abs_diff < 0.001:
        verdict = "✓ 일치"
    elif abs_diff < TX_COST_THRESHOLD:
        verdict = "△ 근사"
    else:
        verdict = "✗ 불일치"
        mismatch_count += 1
        if not is_transition:
            stable_mismatch += 1

    rows.append({
        "date": end_d, "regime": regime, "transition": is_transition,
        "combo": combo_ret, "expected": expected_ret, "diff": diff, "verdict": verdict
    })
    prev_regime = regime

# 출력 (처음 30개 + 불일치 구간)
shown = set()
for r in rows[:30]:
    print(f"{r['date']:<12} {r['regime']:<6} {'←'if r['transition'] else '':<4} "
          f"{r['combo']*100:+6.2f}%  {r['expected']*100:+6.2f}%  {r['diff']*100:+6.2f}%  {r['verdict']}")
    shown.add(r['date'])

mismatches = [r for r in rows if r['verdict'].startswith('✗')]
if mismatches:
    print(f"\n--- 불일치 구간 ({len(mismatches)}개) ---")
    for r in mismatches:
        if r['date'] not in shown:
            print(f"{r['date']:<12} {r['regime']:<6} {'←'if r['transition'] else '':<4} "
                  f"{r['combo']*100:+6.2f}%  {r['expected']*100:+6.2f}%  {r['diff']*100:+6.2f}%  {r['verdict']}")

# 요약
print("\n" + "=" * 70)
print("검증 요약")
print("=" * 70)
total = len(rows)
transitions = sum(1 for r in rows if r['transition'])
stable_ok   = sum(1 for r in rows if not r['transition'] and r['verdict'] in ('✓ 일치', '△ 근사'))
stable_total = total - transitions

print(f"  전체 기간    : {total}개월")
print(f"  레짐 전환달  : {transitions}개월  (거래비용 차이 발생 → 정상)")
print(f"  레짐 유지달  : {stable_total}개월")
print(f"    일치/근사  : {stable_ok}개월")
print(f"    불일치     : {stable_mismatch}개월")

if stable_mismatch == 0:
    print("\n✅ 검증 통과: 레짐 유지 구간에서 수익률이 일치합니다.")
    print("   (전환 달 차이는 거래비용으로 설명 가능)")
else:
    print(f"\n❌ 검증 실패: 레짐 유지 구간에서 {stable_mismatch}개월 불일치.")
    print("   → 전략 선택 로직에 버그 가능성 있음.")
