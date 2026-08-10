#!/usr/bin/env bash
#
# 월별 레짐 파이프라인 일괄 실행
# ─────────────────────────────────────────────────────────────
#   0. 데이터 최신화   collect_macro / backfill_global_indices / collect_technical
#   1. AI 레짐 예측    regime_agent.py (as_of = 해당 월 1일, 직전일까지 데이터로 예측)
#   2. 백테스트        run_pipeline.py --monthly (ai 모드가 1번 결과 읽어 Bull/Bear 판정)
#
# 사용법:
#   ./scripts/run_monthly_with_regime.sh            # 이번 달 예측 (date +%Y-%m)
#   ./scripts/run_monthly_with_regime.sh 2026-07    # 특정 월 강제 (월말에 다음달 미리 돌릴 때)
#   ./scripts/run_monthly_with_regime.sh 2026-07 -y # 확인 프롬프트 스킵
#
# ⚠ 전제: daily_price / marketcap 는 LG 그램 → Railway PG 로 별도 업로드되어 있어야 함.
#         (이 스크립트로는 못 함. 6/30 데이터가 PG에 없으면 그 전 영업일 기준으로 예측됨)
#
set -euo pipefail

# ── 경로 (스크립트 위치 기준 → LG 그램에서도 동작) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT"

# ── 인자 파싱: [YYYY-MM] [-y] ──
REGIME_MONTH=""
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes) ASSUME_YES=1 ;;
    [0-9][0-9][0-9][0-9]-[0-9][0-9]) REGIME_MONTH="$arg" ;;
    *) echo "❌ 알 수 없는 인자: $arg (형식: YYYY-MM 또는 -y)"; exit 1 ;;
  esac
done
[ -z "$REGIME_MONTH" ] && REGIME_MONTH="$(date +%Y-%m)"

echo "════════════════════════════════════════════════════════════"
echo "  월별 레짐 파이프라인"
echo "  대상 월(as_of=${REGIME_MONTH}-01) : ${REGIME_MONTH}"
echo "  Python                          : $PY"
echo "════════════════════════════════════════════════════════════"
echo "  ⚠ daily_price/marketcap 가 PG에 최신 업로드되어 있는지 먼저 확인하세요."
echo "    (LG 그램 → Railway PG, 이 스크립트 범위 밖)"
echo "────────────────────────────────────────────────────────────"

if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "  계속 진행할까요? [y/N] " ans
  case "$ans" in
    y|Y|yes|YES) ;;
    *) echo "  중단."; exit 0 ;;
  esac
fi

step() { echo; echo "▶▶▶ $1"; echo "────────────────────────────────────────────────────────────"; }

# ── 0. 데이터 최신화 ──
step "0-1) 매크로 지표 수집 (collect_macro.py)"
"$PY" scripts/collect_macro.py

step "0-2) 글로벌 지수 종가 backfill (backfill_global_indices.py)"
"$PY" scripts/backfill_global_indices.py

step "0-3) technical_indicators 계산 (collect_technical.py — 종가 의존, 반드시 0-1·0-2 이후)"
"$PY" scripts/legacy/collect_technical.py

# ── 1. AI 레짐 예측 (--model 없이 = production regime_agent_results.json 에 append) ──
step "1) AI 레짐 예측 → regime_agent_results.json (as_of=${REGIME_MONTH}-01)"
"$PY" analysis/regime_agent.py --start "$REGIME_MONTH" --end "$REGIME_MONTH"

# ── 2. 백테스트 (월초 전체: 마스터+재무+TTM+유니버스+레짐조합) ──
step "2) 백테스트 파이프라인 (run_pipeline.py --monthly)"
"$PY" scripts/run_pipeline.py --monthly

echo
echo "════════════════════════════════════════════════════════════"
echo "  ✅ 완료: ${REGIME_MONTH} 레짐 예측 + 월초 백테스트"
echo "════════════════════════════════════════════════════════════"
