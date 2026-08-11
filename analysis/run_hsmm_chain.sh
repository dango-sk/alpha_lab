#!/usr/bin/env bash
# HSMM 레짐 사슬 재실행 (2026-08 개정 반영: KOSPI-only breadth / semi-deviation / 확장평균 목표변동성)
#
#   1) hsmm_final.py               → analysis/hsmm_final_path.csv  (P_bear·익스포저 경로)  ※ 2·3의 입력
#   2) fcf_hsmm_overlay.py         → FCF불 전략 오버레이 A/B/pbear 비교
#   3) hsmm_exposure_volvariants.py→ vol-타겟 분모 4종 robustness + 차트
#
# 사용:  bash analysis/run_hsmm_chain.sh
# 로그:  analysis/results/run_<step>.log  (화면에도 그대로 출력)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
OUT="$REPO/analysis/results"
mkdir -p "$OUT"

[ -x "$PY" ] || { echo "!! .venv 없음: $PY"; exit 1; }
grep -q DATABASE_URL "$REPO/.env" 2>/dev/null || { echo "!! .env에 DATABASE_URL 없음"; exit 1; }

# 피처 캐시(analysis/.cache/hsmm_features.pkl)는 1번이 만들고 2·3번이 재사용한다.
# DB를 새로 읽으려면:  bash analysis/run_hsmm_chain.sh --refresh
REFRESH=""
[ "${1:-}" = "--refresh" ] && REFRESH="--refresh"

run () {                                  # run <라벨> <스크립트> [추가인자...]
  echo; echo "=============================================================="
  echo "  [$1] $2"
  echo "=============================================================="
  local lbl="$1" scr="$2"; shift 2
  PYTHONUTF8=1 PYTHONUNBUFFERED=1 "$PY" -u "$REPO/analysis/$scr" "$@" 2>&1 | tee "$OUT/run_$lbl.log"
}

run 1_final      hsmm_final.py ${REFRESH:+"$REFRESH"}
[ -f "$REPO/analysis/hsmm_final_path.csv" ] || { echo "!! hsmm_final_path.csv 미생성 → 중단"; exit 1; }
run 2_fcf        fcf_hsmm_overlay.py
run 3_volvar     hsmm_exposure_volvariants.py

echo
echo "완료. 산출물:"
echo "  analysis/hsmm_final_path.csv        (월별 P_bear·익스포저)"
echo "  analysis/fcf_overlay_series.csv     (오버레이 월수익 시계열)"
ls -1 "$OUT" | sed 's/^/  results\//'
