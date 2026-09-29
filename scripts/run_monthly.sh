#!/usr/bin/env bash
#
# 월말/월초 실행 파이프라인 (2026-09 개편판)
# ─────────────────────────────────────────────────────────────
# 구 run_monthly_with_regime.sh 대체. 바뀐 점:
#   · AI 레짐(regime_agent.py) 제거 → HSMM(analysis/hsmm_final.py)
#   · 레짐조합 전략 제거 → 메인 전략 CORE 단일
#   · 유니버스 시총하한 5000억 → 2000억 (config/settings.py)
#   · prefetch 캐시 강제 갱신 + 검증 단계 추가
#
# 단계
#   0) 신선도 점검      check_freshness.py           — 원천 데이터 최신 여부
#   1) 데이터 최신화    collect_macro / backfill_global_indices / collect_technical
#   2) prefetch 갱신    PREFETCH_FORCE_REFRESH=1     — ★ 캐시 손상 방지 (2026-09 사고)
#   3) HSMM 레짐        analysis/hsmm_final.py       — exposure 경로 재계산
#   4) 파이프라인       run_pipeline.py --monthly    — 마스터/재무/TTM/유니버스/백테스트
#
# 사용법
#   ./scripts/run_monthly.sh            # 전체 실행 (확인 프롬프트)
#   ./scripts/run_monthly.sh -y         # 프롬프트 스킵
#   ./scripts/run_monthly.sh --from 3   # 3단계부터 (앞 단계 이미 했을 때)
#
# ⚠ 전제: daily_price / market_cap 이 Railway PG 에 최신 업로드되어 있어야 한다.
#         (이 스크립트 범위 밖. 0단계에서 확인만 한다)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$ROOT/.venv/bin/python"     # ★ HSMM 은 반드시 이 환경. numpy 버전이 다르면 exposure 가 달라진다.
cd "$ROOT"

[ -x "$PY" ] || { echo "✗ .venv 없음: $PY"; exit 1; }

ASSUME_YES=0; FROM=0
while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes)  ASSUME_YES=1; shift ;;
    --from)    FROM="$2"; shift 2 ;;
    *) echo "✗ 알 수 없는 인자: $1"; exit 1 ;;
  esac
done

echo "════════════════════════════════════════════════════════════"
echo "  월간 파이프라인  $(date '+%Y-%m-%d %H:%M')"
echo "  Python   : $PY"
echo "  전략     : CORE (메인전략)"
echo "  유니버스 : 시총 2,000억 이상 / rebal_type=monthly"
echo "  레짐     : HSMM exposure (현금 연 2.5%)"
echo "════════════════════════════════════════════════════════════"

if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "  계속할까요? [y/N] " ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "  중단."; exit 0 ;; esac
fi

step() { echo; echo "▶▶▶ $1"; echo "────────────────────────────────────────────────────────────"; }
skip() { [ "$FROM" -gt "$1" ]; }

# ── 0) 신선도 점검 ──────────────────────────────────────────
if ! skip 0; then
  # daily_price 는 별도 장비에서 Railway PG 로 수동 업로드하는 구조라
  # check_freshness.py(매크로 전용)가 못 잡는다. 여기서 먼저 막는다.
  # (2026-09-22 확인: 주가가 22일 밀려 있었음 → 10월 리밸 생성 불가)
  step "0-1) daily_price 지연 점검  ★ 밀려 있으면 이번 달 리밸을 만들 수 없다"
  "$PY" scripts/check_price_freshness.py

  step "0-2) 매크로 지표 신선도"
  "$PY" scripts/check_freshness.py || echo "  ! 매크로 경고 — 계속 진행하되 해석 주의"
fi

# ── 1) 데이터 최신화 ────────────────────────────────────────
if ! skip 1; then
  step "1-1) 매크로 지표"
  "$PY" scripts/collect_macro.py
  step "1-2) 글로벌 지수 종가"
  "$PY" scripts/backfill_global_indices.py
  step "1-3) technical_indicators (종가 의존 → 반드시 마지막)"
  "$PY" scripts/legacy/collect_technical.py
fi

# ── 2) prefetch 캐시 강제 갱신 + 검증 ───────────────────────
# 2026-09 사고: stale 검증이 daily_price 최신일만 보기 때문에 forward(컨센서스)가
# 결손이어도 "신선함"으로 판정됐다. 그 캐시로 돈 백테스트는 컨센 팩터가 전부 0점이었다.
if ! skip 2; then
  step "2) prefetch 캐시 강제 갱신"
  PREFETCH_FORCE_REFRESH=1 "$PY" - <<'PYEOF'
import sys, warnings
warnings.filterwarnings("ignore")
from lib.db import get_conn
import lib.factor_engine as fe
conn = get_conn()
fe.prefetch_all_data(conn)

# 검증: forward(컨센서스) 행수가 라이브 DB 와 일치하는가
from lib.db import read_sql
live = read_sql("SELECT trade_date, count(*) n FROM alpha_lab.fnspace_forward "
                "WHERE trade_date >= (CURRENT_DATE - 40) GROUP BY 1", conn)
conn.close()
cache = fe._prefetch_cache["forward"].groupby("trade_date").size()
bad = [(d, int(n), int(cache.get(d, 0))) for d, n in zip(live.trade_date.astype(str), live.n)
       if int(cache.get(d, 0)) != int(n)]
if bad:
    print("✗ prefetch forward 불일치 (날짜, DB, 캐시):", bad[:5], flush=True)
    sys.exit(1)
print(f"✓ prefetch 검증 통과 — forward 최근 {len(live)}일 일치", flush=True)
PYEOF
fi

# ── 3) HSMM 레짐 ────────────────────────────────────────────
# exposure[t] 는 t 월말 판정 → t→t+1 수익에 적용. 백테스트가 이 CSV 를 읽는다.
if ! skip 3; then
  step "3) HSMM 레짐 재계산 → analysis/hsmm_final_path.csv"
  "$PY" analysis/hsmm_final.py --refresh
  echo "  최근 exposure:"
  tail -4 analysis/hsmm_final_path.csv | awk -F',' '{printf "    %s  pbear %.3f  exposure %s  %s\n", $1, $4, $8, $10}'
fi

# ── 4) 파이프라인 ───────────────────────────────────────────
# --skip-regime-combo : 레짐조합 전략은 2026-09 폐지
if ! skip 4; then
  step "4) 마스터 + 재무 + TTM + 유니버스 + 백테스트"
  "$PY" scripts/run_pipeline.py --monthly --skip-regime-combo
fi

echo
echo "════════════════════════════════════════════════════════════"
echo "  ✅ 완료  $(date '+%Y-%m-%d %H:%M')"
echo "════════════════════════════════════════════════════════════"
echo
echo "  ⚠ 웹뷰 반영에 필요한 마지막 단계 — git push"
echo "     hsmm_final_path.csv 는 웹뷰 차트가 직접 읽는 파일이라"
echo "     커밋하지 않으면 화면에 지난달 exposure 가 계속 표시된다."
echo
if ! git diff --quiet analysis/hsmm_final_path.csv 2>/dev/null; then
  _ym=$(tail -1 analysis/hsmm_final_path.csv | cut -d, -f1)
  _ex=$(tail -1 analysis/hsmm_final_path.csv | cut -d, -f8)
  echo "     git add analysis/hsmm_final_path.csv"
  echo "     git commit -m \"chore(hsmm): ${_ym} 판정 반영 (exposure ${_ex})\""
  echo "     git push origin main"
else
  echo "     (hsmm_final_path.csv 변경 없음 — push 불필요)"
fi
echo
echo "  확인: 웹뷰 → 성과 비교 탭 상단 '레짐 익스포저' 차트"
echo "════════════════════════════════════════════════════════════"
