"""
analysis/exposure_next_month.py  (실험/조회 스크립트, production 미수정)

다음 달(예: 2026-08)에 적용할 HSMM 오버레이 익스포저를 출력.
  익스포저(M월 적용) = M-1월말 시점에 확정된 pbear/변동성으로 계산.
  → '가장 최근 월말'(= 오늘 기준 7월말) 익스포저를 그대로 뽑아 8월에 적용.

세 버전 모두 출력:
  P = pbear만 (vol-타게팅 미적용)              ← 2차 보고서 권고안(우세)
  B = 60일 하방변동성 소프트 vol-타겟           ← 현재 hsmm_final production 코드
  A = 20일 실현변동성 소프트 vol-타겟           ← 박사님 스펙(참고)

사용: DATABASE_URL 세팅 후  .venv/bin/python analysis/exposure_next_month.py
"""
import sys
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")
import numpy as np
import hsmm_final as H


def main():
    df, yms, n, _ret, rvol, dvol, _dd6 = H.build_features()
    pbear_raw, start = H.walk_forward(df, yms, n)
    idx = list(range(start, n))
    pbear = pbear_raw.copy()
    for t in idx[1:]:
        pbear[t] = H.PBEAR_EMA * pbear_raw[t] + (1 - H.PBEAR_EMA) * pbear[t - 1]

    def build_exposure(volmeas):
        if volmeas is None:
            cut = np.zeros(n)
        else:
            cur = np.maximum(volmeas, H.VOL_FLOOR)
            tgt = np.full(n, np.nan); acc = []
            for t in idx:
                acc.append(cur[t]); tgt[t] = float(np.mean(acc))
            cut = 1.0 - np.minimum(1.0, np.nan_to_num(tgt, nan=1.0) / cur)
        vf = 1.0 - pbear * cut
        raw = np.clip((1 - pbear) * vf, H.EXP_FLOOR, 1.0)
        exp = raw.copy(); held = None
        for t in idx:
            if held is None or abs(raw[t] - held) >= H.REBAL_BAND:
                held = round(raw[t] / 0.05) * 0.05
            exp[t] = min(max(held, H.EXP_FLOOR), 1.0)
        return exp

    expP = build_exposure(None)
    expB = build_exposure(dvol)
    expA = build_exposure(rvol)

    last = idx[-1]
    ym = yms[last]
    print(f"\n{'='*60}")
    print(f"  가장 최근 월말 = {ym}  →  다음 달 적용 익스포저")
    print(f"{'='*60}")
    print(f"  pbear(raw)   = {pbear_raw[last]:.3f}")
    print(f"  pbear(EMA)   = {pbear[last]:.3f}")
    print(f"  EXP_FLOOR    = {H.EXP_FLOOR:.2f}")
    print(f"{'-'*60}")
    print(f"  P  pbear-only(권고)          = {expP[last]:.2f}")
    print(f"  B  60일 하방vol(production)   = {expB[last]:.2f}")
    print(f"  A  20일 실현vol(참고)         = {expA[last]:.2f}")
    print(f"{'='*60}")
    print("  최근 6개월 궤적 (pbear-only P / 하방B):")
    for t in idx[-6:]:
        print(f"    {yms[t]}  pbear {pbear[t]:.2f}   P {expP[t]:.2f}   B {expB[t]:.2f}")


if __name__ == "__main__":
    main()
