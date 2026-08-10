# -*- coding: utf-8 -*-
"""analysis/sample_pbear.py — pbear 계산 샘플 덤프 (피처 → pbear → 비중)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd
import hsmm_final as H

df, yms, n, ret, rvol, dvol, dd6 = H.build_features()
# 스트레스(z) 재현
TRz = H.roll_z(df, H.TRAN_COLS).values
stress = TRz[:, 0] - TRz[:, 1]
pbear_raw, start = H.walk_forward(df, yms, n)
idx = list(range(start, n)); pb = pbear_raw.copy()
for t in idx[1:]:
    pb[t] = H.PBEAR_EMA * pbear_raw[t] + (1 - H.PBEAR_EMA) * pb[t - 1]
exp = np.clip(1 - pb, H.EXP_FLOOR, 1.0)

out = pd.DataFrame({
    "월": [yms[t] for t in idx],
    "breadth": [round(df["breadth"].values[t], 2) for t in idx],
    "신저가": [round(df["newlow"].values[t], 2) for t in idx],
    "추세": [round(df["trend"].values[t], 3) for t in idx],
    "스트레스z": [round(stress[t], 2) for t in idx],
    "pbear_raw": [round(pbear_raw[t], 2) for t in idx],
    "pbear(최종)": [round(pb[t], 2) for t in idx],
    "주식비중": [round(exp[t], 2) for t in idx],
})
out.to_csv(Path(__file__).parent / "sample_pbear.csv", index=False, encoding="utf-8-sig")

# 대표 구간 샘플 출력
print(out.to_string(index=False))
