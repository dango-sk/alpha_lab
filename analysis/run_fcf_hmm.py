"""
analysis/run_fcf_hmm.py
HMM(breadth/newlow emission) 레짐 맵을 rf 슬롯에 넣고 FCF로 vs AI v2.
"""
import json, shutil, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"

src = A / "regime_hmm_breadth_map.json"
rf = A / "regime_rf_map.json"
if rf.exists():
    shutil.copy(rf, A / "regime_rf_map.json.bak")
shutil.copy(src, rf)
m = json.load(open(rf))
b = sum(1 for v in m.values() if v == 'Bear')
print(f"HMM 맵 → rf: {len(m)}개월, Bull={len(m)-b}, Bear={b}", flush=True)

from lib.data import run_regime_combo_backtest
out = {}
for mode in ['rf', 'ai']:
    print(f"\n=== {mode} ===", flush=True)
    r = run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode=mode)
    out[mode] = (r or {}).get("REGIME_COMBO", {})
print(f"\n{'#'*56}\n  FCF: HMM(breadth/newlow) vs AI v2\n{'#'*56}")
for mode, c in out.items():
    if c:
        print(f"  {mode:4}: 누적={c.get('total_return'):.3f}  CAGR={c.get('cagr'):.3f}  Sharpe={c.get('sharpe'):.3f}  MDD={c.get('mdd'):.3f}")
print("  (참고: rule min-hold3 = 363%/0.785/-38.6%, AI v2 = 406%/0.823/-40.6%)")
