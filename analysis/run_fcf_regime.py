"""
analysis/run_fcf_regime.py

FCF_YIELD 강세/약세 조합을 4개 레짐으로 비교:
  trend : 우리 추세맵(강세장→Bull, 변동성/약세→Bear)  [rf 슬롯]
  hmm   : HMM p_bear_2s>=0.5 → Bear (6개월 dd 0.72 예측)  [rf 슬롯]
  ma    : 기본 MA50 레짐 (built-in)
  ai    : AI v2 레짐 (built-in)

trend/hmm은 regime_rf_map.json을 갈아끼워 regime_mode="rf"로 돌림.
사용: .venv/bin/python analysis/run_fcf_regime.py
"""
import sys, json, shutil
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.data import run_regime_combo_backtest

A = Path(__file__).parent
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"
RF_MAP = A / "regime_rf_map.json"


def build_hmm_map(thresh=0.5):
    h = pd.read_csv(A / "hmm_regime_features.csv")
    m = {r["pred_month"]: ("Bear" if r["p_bear_2s"] >= thresh else "Bull")
         for _, r in h.iterrows() if pd.notna(r.get("p_bear_2s"))}
    (A / "regime_hmm_map.json").write_text(json.dumps(m, ensure_ascii=False))
    b = sum(1 for v in m.values() if v == "Bear")
    print(f"[HMM map] p>={thresh}: {len(m)}개월, Bear={b}, Bull={len(m)-b}", flush=True)
    return A / "regime_hmm_map.json"


def metrics(res):
    if not res:
        return {}, []
    cand = {"누적": ["total_return", "cumulative_return", "cum_return", "total_ret"],
            "CAGR": ["cagr", "annual_return", "annualized_return"],
            "Sharpe": ["sharpe", "sharpe_ratio"],
            "MDD": ["mdd", "max_drawdown", "max_dd"]}
    out = {}
    for lab, keys in cand.items():
        for k in keys:
            if k in res:
                out[lab] = res[k]; break
    return out, list(res.keys())


def run_rf_with(map_path, label):
    shutil.copy(map_path, RF_MAP)
    print(f"\n{'='*60}\n  {label} (regime_mode=rf, map={map_path.name})\n{'='*60}", flush=True)
    r = run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI",
                                  rebal_type="monthly", regime_mode="rf")
    return (r or {}).get("REGIME_COMBO", {})


def run_builtin(mode):
    print(f"\n{'='*60}\n  {mode} (regime_mode={mode})\n{'='*60}", flush=True)
    r = run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI",
                                  rebal_type="monthly", regime_mode=mode)
    return (r or {}).get("REGIME_COMBO", {})


def main():
    hmm_path = build_hmm_map(0.5)
    trend_path = A / "regime_trend_raw.json"

    res = {}
    try:
        res["trend"] = run_rf_with(trend_path, "trend(추세맵)")
    except Exception as e:
        print(f"trend 실패: {e}", flush=True); res["trend"] = {}
    try:
        res["hmm"] = run_rf_with(hmm_path, "HMM(0.72 dd예측)")
    except Exception as e:
        print(f"hmm 실패: {e}", flush=True); res["hmm"] = {}
    for mode in ["ma", "ai"]:
        try:
            res[mode] = run_builtin(mode)
        except Exception as e:
            print(f"{mode} 실패: {e}", flush=True); res[mode] = {}

    shutil.copy(trend_path, RF_MAP)  # 기본 상태 trend로 복원

    print(f"\n\n{'#'*60}\n  비교 (bull={BULL} / bear={BEAR})\n{'#'*60}")
    fk = None
    for mode, combo in res.items():
        if not combo:
            print(f"  {mode:6}: 결과 없음"); continue
        m, keys = metrics(combo)
        if fk is None:
            fk = keys; print(f"  (결과 키: {keys})\n")
        print(f"  {mode:6}: " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in m.items()))


if __name__ == "__main__":
    main()
