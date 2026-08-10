"""
analysis/hsmm_vs_aiv2_fcf.py
best 모델(Full-Cov HSMM) vs AI v2 만 FCF 비교 (2회).
hmm_fullcov_stabilize 의 헬퍼 재사용. hsmm 맵만 walk-forward 생성.
사용: .venv/bin/python analysis/hsmm_vs_aiv2_fcf.py
"""
import os, json, sys, warnings
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
from analysis.hmm_fullcov_stabilize import (build_panel, bear_state, nb_logpmf_fn,
                                            hsmm_last_state, ai_v2_map, evalr, MIN_TRAIN, SEED)
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"


def gen_hsmm(F, n):
    reg = ["Bull"] * n
    for t in range(MIN_TRAIN, n):
        try:
            Z = StandardScaler().fit_transform(F[:t])
            hm = GaussianHMM(2, "full", n_iter=60, random_state=SEED); hm.fit(Z)
            b = bear_state(hm.means_)
            logB = hm._compute_log_likelihood(Z)
            logstart = np.log(np.clip(hm.startprob_, 1e-12, 1))
            path = hm.predict(Z); durs = {0: [], 1: []}; rs, rl = path[0], 1
            for k in range(1, len(path)):
                if path[k] == rs: rl += 1
                else: durs[rs].append(rl); rs, rl = path[k], 1
            durs[rs].append(rl)
            logD = {i: nb_logpmf_fn(durs[i]) for i in range(2)}
            reg[t] = "Bear" if hsmm_last_state(logB, logstart, logD, b) == b else "Bull"
        except Exception:
            reg[t] = reg[t - 1]
    return reg


def main():
    from lib.data import run_regime_combo_backtest
    F, yms, n, ret, dd6 = build_panel()
    print(f"패널 {n}개월. hsmm 맵 생성...", flush=True)
    hsmm = gen_hsmm(F, n); aiv = ai_v2_map(yms)
    ov = [i for i in range(n) if yms[i] >= "2018-04"]
    rc, ne, ld, fa, wh, bm, br, be, gp = evalr(hsmm, yms, n, ret, dd6, ov)
    print(f"  hsmm(overlap)  Recall {rc}/{ne} Lead{ld:+.1f} FA{fa} Whip{wh} 격차{gp:+.2f}p", flush=True)
    rc2, ne2, ld2, fa2, wh2, bm2, br2, be2, gp2 = evalr(aiv, yms, n, ret, dd6, ov)
    print(f"  ai_v2(overlap) Recall {rc2}/{ne2} Lead{ld2:+.1f} FA{fa2} Whip{wh2} 격차{gp2:+.2f}p", flush=True)

    slot = A / "regime_rf_map.json"; bak = A / "regime_rf_map.json.stbak"
    if slot.exists() and not bak.exists(): bak.write_text(slot.read_text())
    slot.write_text(json.dumps(dict(zip(yms, hsmm)), ensure_ascii=False))
    print("\n[FCF] hsmm 실행...", flush=True)
    rh = (run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="rf") or {}).get("REGIME_COMBO", {})
    if bak.exists(): slot.write_text(bak.read_text())
    print("[FCF] ai_v2 실행...", flush=True)
    ra = (run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="ai") or {}).get("REGIME_COMBO", {})

    print(f"\n{'='*54}\n  {'모델':10} {'CAGR':>7} {'Sharpe':>7} {'MDD':>7} {'Calmar':>7}")
    for name, c in [("hsmm", rh), ("ai_v2", ra)]:
        if not c: print(f"  {name:10} (no result)"); continue
        cg, md, sh = c.get("cagr"), c.get("mdd"), c.get("sharpe")
        cal = (cg / abs(md)) if (cg is not None and md) else float("nan")
        print(f"  {name:10} {cg*100:>6.1f}% {sh:>7.2f} {md*100:>6.1f}% {cal:>7.2f}")


if __name__ == "__main__":
    main()
