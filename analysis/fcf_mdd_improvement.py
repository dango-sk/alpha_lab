"""
analysis/fcf_mdd_improvement.py
MDD 개선 효과 측정: hsmm+환율(consensus) / ai_v2 / bull_only(레짐X) FCF 비교.
레짐 onset 방어전환이 낙폭을 얼마나 줄였나 = bull_only MDD - 모델 MDD.
사용: .venv/bin/python analysis/fcf_mdd_improvement.py   (FCF 3회, ~20분)
"""
import os, json, sys, warnings
from pathlib import Path
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"


def main():
    from lib.data import run_regime_combo_backtest
    cons = json.load(open(A / "regime_hsmm_fx_consensus.json"))
    bull_only = {y: "Bull" for y in cons}
    slot = A / "regime_rf_map.json"; bak = A / "regime_rf_map.json.mddbak"
    if slot.exists() and not bak.exists(): bak.write_text(slot.read_text())

    def run_rf(mp):
        slot.write_text(json.dumps(mp, ensure_ascii=False))
        return (run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="rf") or {}).get("REGIME_COMBO", {})

    print("[FCF] hsmm+환율 ...", flush=True); r_h = run_rf(cons)
    print("[FCF] bull_only(레짐X) ...", flush=True); r_b = run_rf(bull_only)
    if bak.exists(): slot.write_text(bak.read_text())  # 원복
    print("[FCF] ai_v2 ...", flush=True)
    r_a = (run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="ai") or {}).get("REGIME_COMBO", {})

    print(f"\n{'='*70}\n  MDD 개선 효과 (FCF, overlap 2018~2026)\n{'='*70}")
    print(f"  {'모델':16} {'누적수익':>9} {'CAGR':>7} {'Sharpe':>7} {'MDD':>8} {'vs레짐X MDD개선':>13}")
    base_mdd = r_b.get("mdd")
    for name, c in [("bull_only(레짐X)", r_b), ("hsmm+환율", r_h), ("ai_v2", r_a)]:
        if not c: print(f"  {name:16} (no result)"); continue
        tr, cg, md, sh = c.get("total_return"), c.get("cagr"), c.get("mdd"), c.get("sharpe")
        imp = (base_mdd - md) * 100 if (base_mdd is not None and md is not None) else None
        imps = f"{imp:>+11.1f}%p" if (imp is not None and name != "bull_only(레짐X)") else ("   (기준)" if name == "bull_only(레짐X)" else "")
        print(f"  {name:16} {tr*100:>8.0f}% {cg*100:>6.1f}% {sh:>7.2f} {md*100:>7.1f}% {imps:>13}")
    print("\n  MDD개선 = 레짐X(항상 Bull) 대비 낙폭 감소(+면 개선). 수익(CAGR) 희생 없이 MDD 줄면 성공.")


if __name__ == "__main__":
    main()
