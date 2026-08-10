"""
analysis/fcf_us10y_check.py
+US10Y 분별력 개선(+1.53→+2.41)이 FCF 성과로 이어지는지 검증.
비교: base(+FX) / +US10Y / ai_v2. (consensus ≥3/5 저장맵 사용)
지표: 누적수익·CAGR·Sharpe·MDD·Calmar·turnover·Whipsaw·Bear월.
사용: DATABASE_URL=<ip> .venv/bin/python analysis/fcf_us10y_check.py
"""
import os, json, sys, warnings
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"
OV = "2018-04"


def stats(mp, yms_sorted):
    ov = [y for y in yms_sorted if y >= OV]
    seq = [mp[y] for y in ov]
    whip = sum(1 for k in range(1, len(seq)) if seq[k] != seq[k-1])
    bearm = sum(1 for x in seq if x == "Bear")
    return whip, bearm


def main():
    from lib.data import run_regime_combo_backtest
    base = json.load(open(A / "regime_feat_base_FX.json"))
    us10 = json.load(open(A / "regime_feat__US10Y.json"))
    # ai_v2 맵 재구성 (whip/bear월용)
    ers = {r["as_of"][:7]: r.get("expected_return", 0) for r in json.load(open(A / "regime_agent_results.json"))}
    amap = {}; prev = "Bull"
    for y in sorted(ers):
        er = ers[y]; cur = ("Bear" if er <= -2 else "Bull") if prev == "Bull" else ("Bull" if er >= 1 else "Bear"); amap[y] = cur; prev = cur

    slot = A / "regime_rf_map.json"; bak = A / "regime_rf_map.json.u10bak"
    if slot.exists() and not bak.exists(): bak.write_text(slot.read_text())
    res = {}
    for name, mp in [("base(+FX)", base), ("+US10Y", us10)]:
        slot.write_text(json.dumps(mp, ensure_ascii=False))
        print(f"[FCF] {name} ...", flush=True)
        c = (run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="rf") or {}).get("REGIME_COMBO", {})
        res[name] = (c, stats(mp, sorted(mp)))
    if bak.exists(): slot.write_text(bak.read_text())
    print("[FCF] ai_v2 ...", flush=True)
    c = (run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="ai") or {}).get("REGIME_COMBO", {})
    res["ai_v2"] = (c, stats(amap, sorted(amap)))

    print(f"\n{'='*92}\n  +US10Y FCF 검증 (overlap 2018~2026)\n{'='*92}")
    print(f"  {'모델':12}{'누적수익':>9}{'CAGR':>7}{'Sharpe':>7}{'MDD':>8}{'Calmar':>7}{'turnover':>9}{'Whip':>6}{'Bear월':>6}")
    for name, (c, st) in res.items():
        if not c: print(f"  {name:12}(no result)"); continue
        tr, cg, md, sh, to = c.get("total_return"), c.get("cagr"), c.get("mdd"), c.get("sharpe"), c.get("avg_turnover")
        cal = (cg/abs(md)) if (cg is not None and md) else float('nan'); whip, bearm = st
        print(f"  {name:12}{tr*100:>8.0f}%{cg*100:>6.1f}%{sh:>7.2f}{md*100:>7.1f}%{cal:>7.2f}{(to or 0):>9.2f}{whip:>6}{bearm:>6}")
    print("\n  핵심: +US10Y가 base 대비 누적/Sharpe/Calmar↑ & ai_v2(406%/MDD-40.6%)와 격차 줄면 채택.")


if __name__ == "__main__":
    main()
