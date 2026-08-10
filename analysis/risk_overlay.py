"""
analysis/risk_overlay.py
Risk overlay 시뮬: 강세전략(항상 보유)에 Bear월만 익스포저 축소(현금≈0%). 백테스트 미수정·포트폴리오 레벨.
강세전략(=all-Bull) 월수익 1회 추출 → hsmm 4/5 Bear월에 e 곱 → 누적/CAGR/Sharpe/MDD/Calmar.
비교: e=1.0(=bull_only) / 0.7 / 0.5 / 0.3 / 0.0. (참고: 전략교체 combo 329~353%, ai_v2 406%/MDD-40.6%)
사용: DATABASE_URL=<ip> .venv/bin/python analysis/risk_overlay.py
"""
import os, json, sys, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"
OV = "2018-04"; CASH = 0.0  # Bear 축소분은 현금 0% 가정(보수적)


def monthly_map(c):
    rd = c.get("rebalance_dates") or []; mr = c.get("monthly_returns") or []
    out = {}
    for i, dt in enumerate(rd):
        if i >= len(mr): break
        ym = str(dt)[:7]; v = mr[i]
        out[ym] = float(v) if not isinstance(v, dict) else float(v.get("return", v.get("ret", 0)))
    return out


def metrics(rets):  # rets: dict ym->monthly return(소수)
    yms = sorted(rets); r = np.array([rets[y] for y in yms])
    cum = np.prod(1 + r) - 1
    yrs = len(r) / 12
    cagr = (1 + cum) ** (1 / yrs) - 1
    sharpe = (r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else float('nan')
    eq = np.cumprod(1 + r); mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    cal = cagr / abs(mdd) if mdd else float('nan')
    return cum, cagr, sharpe, mdd, cal


def main():
    from lib.data import run_regime_combo_backtest
    # 강세전략 항상 보유 (all-Bull map)
    slot = A / "regime_rf_map.json"; bak = A / "regime_rf_map.json.ovbak"
    if slot.exists() and not bak.exists(): bak.write_text(slot.read_text())
    bc = json.load(open(A / "hsmm_fx_bearcount.json"))
    allbull = {y: "Bull" for y in bc}
    slot.write_text(json.dumps(allbull, ensure_ascii=False))
    print("[FCF] 강세전략(all-Bull) 월수익 추출...", flush=True)
    c = (run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="rf") or {}).get("REGIME_COMBO", {})
    if bak.exists(): slot.write_text(bak.read_text())
    bull_ret = monthly_map(c)
    bull_ret = {y: v for y, v in bull_ret.items() if y >= OV}
    (A / "bull_only_monthly_ret.json").write_text(json.dumps(bull_ret, ensure_ascii=False))  # 재사용 저장
    print(f"  강세전략 월수익 {len(bull_ret)}개월 ({min(bull_ret)}~{max(bull_ret)}) → 저장", flush=True)

    # hsmm 4/5 Bear월
    yms = sorted(bull_ret)
    reg = {y: ("Bear" if bc.get(y, 0) >= 4 else "Bull") for y in yms}
    nbear = sum(1 for y in yms if reg[y] == "Bear")
    ntrans = sum(1 for i in range(1, len(yms)) if reg[yms[i]] != reg[yms[i-1]])
    print(f"  hsmm 4/5 Bear월 {nbear}/{len(yms)}, Bull↔Bear 전환 {ntrans}회\n")

    def overlay(e, slip):
        # Bear월: e 노출. 전환월: (1-e)만큼 매매 → slip(편도) 비용 차감. (강세전략 자체 리밸비용은 bull_ret에 이미 포함)
        out = {}
        for i, y in enumerate(yms):
            r = bull_ret[y] if reg[y] == "Bull" else e * bull_ret[y] + (1 - e) * CASH
            if i > 0 and reg[y] != reg[yms[i-1]]:
                r -= (1 - e) * slip   # 익스포저 변경 거래비용
            out[y] = r
        return out

    print(f"{'='*86}\n  Risk Overlay (강세전략 항상 보유 + Bear월 익스포저 e; 전환 거래비용 반영)\n{'='*86}")
    print(f"  {'e':>4}  {'slip(편도)':>10}{'누적수익':>9}{'CAGR':>7}{'Sharpe':>7}{'MDD':>8}{'Calmar':>7}")
    for e in [1.0, 0.7, 0.5, 0.3, 0.0]:
        for slip in [0.0, 0.0015, 0.0030]:
            if e == 1.0 and slip > 0: continue  # e=1은 전환매매 없음
            cum, cagr, sh, md, cal = metrics(overlay(e, slip))
            lab = "gross" if slip == 0 else f"{slip*100:.2f}%"
            tag = "  ← overlay X(bull_only)" if e == 1.0 else ("  ← Bear 전액현금" if e == 0.0 and slip == 0.0030 else "")
            print(f"  {e:>4.1f}  {lab:>10}{cum*100:>8.0f}%{cagr*100:>6.1f}%{sh:>7.2f}{md*100:>7.1f}%{cal:>7.2f}{tag}")
    print(f"\n  참고: 전략교체 combo(hsmm) 329~353%/MDD-38% · ai_v2 406%/0.82/MDD-40.6%")
    print("  비용모델: 강세전략 리밸비용은 bull_ret에 내재. overlay는 Bear 진입/탈출 시 (1-e)×slip(편도) 추가 차감.")
    print("  목표: bull_only 대비 MDD↓ + 누적/Sharpe 유지(거래비용 net 기준)하며 ai_v2 근접하는 e 탐색.")


if __name__ == "__main__":
    main()
