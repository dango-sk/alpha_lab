"""
analysis/hsmm_vs_aiv2_attribution.py
hsmm 4/5 vs ai_v2 월별 수익 attribution — "왜 ai_v2보다 53%p 낮은가".
두 백테스트는 동일 bull/bear 하위전략 → 차이는 월별 레짐 라벨에서만 발생.
출력:
 1) 월별: hsmm_reg/ai_reg/hsmm_ret/ai_ret/diff + 누적gap
 2) 레짐쌍별 attribution (특히 hsmm=Bear & ai=Bull = hsmm이 방어하다 놓친 상승)
 3) gap 기여 큰 disagreement 월 Top
 4) hsmm 손절 발생 월 → ai_v2 레짐 교차표 + 손절 건수(레짐쌍별)
사용: DATABASE_URL=<ip> .venv/bin/python analysis/hsmm_vs_aiv2_attribution.py
"""
import os, io, re, json, sys, warnings, contextlib
import numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"
STOP_RE = re.compile(r"\[손절\]\s+(\S+)\s+\|\s+([\d-]+)\s+\|")


def run_capture(mode):
    from lib.data import run_regime_combo_backtest
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode=mode)
    c = (r or {}).get("REGIME_COMBO", {})
    stops = {}
    for m in STOP_RE.finditer(buf.getvalue()):
        ym = m.group(2)[:7]; stops[ym] = stops.get(ym, 0) + 1
    return c, stops


def monthly_map(c):
    rd = c.get("rebalance_dates") or []; mr = c.get("monthly_returns") or []
    out = {}
    for i, d in enumerate(rd):
        if i >= len(mr): break
        ym = str(d)[:7]; v = mr[i]
        out[ym] = float(v) if not isinstance(v, dict) else float(v.get("return", v.get("ret", 0)))
    return out


def main():
    # 레짐 맵
    bc = json.load(open(A / "hsmm_fx_bearcount.json"))
    hmap = {y: ("Bear" if v >= 4 else "Bull") for y, v in bc.items()}
    ers = {r["as_of"][:7]: r.get("expected_return", 0) for r in json.load(open(A / "regime_agent_results.json"))}
    amap = {}; prev = "Bull"
    for y in sorted(ers):
        er = ers[y]; cur = ("Bear" if er <= -2 else "Bull") if prev == "Bull" else ("Bull" if er >= 1 else "Bear"); amap[y] = cur; prev = cur

    # hsmm 4/5 백테스트
    slot = A / "regime_rf_map.json"; bak = A / "regime_rf_map.json.atbak"
    if slot.exists() and not bak.exists(): bak.write_text(slot.read_text())
    slot.write_text(json.dumps(hmap, ensure_ascii=False))
    print("[1/2] hsmm 4/5 백테스트...", flush=True)
    ch, sh = run_capture("rf")
    if bak.exists(): slot.write_text(bak.read_text())
    print("[2/2] ai_v2 백테스트...", flush=True)
    ca, sa = run_capture("ai")

    mh, ma = monthly_map(ch), monthly_map(ca)
    yms = sorted(set(mh) & set(ma))
    rows = []
    for y in yms:
        rh, ra = mh[y], ma[y]
        rows.append(dict(ym=y, hreg=hmap.get(y, "Bull"), areg=amap.get(y, "Bull"),
                         hret=rh*100, aret=ra*100, gap=(rh-ra)*100,
                         hstop=sh.get(y, 0), astop=sa.get(y, 0)))
    df = pd.DataFrame(rows)
    df["cum_h"] = (1 + df.hret/100).cumprod(); df["cum_a"] = (1 + df.aret/100).cumprod()
    df.to_csv(A / "hsmm_vs_aiv2_attribution.csv", index=False)  # 먼저 저장
    print(f"\n공통 {len(df)}개월. 누적: hsmm {df.cum_h.iloc[-1]*100-100:+.0f}% / ai_v2 {df.cum_a.iloc[-1]*100-100:+.0f}%")

    print(f"\n=== 레짐쌍별 attribution (월 gap 합, hsmm−ai) ===")
    print(f"  {'hsmm/ai':14}{'개월':>5}{'Σgap(%)':>10}{'평균gap':>9}{'hsmm손절':>9}")
    for (hr, ar), g in df.groupby(["hreg", "areg"]):
        print(f"  {hr+'/'+ar:14}{len(g):>5}{g['gap'].sum():>9.1f}{g['gap'].mean():>+9.2f}{g['hstop'].sum():>9}")
    dis = df[df.hreg != df.areg]
    print(f"\n  레짐 불일치 {len(dis)}개월, Σgap {dis['gap'].sum():+.1f}%p | 일치 {len(df)-len(dis)}개월 Σgap {df[df.hreg==df.areg]['gap'].sum():+.1f}%p")
    hb_al = df[(df.hreg=="Bear") & (df.areg=="Bull")]
    print(f"  ★ hsmm=Bear & ai=Bull: {len(hb_al)}개월, Σgap {hb_al['gap'].sum():+.1f}%p (hsmm 방어하다 놓친 상승), hsmm손절 {hb_al['hstop'].sum()}건")

    print(f"\n=== gap 워스트 월 Top10 (gap 작은 순=hsmm 손해) ===")
    print(f"  {'월':9}{'hsmm':>5}{'ai':>5}{'hret%':>8}{'aret%':>8}{'gap':>7}{'hstop':>6}")
    for r in df.nsmallest(10, "gap").itertuples():
        print(f"  {r.ym:9}{r.hreg:>5}{r.areg:>5}{r.hret:>+7.1f}{r.aret:>+7.1f}{r.gap:>+7.1f}{r.hstop:>6}")

    print(f"\n=== hsmm 손절 발생 월 → ai_v2 레짐 (#1) ===")
    sm = df[df.hstop > 0]
    print(f"  hsmm 손절 발생 {len(sm)}개월 / 총손절 {sm.hstop.sum()}건")
    for ar, g in sm.groupby("areg"):
        print(f"    그 달 ai_v2={ar}: {len(g)}개월, 손절 {g.hstop.sum()}건, 그달 평균 hsmm수익 {g.hret.mean():+.2f}% vs ai {g.aret.mean():+.2f}%")
    df.to_csv(A / "hsmm_vs_aiv2_attribution.csv", index=False)
    print("\n  상세: analysis/hsmm_vs_aiv2_attribution.csv")
    print("  주의: 보유 carryover로 일치월에도 약간 diff 가능(경로의존). 근본원인은 불일치월.")


if __name__ == "__main__":
    main()
