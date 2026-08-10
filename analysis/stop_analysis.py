"""
analysis/stop_analysis.py
hsmm 4/5 레짐에서 trailing stop 손절 종목의 *이후* 추이 분석.
CAGR 손실 원인 진단: (1)레짐신호 (2)trailing stop (3)모델이 놓친 특성.
손절 후 월말수익률·분포·MAE(최대추가하락)/MFE(최대반등)·적절vs너무빠른 손절 비중. Bear vs Bull 비교.
사용: DATABASE_URL=<ip> .venv/bin/python analysis/stop_analysis.py
"""
import os, io, re, json, sys, warnings, contextlib
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"
WIN_TD = 21  # 손절 후 MAE/MFE 측정 거래일 (~1개월)
STOP_RE = re.compile(r"\[손절\]\s+(\S+)\s+\|\s+([\d-]+)\s+\|\s+최초편입가=([\d,]+)\s+고점=([\d,]+)\s+현재=([\d,]+)\s+고점대비=(-?[\d.]+)%")


def main():
    from lib.data import run_regime_combo_backtest
    # 4/5 고확신 Bear 맵
    bc = json.load(open(A / "hsmm_fx_bearcount.json"))
    mp = {y: ("Bear" if c >= 4 else "Bull") for y, c in bc.items()}
    slot = A / "regime_rf_map.json"; bak = A / "regime_rf_map.json.sabak"
    if slot.exists() and not bak.exists(): bak.write_text(slot.read_text())
    slot.write_text(json.dumps(mp, ensure_ascii=False))

    print("[1/3] hsmm 4/5 백테스트 실행 + 손절로그 캡처...", flush=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="rf")
    if bak.exists(): slot.write_text(bak.read_text())
    log = buf.getvalue()
    stops = []
    for m in STOP_RE.finditer(log):
        code, dt, entry, peak, cur, dd = m.groups()
        stops.append(dict(code=code, dt=pd.Timestamp(dt), stop_px=float(cur.replace(",", "")),
                          entry=float(entry.replace(",", "")), regime=mp.get(dt[:7], "Bull")))
    sdf = pd.DataFrame(stops)
    print(f"  손절 {len(sdf)}건 (Bear월 {sum(sdf.regime=='Bear')}, Bull월 {sum(sdf.regime=='Bull')})", flush=True)

    print("[2/3] 손절 종목 이후 가격 fetch...", flush=True)
    codes = tuple(sdf.code.unique())
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    px = pd.read_sql(f"SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE stock_code IN %s", conn, params=(codes,))
    conn.close()
    px['dt'] = pd.to_datetime(px['dt'])
    bycode = {c: g.sort_values('dt') for c, g in px.groupby('stock_code')}

    print("[3/3] 이후 추이 계산...\n", flush=True)
    recs = []
    for r in sdf.itertuples():
        g = bycode.get(r.code)
        if g is None: continue
        fut = g[g.dt > r.dt]
        if fut.empty: continue
        # 월말까지
        me = fut[fut.dt.dt.to_period('M') == r.dt.to_period('M')]
        ret_me = (me.close.iloc[-1] / r.stop_px - 1) * 100 if len(me) else np.nan
        # 다음 WIN_TD 거래일 MAE/MFE
        w = fut.head(WIN_TD)
        mae = (w.close.min() / r.stop_px - 1) * 100   # 최대 추가하락(음수일수록 손절 잘함)
        mfe = (w.close.max() / r.stop_px - 1) * 100    # 최대 반등(클수록 너무 빨리 손절)
        ret_w = (w.close.iloc[-1] / r.stop_px - 1) * 100
        recs.append(dict(code=r.code, dt=r.dt, regime=r.regime, ret_me=ret_me, ret_w=ret_w, mae=mae, mfe=mfe))
    R = pd.DataFrame(recs)

    def summarize(df, label):
        if df.empty: print(f"  [{label}] 없음"); return
        n = len(df)
        bounce = (df.ret_w > 0).mean()      # 손절 후 (1개월) 반등 비율
        too_early = (df.mfe >= 10).mean()    # 손절 후 +10%↑ 반등 = 너무 빨랐음
        good = (df.mae <= -5).mean()         # 손절 후 -5%↓ 추가하락 = 적절
        print(f"\n  === [{label}] 손절 {n}건 ===")
        print(f"    손절후 월말 평균수익률: {df.ret_me.mean():+.2f}%  (중앙값 {df.ret_me.median():+.2f}%)")
        print(f"    손절후 1개월 평균수익률: {df.ret_w.mean():+.2f}%")
        print(f"    분포(1개월): p10 {df.ret_w.quantile(.1):+.1f}% / p50 {df.ret_w.median():+.1f}% / p90 {df.ret_w.quantile(.9):+.1f}%")
        print(f"    평균 MAE(최대추가하락) {df.mae.mean():+.1f}%  |  평균 MFE(최대반등) {df.mfe.mean():+.1f}%")
        print(f"    손절후 반등(>0) 비율 {bounce:.0%}  |  너무빠른손절(MFE>=+10%) {too_early:.0%}  |  적절(MAE<=-5%) {good:.0%}")

    summarize(R[R.regime == "Bear"], "Bear 구간")
    summarize(R[R.regime == "Bull"], "Bull 구간")
    summarize(R, "전체")
    # 너무 빨리 손절한 대표 사례 (Bear, MFE 큰 순)
    be = R[R.regime == "Bear"].nlargest(8, "mfe")
    print(f"\n  === Bear 구간 '너무 빨리 손절' Top8 (MFE 큰 순) ===")
    print(f"  {'종목':8}{'손절일':12}{'월말%':>8}{'1M%':>8}{'MAE%':>8}{'MFE%':>8}")
    for r in be.itertuples():
        print(f"  {r.code:8}{r.dt.strftime('%Y-%m-%d'):12}{r.ret_me:>+7.1f}{r.ret_w:>+8.1f}{r.mae:>+8.1f}{r.mfe:>+8.1f}")
    R.to_csv(A / "stop_analysis_detail.csv", index=False)
    print("\n  상세: analysis/stop_analysis_detail.csv")
    print("  판정 가이드: Bear 손절후 평균수익 +면 'stop 너무 빠름'(반등 먹음)=원인(2). −면 stop은 적절→원인(1)or(3).")


if __name__ == "__main__":
    main()
