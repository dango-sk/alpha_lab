"""
analysis/run_fcf_final.py

최종 양방향 레짐(천장 이중트리거 + 바닥 newlow/가격 + min-hold=3) 맵 생성 →
regime_rf_map.json → FCF로 rf(최종) vs ai(AI v2) 비교.
사용: .venv/bin/python analysis/run_fcf_final.py
"""
import os, json, shutil, sys, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"
MIN_HOLD = 3


def build_map():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    breadth = pd.read_csv(A / "breadth_monthly.csv", parse_dates=['dt']).set_index('dt')['breadth']
    print("newlow 로드...", flush=True)
    dp = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float c FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    dp['dt'] = pd.to_datetime(dp['dt']); wide = dp.pivot_table(index='dt', columns='stock_code', values='c').sort_index()
    rm = wide.rolling(252, min_periods=120).min(); newlow = ((wide <= rm) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1)
    k = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators WHERE indicator='kospi' AND freq='D' ORDER BY period", conn)
    conn.close()
    k['dt'] = pd.to_datetime(k['dt']); kospi = k.set_index('dt')['v'].sort_index()
    ym = pd.PeriodIndex(kospi.index, freq='M'); last = {}
    for d, p in zip(kospi.index, ym): last[p] = d
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, d): c = asof(s, e); p = s[s.index <= e - timedelta(days=d)]; return (c/p.iloc[-1]-1) if len(p) and p.iloc[-1] else np.nan
    rows = []
    for i in range(len(mends) - 1):
        e = mends[i]; ks = kospi[kospi.index <= e]
        if len(ks) < 150: continue
        rows.append({'ym': pd.Period(mends[i+1], freq='M').strftime('%Y-%m'), 'breadth': asof(breadth, e),
                     'br_chg': asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
                     'mom_decel': pct(kospi, e, 30) - pct(kospi, e, 180), 'newlow': asof(newlow, e),
                     'p_gt_ma': int(asof(kospi, e) > ks.iloc[-150:].mean())})
    df = pd.DataFrame(rows).dropna().reset_index(drop=True); n = len(df)
    def ep(arr, t): h = arr[:t]; return (h < arr[t]).mean() if t > 0 else .5
    g = np.zeros(n, bool); sd = np.zeros(n, bool); bt = np.zeros(n, bool)
    for t in range(12, n):
        g[t] = np.mean([1 - ep(df['breadth'].values, t), ep(df['mom_decel'].values, t)]) >= 0.6
        sd[t] = ep(df['br_chg'].values, t) <= 0.15
        bt[t] = (ep(df['newlow'].values, t) >= 0.6) or (df['p_gt_ma'].iloc[t] == 1)
    state = 'Bull'; held = 99; m = {}
    for t in range(n):
        sw = False
        if t >= 13 and held >= MIN_HOLD:
            if state == 'Bull' and (g[t] or sd[t]): state = 'Bear'; sw = True
            elif state == 'Bear' and bt[t]: state = 'Bull'; sw = True
        m[df['ym'].iloc[t]] = state; held = 0 if sw else held + 1
    b = sum(1 for v in m.values() if v == 'Bear')
    print(f"최종 맵: {len(m)}개월, Bull={len(m)-b}, Bear={b}", flush=True)
    rf = A / "regime_rf_map.json"
    if rf.exists(): shutil.copy(rf, A / "regime_rf_map.json.bak")
    rf.write_text(json.dumps(m, ensure_ascii=False))


def main():
    build_map()
    from lib.data import run_regime_combo_backtest
    out = {}
    for mode in ['rf', 'ai']:
        print(f"\n=== {mode} ===", flush=True)
        r = run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode=mode)
        out[mode] = (r or {}).get("REGIME_COMBO", {})
    print(f"\n{'#'*56}\n  FCF: 최종 레짐(min-hold=3) vs AI v2\n{'#'*56}")
    for mode, c in out.items():
        if c:
            print(f"  {mode:4}: 누적={c.get('total_return'):.3f}  CAGR={c.get('cagr'):.3f}  Sharpe={c.get('sharpe'):.3f}  MDD={c.get('mdd'):.3f}")


if __name__ == '__main__':
    main()
