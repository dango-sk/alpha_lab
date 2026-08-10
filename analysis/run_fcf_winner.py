"""
analysis/run_fcf_winner.py
덜 방어적 변형: 천장 임계 0.7(강한천장만) + 급락트리거 + 바닥 + min-hold3.
여러 천장임계(0.65/0.70/0.75)를 한 번에 FCF로 비교 vs AI v2.
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


def precompute():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    breadth = pd.read_csv(A / "breadth_monthly.csv", parse_dates=['dt']).set_index('dt')['breadth']
    print("newlow 로드...", flush=True)
    dp = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float c FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    dp['dt'] = pd.to_datetime(dp['dt']); wide = dp.pivot_table(index='dt', columns='stock_code', values='c').sort_index()
    rmn = wide.rolling(252, min_periods=120).min(); newlow = ((wide <= rmn) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1)
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
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def build_map(df, top_th):
    n = len(df)
    def ep(arr, t): h = arr[:t]; return (h < arr[t]).mean() if t > 0 else .5
    g = np.zeros(n, bool); sd = np.zeros(n, bool); bt = np.zeros(n, bool)
    for t in range(12, n):
        g[t] = np.mean([1 - ep(df['breadth'].values, t), ep(df['mom_decel'].values, t)]) >= top_th
        sd[t] = ep(df['br_chg'].values, t) <= 0.15
        bt[t] = (ep(df['newlow'].values, t) >= 0.6) or (df['p_gt_ma'].iloc[t] == 1)
    state = 'Bull'; held = 99; m = {}
    for t in range(n):
        sw = False
        if t >= 13 and held >= MIN_HOLD:
            if state == 'Bull' and (g[t] or sd[t]): state = 'Bear'; sw = True
            elif state == 'Bear' and bt[t]: state = 'Bull'; sw = True
        m[df['ym'].iloc[t]] = state; held = 0 if sw else held + 1
    return m


def main():
    df = precompute()
    from lib.data import run_regime_combo_backtest
    res = {}
    for th in [0.65, 0.70, 0.75]:
        m = build_map(df, th)
        b = sum(1 for v in m.values() if v == 'Bear')
        (A / "regime_rf_map.json").write_text(json.dumps(m, ensure_ascii=False))
        print(f"\n=== top_th={th} (Bear {b}/{len(m)}) ===", flush=True)
        r = run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="rf")
        res[f"th{th}"] = (r or {}).get("REGIME_COMBO", {})
    r = run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="ai")
    res["ai"] = (r or {}).get("REGIME_COMBO", {})
    print(f"\n{'#'*56}\n  덜 방어적 변형 vs AI v2\n{'#'*56}")
    for nm, c in res.items():
        if c: print(f"  {nm:7}: 누적={c.get('total_return'):.3f}  CAGR={c.get('cagr'):.3f}  Sharpe={c.get('sharpe'):.3f}  MDD={c.get('mdd'):.3f}")


if __name__ == '__main__':
    main()
