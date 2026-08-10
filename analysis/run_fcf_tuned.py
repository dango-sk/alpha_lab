"""
analysis/run_fcf_tuned.py

"너무 방어적" 해결: 강한 천장신호(top_score>=0.75)일 때만 Bear, 식으면 바로 Bull.
→ Bear 월 축소 → 강세장 상승 보존. breadth 캐시 + mom_decel만(빠름).
FCF로 rf(튜닝) vs ai 비교.
사용: .venv/bin/python analysis/run_fcf_tuned.py
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
TOP_TH = 0.75   # 상위25% 강한 천장에서만 방어


def build_map():
    breadth = pd.read_csv(A / "breadth_monthly.csv", parse_dates=['dt']).set_index('dt')['breadth']
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
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
        e = mends[i]
        rows.append({'ym': pd.Period(mends[i+1], freq='M').strftime('%Y-%m'),
                     'breadth': asof(breadth, e), 'mom_decel': pct(kospi, e, 30) - pct(kospi, e, 180)})
    df = pd.DataFrame(rows).dropna().reset_index(drop=True); n = len(df)
    def ep(arr, t): h = arr[:t]; return (h < arr[t]).mean() if t > 0 else 0.5
    m = {}
    for t in range(n):
        if t >= 12:
            top = np.mean([1 - ep(df['breadth'].values, t), ep(df['mom_decel'].values, t)])
            m[df['ym'].iloc[t]] = 'Bear' if top >= TOP_TH else 'Bull'   # 메모리리스: 강한 천장만 방어
        else:
            m[df['ym'].iloc[t]] = 'Bull'
    b = sum(1 for v in m.values() if v == 'Bear')
    print(f"튜닝 맵(top>={TOP_TH}): {len(m)}개월, Bull={len(m)-b}, Bear={b}", flush=True)
    rf = A / "regime_rf_map.json"
    if rf.exists(): shutil.copy(rf, A / "regime_rf_map.json.bak")
    rf.write_text(json.dumps(m, ensure_ascii=False))


def main():
    build_map()
    from lib.data import run_regime_combo_backtest
    out = {}
    for mode in ['rf', 'ai']:
        print(f"\n=== regime_mode={mode} ===", flush=True)
        r = run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode=mode)
        out[mode] = (r or {}).get("REGIME_COMBO", {})
    print(f"\n{'#'*56}\n  FCF: 튜닝레짐(top>={TOP_TH}) vs AI v2\n{'#'*56}")
    for mode, c in out.items():
        if not c: print(f"  {mode}: 없음"); continue
        print(f"  {mode:4}: 누적={c.get('total_return'):.3f}  CAGR={c.get('cagr'):.3f}  "
              f"Sharpe={c.get('sharpe'):.3f}  MDD={c.get('mdd'):.3f}")


if __name__ == '__main__':
    main()
