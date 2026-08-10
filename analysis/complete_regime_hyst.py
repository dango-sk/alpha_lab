"""
analysis/complete_regime_hyst.py

양방향 레짐 + 히스테리시스(안정화):
  Bull→Bear: 급락(즉시) OR 점진천장 2개월연속 확인 (+ 최소2개월 보유 후)
  Bear→Bull: 바닥신호(washout|가격>MA) 2개월연속 확인 (+ 최소2개월 보유)
whipsaw(전환수) ↓ + 격차(정보량) 보존 목표.
사용: .venv/bin/python analysis/complete_regime_hyst.py
"""
import os, json, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
MIN_HOLD = 2


def main():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    breadth = pd.read_csv(A / "breadth_monthly.csv", parse_dates=['dt']).set_index('dt')['breadth']
    print("개별종목 로드(newlow)...", flush=True)
    dp = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float c FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    dp['dt'] = pd.to_datetime(dp['dt']); wide = dp.pivot_table(index='dt', columns='stock_code', values='c').sort_index()
    rollmin = wide.rolling(252, min_periods=120).min()
    newlow = ((wide <= rollmin) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1)
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
                     'p_gt_ma': int(asof(kospi, e) > ks.iloc[-150:].mean()),
                     'ret': (kospi.loc[mends[i+1]] / kospi.loc[e] - 1) * 100})
    df = pd.DataFrame(rows).dropna().reset_index(drop=True); n = len(df)
    def ep(arr, t): h = arr[:t]; return (h < arr[t]).mean() if t > 0 else .5
    gradual = np.zeros(n, bool); sudden = np.zeros(n, bool); botsig = np.zeros(n, bool)
    for t in range(12, n):
        gradual[t] = np.mean([1 - ep(df['breadth'].values, t), ep(df['mom_decel'].values, t)]) >= 0.6
        sudden[t] = ep(df['br_chg'].values, t) <= 0.15
        botsig[t] = (ep(df['newlow'].values, t) >= 0.60) or (df['p_gt_ma'].iloc[t] == 1)

    # 히스테리시스 상태기계
    reg = []; state = 'Bull'; held = 99
    for t in range(n):
        sw = False
        if t >= 13:
            if state == 'Bull':
                if sudden[t]:                                              # 급락 즉시
                    state = 'Bear'; sw = True
                elif gradual[t] and gradual[t-1] and held >= MIN_HOLD:      # 점진 2개월 확인
                    state = 'Bear'; sw = True
            else:
                if botsig[t] and botsig[t-1] and held >= MIN_HOLD:         # 바닥 2개월 확인
                    state = 'Bull'; sw = True
        reg.append(state); held = 0 if sw else held + 1
    df['regime'] = reg
    json.dump(dict(zip(df['ym'], df['regime'])), open(A / "regime_hyst_map.json", 'w'), ensure_ascii=False)

    ev = df.iloc[13:]
    trans = (ev['regime'].values[1:] != ev['regime'].values[:-1]).sum()
    nb = (ev['regime'] == 'Bear').sum()
    bull_r = ev[ev['regime'] == 'Bull']['ret'].mean(); bear_r = ev[ev['regime'] == 'Bear']['ret'].mean()
    print(f"\n평가 {len(ev)}개월, Bear {nb}({nb/len(ev)*100:.0f}%), 전환 {trans}회 (히스테리시스 전 ~47회)")
    print(f"  Bull {bull_r:+.2f}%/월 vs Bear {bear_r:+.2f}%/월 → 격차 {bull_r-bear_r:+.2f}%p (AI v2 0.36%p)")
    print("\n=== 레짐 전환 타임라인 ===")
    prev = None
    for _, r in df.iloc[13:].iterrows():
        if r['regime'] != prev:
            print(f"  {r['ym']}: → {r['regime']}  (그달 {r['ret']:+.1f}%, breadth {r['breadth']:.2f})")
            prev = r['regime']
    print("\n저장: analysis/regime_hyst_map.json")


if __name__ == '__main__':
    main()
