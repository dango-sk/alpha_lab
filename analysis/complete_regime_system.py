"""
analysis/complete_regime_system.py

완성된 양방향 레짐 상태기계:
  Bull→Bear (천장): 이중 트리거 = breadth combo>=0.6 OR breadth 급락(하위15%)
  Bear→Bull (바닥): newlow washout(상위40%) OR 가격>MA150 (예측+반응 backstop)
평가: 전환 포착, Bull/Bear 실제수익 격차, AI v2 대비, 주요 사건 타임라인.
출력: analysis/regime_complete_map.json (월별 Bull/Bear)
사용: .venv/bin/python analysis/complete_regime_system.py
"""
import os, json, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent


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
        rows.append({'ym': pd.Period(mends[i+1], freq='M').strftime('%Y-%m'),
                     'breadth': asof(breadth, e), 'br_chg': asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
                     'mom_decel': pct(kospi, e, 30) - pct(kospi, e, 180), 'newlow': asof(newlow, e),
                     'p_gt_ma': int(asof(kospi, e) > ks.iloc[-150:].mean()),
                     'ret': (kospi.loc[mends[i+1]] / kospi.loc[e] - 1) * 100})
    df = pd.DataFrame(rows).dropna().reset_index(drop=True); n = len(df)
    def ep(arr, t): h = arr[:t]; return (h < arr[t]).mean() if t > 0 else .5

    # 상태기계
    reg = []; state = 'Bull'
    for t in range(n):
        if t >= 12:
            top = np.mean([1 - ep(df['breadth'].values, t), ep(df['mom_decel'].values, t)])
            sudden = ep(df['br_chg'].values, t) <= 0.15
            washout = ep(df['newlow'].values, t) >= 0.60
            if state == 'Bull' and (top >= 0.6 or sudden): state = 'Bear'
            elif state == 'Bear' and (washout or df['p_gt_ma'].iloc[t] == 1): state = 'Bull'
        reg.append(state)
    df['regime'] = reg
    json.dump(dict(zip(df['ym'], df['regime'])), open(A / "regime_complete_map.json", 'w'), ensure_ascii=False)

    ev = df.iloc[12:]
    nb = (ev['regime'] == 'Bear').sum()
    print(f"\n평가 {len(ev)}개월, Bear {nb} ({nb/len(ev)*100:.0f}%)")
    bull_r = ev[ev['regime'] == 'Bull']['ret'].mean(); bear_r = ev[ev['regime'] == 'Bear']['ret'].mean()
    print(f"\n=== Bull/Bear 실제 다음달 수익 격차 ===")
    print(f"  Bull: {bull_r:+.2f}%/월  ({(ev['regime']=='Bull').sum()}개월)")
    print(f"  Bear: {bear_r:+.2f}%/월  ({nb}개월)")
    print(f"  격차: {bull_r-bear_r:+.2f}%p  (AI v2는 0.07%p, 클수록 레짐이 유익)")

    # AI v2 비교
    aij = A / "regime_agent_multimodel_results_gemini.json"
    if aij.exists():
        prev = 'Bull'; ai = {}
        for r in sorted(json.load(open(aij)), key=lambda x: x['as_of']):
            j = r.get('judgment'); ai[r['as_of'][:7]] = 'Bear' if j == '약세' else ('Bull' if j == '강세' else prev); prev = ai[r['as_of'][:7]]
        ov = ev[ev['ym'].isin(ai)].copy(); ov['ai'] = ov['ym'].map(ai)
        ab = ov[ov['ai'] == 'Bull']['ret'].mean(); abr = ov[ov['ai'] == 'Bear']['ret'].mean()
        print(f"\n  [AI v2 동일구간] Bull {ab:+.2f}% vs Bear {abr:+.2f}% → 격차 {ab-abr:+.2f}%p")

    print("\n=== 주요 사건 타임라인 (레짐 전환) ===")
    prev = None
    for _, r in df.iterrows():
        if r['regime'] != prev:
            print(f"  {r['ym']}: → {r['regime']}  (그달수익 {r['ret']:+.1f}%, breadth {r['breadth']:.2f})")
            prev = r['regime']
    print("\n저장: analysis/regime_complete_map.json")


if __name__ == '__main__':
    main()
