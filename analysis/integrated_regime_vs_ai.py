"""
analysis/integrated_regime_vs_ai.py

양방향 전환 상태기계 레짐 vs AI v2 비교.
- 천장 신호: breadth 낮음 + mom_decel 높음 (deterioration)
- 바닥 신호: newlow 비율 높음 (washout)
- 상태기계: 강세→(천장신호 강) 약세 / 약세→(바닥신호 강) 강세
지표: 다음달 방향 정확도 + 레짐별 실제 다음달 수익. AI v2(gemini judgment)와 동일구간 비교.
사용: .venv/bin/python analysis/integrated_regime_vs_ai.py
"""
import os, json, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path('.env'))
A = Path(__file__).parent


def main():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    breadth = pd.read_csv(A / "breadth_monthly.csv", parse_dates=['dt']).set_index('dt')['breadth']
    print("개별종목 로드(newlow)...", flush=True)
    dp = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float c FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    dp['dt'] = pd.to_datetime(dp['dt']); wide = dp.pivot_table(index='dt', columns='stock_code', values='c').sort_index()
    rollmin = wide.rolling(252, min_periods=120).min()
    newlow = ((wide <= rollmin) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1)

    def macd(ind):
        d = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators "
                        "WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
        d['dt'] = pd.to_datetime(d['dt']); return d.set_index('dt')['v'].sort_index()
    kospi = macd('kospi'); conn.close()

    ym = pd.PeriodIndex(kospi.index, freq='M'); last = {}
    for d, p in zip(kospi.index, ym): last[p] = d
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]

    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, d): c = asof(s, e); p = s[s.index <= e - timedelta(days=d)]; return (c/p.iloc[-1]-1) if len(p) and p.iloc[-1] else np.nan

    # 월별 신호 수집
    recs = []
    for i in range(len(mends) - 1):
        e = mends[i]
        recs.append({'ym': pd.Period(mends[i+1], freq='M').strftime('%Y-%m'),
                     'breadth': asof(breadth, e), 'mom_decel': pct(kospi, e, 30) - pct(kospi, e, 180),
                     'newlow': asof(newlow, e),
                     'ret': (kospi.loc[mends[i+1]] / kospi.loc[e] - 1) * 100})
    df = pd.DataFrame(recs).dropna().reset_index(drop=True); n = len(df)

    # expanding 분위수 신호
    def ep(arr, t): h = arr[:t]; return (h < arr[t]).mean() if t > 0 else 0.5
    top_sc = np.full(n, np.nan); bot_sc = np.full(n, np.nan)
    for t in range(12, n):
        top_sc[t] = np.mean([1 - ep(df['breadth'].values, t), ep(df['mom_decel'].values, t)])  # 높=천장위험
        bot_sc[t] = ep(df['newlow'].values, t)                                                  # 높=washout=바닥

    # 상태기계 (강세 시작, 천장신호>0.6 약세전환 / 바닥신호>0.6 강세복귀)
    reg = []; state = 'Bull'
    for t in range(n):
        if not np.isnan(top_sc[t]):
            if state == 'Bull' and top_sc[t] >= 0.6: state = 'Bear'
            elif state == 'Bear' and bot_sc[t] >= 0.6: state = 'Bull'
        reg.append(state)
    df['regime'] = reg

    # AI v2
    aij = A / "regime_agent_multimodel_results_gemini.json"
    ai = {}
    if aij.exists():
        prev = 'Bull'
        for r in sorted(json.load(open(aij)), key=lambda x: x['as_of']):
            j = r.get('judgment'); cur = 'Bear' if j == '약세' else ('Bull' if j == '강세' else prev)
            ai[r['as_of'][:7]] = cur; prev = cur
    df['ai'] = df['ym'].map(ai)

    # 평가: 신호 안정구간(첫 24개월 제외)
    ev = df.iloc[24:].dropna(subset=['ai']).copy()
    print(f"\n비교 구간 {len(ev)}개월 ({ev['ym'].iloc[0]}~{ev['ym'].iloc[-1]})\n")

    def report(name, col):
        up = ev['ret'] > 0
        pred_up = ev[col] == 'Bull'
        acc = (pred_up == up).mean()
        bull_ret = ev[ev[col] == 'Bull']['ret'].mean(); bear_ret = ev[ev[col] == 'Bear']['ret'].mean()
        nb = (ev[col] == 'Bear').sum()
        print(f"  {name:14} 방향정확도 {acc*100:.0f}%  | Bull수익 {bull_ret:+.2f}% vs Bear수익 {bear_ret:+.2f}%  (Bear {nb}/{len(ev)}개월)")

    print("== 레짐 예측 비교 (우리 양방향 vs AI v2) ==")
    report("우리(천장+바닥)", 'regime')
    report("AI v2", 'ai')
    base = max((ev['ret'] > 0).mean(), (ev['ret'] <= 0).mean())
    print(f"\n  (베이스라인 '항상 강세' 정확도 {base*100:.0f}%)")
    print("  판정: 방향정확도는 둘 다 베이스라인 근처일 것. 핵심은 Bull-Bear 수익 *격차*가 큰가.")


if __name__ == '__main__':
    main()
