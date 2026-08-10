"""
analysis/hmm_breadth_regime.py

우리 발견(breadth·newlow·mom_decel)을 *Gaussian HMM* emission으로 → 레짐 학습.
rule-based 상태기계 대신 HMM이 상태·전이확률을 학습 (전이행렬=자연 히스테리시스).

emission: [breadth, mom_decel, newlow] (표준화), 2-state walk-forward.
약세 state = breadth 평균 최저 state. P(약세) → 레짐.
평가: Bull/Bear 수익격차 + 전환수 + AI v2 비교. 맵 저장.
사용: .venv/bin/python analysis/hmm_breadth_regime.py
"""
import os, json, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
MIN_TRAIN = 36
SEED = 42


def main():
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
        e = mends[i]
        rows.append({'ym': pd.Period(mends[i+1], freq='M').strftime('%Y-%m'),
                     'breadth': asof(breadth, e), 'mom_decel': pct(kospi, e, 30) - pct(kospi, e, 180),
                     'newlow': asof(newlow, e), 'ret': (kospi.loc[mends[i+1]] / kospi.loc[e] - 1) * 100})
    df = pd.DataFrame(rows).dropna().reset_index(drop=True); n = len(df)
    feats = ['breadth', 'mom_decel', 'newlow']
    X = df[feats].values

    # walk-forward HMM
    pbear = np.full(n, np.nan)
    for t in range(MIN_TRAIN, n):
        Xtr = X[:t]
        try:
            sc = StandardScaler().fit(Xtr); Z = sc.transform(Xtr)
            hm = GaussianHMM(n_components=2, covariance_type='diag', n_iter=50, random_state=SEED)
            hm.fit(Z)
            post = hm.predict_proba(Z)
            bear = int(np.argmin(hm.means_[:, 0]))   # breadth 평균 최저 = 약세
            pbear[t] = post[-1, bear]
        except Exception:
            continue
    df['pbear'] = pbear
    df['regime'] = np.where(df['pbear'] >= 0.5, 'Bear', 'Bull')
    ev = df.dropna(subset=['pbear'])
    json.dump(dict(zip(df.dropna(subset=['pbear'])['ym'], df.dropna(subset=['pbear'])['regime'])),
              open(A / "regime_hmm_breadth_map.json", 'w'), ensure_ascii=False)

    tr = (ev['regime'].values[1:] != ev['regime'].values[:-1]).sum()
    nb = (ev['regime'] == 'Bear').sum()
    br = ev[ev['regime'] == 'Bull']['ret'].mean(); ber = ev[ev['regime'] == 'Bear']['ret'].mean()
    print(f"\nHMM(breadth/newlow/mom) 평가 {len(ev)}개월, Bear {nb}({nb/len(ev)*100:.0f}%), 전환 {tr}회")
    print(f"  Bull {br:+.2f}%/월 vs Bear {ber:+.2f}%/월 → 격차 {br-ber:+.2f}%p")
    print(f"  (참고: rule상태기계 min-hold3 = 격차 +0.96%p / 전환 16회, AI v2 0.36%p)")

    # AI v2 비교
    aij = A / "regime_agent_multimodel_results_gemini.json"
    if aij.exists():
        prev = 'Bull'; ai = {}
        for r in sorted(json.load(open(aij)), key=lambda x: x['as_of']):
            j = r.get('judgment'); ai[r['as_of'][:7]] = 'Bear' if j == '약세' else ('Bull' if j == '강세' else prev); prev = ai[r['as_of'][:7]]
        ov = ev[ev['ym'].isin(ai)].copy(); ov['ai'] = ov['ym'].map(ai)
        ab = ov[ov['ai'] == 'Bull']['ret'].mean(); abr = ov[ov['ai'] == 'Bear']['ret'].mean()
        print(f"  [AI v2 동일구간] 격차 {ab-abr:+.2f}%p")

    print("\n=== 레짐 전환 타임라인 ===")
    prev = None
    for _, r in ev.iterrows():
        if r['regime'] != prev:
            print(f"  {r['ym']}: → {r['regime']}  (그달 {r['ret']:+.1f}%, breadth {r['breadth']:.2f}, P약세 {r['pbear']:.2f})")
            prev = r['regime']
    print("\n저장: regime_hmm_breadth_map.json")


if __name__ == '__main__':
    main()
