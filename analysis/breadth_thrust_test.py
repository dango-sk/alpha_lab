"""
analysis/breadth_thrust_test.py

바닥(약세장 회복) 예측에 *breadth thrust* (상승종목비율 급등) + 신규저점 소진 시도.
지금까지 안 쓴 교과서적 바닥 신호. 약세장(P<MA150)에서 향후 2개월 회복 예측.

신호:
  adv10      : 상승종목비율 10일평균 (높을수록 thrust=회복)
  thrust     : adv10 - adv10 20일전 (급등 정도)
  newlow_pct : 52주 신저가 종목 비율 (높을수록 항복)
  newlow_chg : 신저가 비율 변화 (감소=소진=회복)
no-fit 조합 + LR, walk-forward OOS.
사용: .venv/bin/python analysis/breadth_thrust_test.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); load_dotenv(Path('.env'))


def main():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    print("개별종목 로드+thrust 계산...", flush=True)
    dp = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float c FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    dp['dt'] = pd.to_datetime(dp['dt']); wide = dp.pivot_table(index='dt', columns='stock_code', values='c').sort_index()
    ret = wide.pct_change()
    valid = wide.notna() & ret.notna()
    adv = ((ret > 0) & valid).sum(axis=1) / valid.sum(axis=1).clip(lower=1)   # 상승종목비율
    adv = adv[valid.sum(axis=1) > 200]
    adv10 = adv.rolling(10).mean()
    # 52주 신저가 비율
    roll_min = wide.rolling(252, min_periods=120).min()
    newlow = ((wide <= roll_min) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1)
    newlow = newlow.reindex(adv.index)

    def macd(ind):
        d = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators "
                        "WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
        d['dt'] = pd.to_datetime(d['dt']); return d.set_index('dt')['v'].sort_index()
    kospi = macd('kospi'); conn.close()

    ym = pd.PeriodIndex(kospi.index, freq='M'); last = {}
    for d, p in zip(kospi.index, ym): last[p] = d
    mends = [last[p] for p in sorted(last) if last[p] >= adv.index.min()]

    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan

    rows = []
    for i in range(len(mends) - 1):
        e = mends[i]; ks = kospi[kospi.index <= e]
        if len(ks) < 150 or i + 2 >= len(mends): continue
        if asof(kospi, e) >= ks.iloc[-150:].mean(): continue   # 약세장만
        a10 = asof(adv10, e); a10_prev = asof(adv10, e - timedelta(days=20))
        nl = asof(newlow, e); nl_prev = asof(newlow, e - timedelta(days=20))
        rec = int(any(kospi.loc[mends[i+j]] > kospi[kospi.index <= mends[i+j]].iloc[-150:].mean() for j in [1, 2]))
        rows.append({'adv10': a10, 'thrust': a10 - a10_prev, 'newlow': nl, 'newlow_chg': nl - nl_prev, 'rec': rec})
    df = pd.DataFrame(rows).dropna().reset_index(drop=True); n = len(df); y = df['rec'].values
    print(f"\n약세장 {n}개월, 회복 {y.sum()}건 (base {y.mean()*100:.0f}%)\n")

    def ep(arr, t): h = arr[:t]; return (h < arr[t]).mean()
    # no-fit: thrust↑·adv10↑·newlow_chg↓(소진) → 회복
    sc = np.full(n, np.nan)
    for t in range(16, n):
        sc[t] = np.mean([ep(df['adv10'].values, t), ep(df['thrust'].values, t), 1 - ep(df['newlow_chg'].values, t)])
    m = ~np.isnan(sc)
    if m.sum() > 12 and len(set(y[m])) > 1:
        print(f"① no-fit thrust 조합  OOS AUC {roc_auc_score(y[m], sc[m]):.3f}  (평가 {m.sum()})")
        d2 = df[m].copy(); d2['s'] = sc[m]; d2['t'] = pd.qcut(d2['s'], 2, labels=['저thrust', '고thrust'], duplicates='drop')
        for t in d2['t'].cat.categories:
            sub = d2[d2['t'] == t]; print(f"   {t}: 회복 {sub['rec'].mean()*100:.0f}% ({sub['rec'].sum()}/{len(sub)})")
    # 개별 신호 상관 (참고)
    print("\n개별 신호 AUC (회복 예측):")
    for c in ['adv10', 'thrust', 'newlow', 'newlow_chg']:
        sc1 = np.full(n, np.nan)
        for t in range(16, n): sc1[t] = ep(df[c].values, t)
        mm = ~np.isnan(sc1)
        if mm.sum() > 12 and len(set(y[mm])) > 1:
            print(f"   {c:12} AUC {roc_auc_score(y[mm], sc1[mm]):.3f}")
    # LR
    X = df[['adv10', 'thrust', 'newlow', 'newlow_chg']].values; p = np.full(n, np.nan)
    for i in range(18, n):
        ytr = y[:i]
        if ytr.sum() < 4 or len(ytr) - ytr.sum() < 4: continue
        s = StandardScaler().fit(X[:i]); lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000)
        lr.fit(s.transform(X[:i]), ytr); p[i] = lr.predict_proba(s.transform(X[i:i+1]))[0, 1]
    m2 = ~np.isnan(p)
    if m2.sum() > 10 and len(set(y[m2])) > 1:
        print(f"\n② LR  OOS AUC {roc_auc_score(y[m2], p[m2]):.3f}  (평가 {m2.sum()})")


if __name__ == '__main__':
    main()
