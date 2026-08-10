"""
analysis/transition_predict_bear.py

모델 B: 현재 *약세장*(P<MA150)에서 "향후 2개월 내 MA150 상향돌파(상승장 복귀)" 예측.
모델 A(강세→전환)의 대칭. breadth 캐시(breadth_monthly.csv) 재사용.
feature: breadth, breadth 변화, mom_decel, sox_rs, riskoff.
no-fit 조합(breadth+mom_decel 높을수록 회복) + 최소 LR. walk-forward OOS.
사용: .venv/bin/python analysis/transition_predict_bear.py
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
A = Path(__file__).parent


def macd(conn, ind):
    d = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators "
                    "WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
    d['dt'] = pd.to_datetime(d['dt']); return d.set_index('dt')['v'].sort_index()


def main():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    breadth = pd.read_csv(A / "breadth_monthly.csv", parse_dates=['dt']).set_index('dt')['breadth']
    kospi = macd(conn, 'kospi'); sox = macd(conn, 'sox'); dxy = macd(conn, 'dxy'); us10y = macd(conn, 'us10y')
    conn.close()
    ym = pd.PeriodIndex(kospi.index, freq='M'); last = {}
    for d, p in zip(kospi.index, ym): last[p] = d
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]

    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, d): c = asof(s, e); p = s[s.index <= e - timedelta(days=d)]; return (c/p.iloc[-1]-1) if len(p) and p.iloc[-1] else np.nan
    def chg(s, e, d): c = asof(s, e); p = s[s.index <= e - timedelta(days=d)]; return (c-p.iloc[-1]) if len(p) else np.nan

    rows = []
    for i in range(len(mends) - 1):
        e = mends[i]; ks = kospi[kospi.index <= e]
        if len(ks) < 150 or i + 2 >= len(mends): continue
        if asof(kospi, e) >= ks.iloc[-150:].mean(): continue   # 약세장(P<MA150)만
        recover = int(any(kospi.loc[mends[i+j]] > kospi[kospi.index <= mends[i+j]].iloc[-150:].mean() for j in [1, 2]))
        rows.append({
            'breadth': asof(breadth, e), 'breadth_chg': asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
            'mom_decel': pct(kospi, e, 30) - pct(kospi, e, 180),
            'sox_rs': pct(sox, e, 60) - pct(kospi, e, 60),
            'riskoff': pct(dxy, e, 30) - chg(us10y, e, 30) / 100,
            'recover': recover,
        })
    df = pd.DataFrame(rows).dropna().reset_index(drop=True); n = len(df); y = df['recover'].values
    print(f"약세장 {n}개월, 회복 {y.sum()}건 (base rate {y.mean()*100:.0f}%)\n")
    if n < 30:
        print("표본 너무 작음 — 참고용");
    feats = ['breadth', 'breadth_chg', 'mom_decel', 'sox_rs', 'riskoff']

    # no-fit: breadth 높을수록 + mom_decel 높을수록 회복
    def epct(arr, t): h = arr[:t]; return (h < arr[t]).mean()
    score = np.full(n, np.nan)
    for t in range(20, n):
        score[t] = np.mean([epct(df['breadth'].values, t), epct(df['mom_decel'].values, t)])
    m = ~np.isnan(score)
    if m.sum() > 15 and len(set(y[m])) > 1:
        print(f"① no-fit(breadth+mom_decel↑→회복)  OOS AUC {roc_auc_score(y[m], score[m]):.3f}  (평가 {m.sum()})")
        d2 = df[m].copy(); d2['sc'] = score[m]; d2['t'] = pd.qcut(d2['sc'], 2, labels=['저(회복약)', '고(회복강)'], duplicates='drop')
        for t in d2['t'].cat.categories:
            sub = d2[d2['t'] == t]; print(f"     {t}: 회복 {sub['recover'].mean()*100:.0f}% ({sub['recover'].sum()}/{len(sub)})")

    # 최소 LR
    X = df[feats].values; p = np.full(n, np.nan)
    for i in range(24, n):
        ytr = y[:i]
        if ytr.sum() < 5 or len(ytr) - ytr.sum() < 5: continue
        sc = StandardScaler().fit(X[:i]); lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000)
        lr.fit(sc.transform(X[:i]), ytr); p[i] = lr.predict_proba(sc.transform(X[i:i+1]))[0, 1]
    m2 = ~np.isnan(p)
    if m2.sum() > 15 and len(set(y[m2])) > 1:
        auc = roc_auc_score(y[m2], p[m2]); pred = (p[m2] >= 0.5).astype(int); yy = y[m2]
        tp = ((pred==1)&(yy==1)).sum(); fp = ((pred==1)&(yy==0)).sum(); fn = ((pred==0)&(yy==1)).sum()
        print(f"\n② 최소 LR  OOS AUC {auc:.3f}  Recall {tp/max(tp+fn,1)*100:.0f}%  Precision {tp/max(tp+fp,1)*100:.0f}%  (평가 {m2.sum()})")
    print("\n주의: 약세장 표본이 강세장보다 적음 → 결과 해석 신중히.")


if __name__ == '__main__':
    main()
