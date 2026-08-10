"""
analysis/transition_predict_model.py

추세 전환 *예측* 모델: 현재 강세장(P>MA150)인 달에서 "향후 2개월 내 MA150 하향돌파(전환)"를 예측.
feature: breadth, breadth 변화, mom_decel, SOX 상대강도, riskoff.
모델: ① no-fit 조합(breadth+mom_decel, 경제적 방향 고정, 과적합 0) ② 최소 LR.
walk-forward OOS. base rate(27%) 대비 정확도/recall/precision.

입력: Railway PG (daily_price, macro). breadth는 analysis/breadth_monthly.csv 캐시.
사용: .venv/bin/python analysis/transition_predict_model.py
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
BR_CACHE = A / "breadth_monthly.csv"


def macd(conn, ind):
    d = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators "
                    "WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
    d['dt'] = pd.to_datetime(d['dt']); return d.set_index('dt')['v'].sort_index()


def get_breadth_daily(conn):
    if BR_CACHE.exists():
        s = pd.read_csv(BR_CACHE, parse_dates=['dt']).set_index('dt')['breadth']
        print(f"  breadth 캐시 로드 ({len(s)})", flush=True); return s
    print("  개별종목 로드+breadth 계산 (수십초)...", flush=True)
    dp = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float c FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    dp['dt'] = pd.to_datetime(dp['dt']); wide = dp.pivot_table(index='dt', columns='stock_code', values='c').sort_index()
    ma = wide.rolling(120, min_periods=60).mean(); valid = wide.notna() & ma.notna()
    br = ((wide > ma) & valid).sum(axis=1) / valid.sum(axis=1).clip(lower=1)
    br = br[valid.sum(axis=1) > 200]
    br.rename('breadth').reset_index().to_csv(BR_CACHE, index=False)
    return br


def main():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    breadth = get_breadth_daily(conn)
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
        if asof(kospi, e) <= ks.iloc[-150:].mean(): continue   # 강세장(P>MA150)만
        broke = int(any(kospi.loc[mends[i+j]] < kospi[kospi.index <= mends[i+j]].iloc[-150:].mean() for j in [1, 2]))
        rows.append({
            'breadth': asof(breadth, e), 'breadth_chg': asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
            'mom_decel': pct(kospi, e, 30) - pct(kospi, e, 180),
            'sox_rs': pct(sox, e, 60) - pct(kospi, e, 60),
            'riskoff': pct(dxy, e, 30) - chg(us10y, e, 30) / 100,
            'broke': broke,
        })
    df = pd.DataFrame(rows).dropna().reset_index(drop=True); n = len(df)
    print(f"\n강세장 {n}개월, 전환 {df['broke'].sum()}건 (base rate {df['broke'].mean()*100:.0f}%)\n")

    feats = ['breadth', 'breadth_chg', 'mom_decel', 'sox_rs', 'riskoff']
    y = df['broke'].values

    # ① no-fit 조합: breadth 낮을수록 + mom_decel 높을수록 전환 (경제방향 고정), expanding 분위수
    def epct(arr, t, invert=False):
        h = arr[:t]; p = (h < arr[t]).mean(); return (1 - p) if invert else p
    score = np.full(n, np.nan)
    for t in range(24, n):
        s_br = epct(df['breadth'].values, t, invert=True)   # 낮을수록 bearish
        s_md = epct(df['mom_decel'].values, t)              # 높을수록 bearish
        score[t] = np.mean([s_br, s_md])
    m = ~np.isnan(score)
    auc_nf = roc_auc_score(y[m], score[m])
    print(f"① no-fit 조합(breadth+mom_decel)  OOS AUC {auc_nf:.3f}  (평가 {m.sum()}개월)")
    # 고/저 score 전환율
    d2 = df[m].copy(); d2['sc'] = score[m]; d2['t'] = pd.qcut(d2['sc'], 2, labels=['저위험', '고위험'])
    for t in ['고위험', '저위험']:
        sub = d2[d2['t'] == t]; print(f"     {t}: 전환 {sub['broke'].mean()*100:.0f}% ({sub['broke'].sum()}/{len(sub)})")

    # ② 최소 LR (walk-forward)
    X = df[feats].values; p = np.full(n, np.nan)
    for i in range(30, n):
        ytr = y[:i]
        if ytr.sum() < 5 or len(ytr) - ytr.sum() < 5: continue
        sc = StandardScaler().fit(X[:i]); lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000)
        lr.fit(sc.transform(X[:i]), ytr); p[i] = lr.predict_proba(sc.transform(X[i:i+1]))[0, 1]
    m2 = ~np.isnan(p); auc_lr = roc_auc_score(y[m2], p[m2]) if len(set(y[m2])) > 1 else float('nan')
    pred = (p[m2] >= 0.5).astype(int); yy = y[m2]
    tp = ((pred==1)&(yy==1)).sum(); fp = ((pred==1)&(yy==0)).sum(); fn = ((pred==0)&(yy==1)).sum()
    print(f"\n② 최소 LR(5feat)  OOS AUC {auc_lr:.3f}  Recall {tp/max(tp+fn,1)*100:.0f}%  Precision {tp/max(tp+fp,1)*100:.0f}%  (평가 {m2.sum()})")
    print("\n판정: AUC가 0.65↑ & 고위험 전환율 >> 저위험 이면 → 전환 예측 모델 성립.")


if __name__ == '__main__':
    main()
