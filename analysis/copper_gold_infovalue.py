"""
analysis/copper_gold_infovalue.py
cg_chg(구리/금 3m변화)가 현재 HSMM 6feature에 없는 *독립 정보*인지 검증 (HMM fit 없음).
1) 상관(Pearson/Spearman) vs breadth/Δbreadth/newlow/Δnewlow/mom/fx3m
2) 독립성: cg~6feat OLS R² + Mutual Info + bear(fwd6m dd<=-15) 예측 증분 OOS AUC
3) 2018/2022 긴축베어서 cg_chg가 breadth/mom보다 먼저 risk-off로 꺾였는지 월별
사용: DATABASE_URL=<ip> .venv/bin/python analysis/copper_gold_infovalue.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2, yfinance as yf
from datetime import timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
A = Path(__file__).parent
COLS = ['breadth', 'Δbreadth', 'newlow', 'Δnewlow', 'mom', 'fx3m']
MIN_TRAIN = 36


def build():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    kk = pd.read_sql("SELECT left(period,10) p, value::float v FROM alpha_lab.macro_indicators WHERE indicator='kospi' AND freq='D'", conn)
    uu = pd.read_sql("SELECT left(period,10) p, value::float v FROM alpha_lab.macro_indicators WHERE indicator='usd_krw' AND freq='D'", conn); conn.close()
    kk['p'] = pd.to_datetime(kk['p']); kospi = kk.set_index('p')['v'].sort_index()
    uu['p'] = pd.to_datetime(uu['p']); usdkrw = uu.set_index('p')['v'].sort_index()
    d["dt"] = pd.to_datetime(d["dt"])
    # 구리/금 (yfinance)
    def ym(tk):
        x = yf.download(tk, start='1999-01-01', auto_adjust=False, progress=False)
        x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]; return x['Close']
    cop = ym('HG=F'); gld = ym('GC=F'); cg = (cop / gld).dropna()
    wide = pd.concat([h, d[["dt", "stock_code", "close"]]], ignore_index=True).drop_duplicates(["dt", "stock_code"]).pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    ma = wide.rolling(120, min_periods=60).mean(); val = wide.notna() & ma.notna()
    breadth = ((wide > ma) & val).sum(axis=1) / val.sum(axis=1).clip(lower=1); breadth = breadth[val.sum(axis=1) > 100]
    rmn = wide.rolling(252, min_periods=120).min(); newlow = ((wide <= rmn) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1); newlow = newlow.reindex(breadth.index)
    ym2 = pd.PeriodIndex(kospi.index, freq="M"); last = {}
    for dt_, p in zip(kospi.index, ym2): last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1)*100 if len(p) and p.iloc[-1] else np.nan
    yms = [pd.Period(e, 'M').strftime("%Y-%m") for e in mends]; n = len(mends)
    rows = []
    for e in mends:
        rows.append([asof(breadth, e)*100, (asof(breadth, e)-asof(breadth, e-timedelta(days=30)))*100,
                     asof(newlow, e)*100, (asof(newlow, e)-asof(newlow, e-timedelta(days=30)))*100,
                     pct(kospi, e, 30)-pct(kospi, e, 180), pct(usdkrw, e, 90), pct(cg, e, 90)])
    X = pd.DataFrame(rows, columns=COLS + ['cg_chg'], index=yms)
    Px = np.array([asof(kospi, e) for e in mends])
    dd6 = np.full(n, np.nan)
    for i in range(n-1):
        if i+6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path/path.cummax()-1).min()*100
    X['bear'] = (pd.Series(dd6, index=yms) <= -15).astype(float)
    return X


def main():
    X = build()
    V = X.dropna(subset=['cg_chg'] + COLS).copy()
    print(f"분석구간 {V.index.min()}~{V.index.max()} ({len(V)}개월, cg_chg 유효)\n")

    print("=== 1) cg_chg 상관 (Pearson / Spearman) ===")
    for c in COLS:
        p = V['cg_chg'].corr(V[c]); s = V['cg_chg'].corr(V[c], method='spearman')
        print(f"  cg_chg vs {c:10}: Pearson {p:+.2f}  Spearman {s:+.2f}")

    print("\n=== 2) 독립성 ===")
    sc = StandardScaler(); Z = sc.fit_transform(V[COLS]); y = sc.fit_transform(V[['cg_chg']]).ravel()
    r2 = LinearRegression().fit(Z, y).score(Z, y)
    print(f"  cg_chg ~ 6feature OLS R² = {r2:.3f}  → 독립정보 비중 {(1-r2)*100:.0f}% (R² 낮을수록 독립)")
    mi = mutual_info_regression(V[COLS], V['cg_chg'], random_state=42)
    print("  Mutual Info(cg_chg; feature):", ", ".join(f"{c} {m:.2f}" for c, m in zip(COLS, mi)))
    # 증분 OOS AUC (bear 예측, walk-forward logistic)
    def oos_auc(cols):
        Xa = V[cols].values; yb = V['bear'].values; m = ~np.isnan(yb); n = len(V); p = np.full(n, np.nan)
        for t in range(MIN_TRAIN, n):
            tr = [i for i in range(t) if m[i]]
            if len(set(yb[tr])) < 2: continue
            scl = StandardScaler().fit(Xa[tr]); clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000)
            clf.fit(scl.transform(Xa[tr]), yb[tr]); p[t] = clf.predict_proba(scl.transform(Xa[t:t+1]))[0, 1]
        mm = ~np.isnan(p) & m
        return roc_auc_score(yb[mm], p[mm]) if len(set(yb[mm])) > 1 else float('nan')
    a0 = oos_auc(COLS); a1 = oos_auc(COLS + ['cg_chg'])
    print(f"  bear 예측 증분 OOS AUC: 6feature {a0:.3f} → +cg_chg {a1:.3f}  (Δ{a1-a0:+.3f})")

    print("\n=== 3) 2018/2022 긴축베어 — cg_chg가 먼저 꺾였나 (월별) ===")
    for lo, hi, name in [("2017-10", "2018-10", "2018 긴축"), ("2021-06", "2022-10", "2022 긴축")]:
        b = X[(X.index >= lo) & (X.index <= hi)]
        print(f"  [{name}] (cg_chg<0=구리/금 하락=risk-off; breadth↓·mom↓도 risk-off)")
        print(f"  {'월':9}{'cg_chg':>8}{'breadth':>8}{'Δbreadth':>9}{'mom':>7}{'fx3m':>7}")
        for ym_, r in b.iterrows():
            print(f"  {ym_:9}{r['cg_chg']:>+7.1f}{r['breadth']:>8.1f}{r['Δbreadth']:>+8.1f}{r['mom']:>+7.1f}{r['fx3m']:>+6.1f}")
        # 첫 risk-off 월
        def first(cond):
            s = b[cond(b)]; return s.index[0] if len(s) else "—"
        print(f"   첫 cg_chg<-3%: {first(lambda d: d['cg_chg']<-3)} | 첫 Δbreadth<0: {first(lambda d: d['Δbreadth']<0)} | 첫 mom<0: {first(lambda d: d['mom']<0)}\n")
    print("  판정: R² 낮고(독립) 증분 AUC↑면 새 정보. 2018/2022서 cg_chg 첫 risk-off가 breadth/mom보다 빠르면 선행성.")


if __name__ == "__main__":
    main()
