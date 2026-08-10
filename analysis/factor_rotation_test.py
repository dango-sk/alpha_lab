"""
analysis/factor_rotation_test.py

"예측력=PnL" 유일 설계: 매월 bear_key vs bull_key 중 *누가 이길지* 직접 예측.
타겟: sign(bear_ret - bull_ret). feature: breadth, mom_decel, kospi/spx 모멘텀.
맞히면 그 팩터 선택 → 회전 PnL. AUC↑ = PnL↑ (구조적 일치).
비교: 회전(예측) vs 항상bull / 항상bear / 완벽회전(상한).
사용: .venv/bin/python analysis/factor_rotation_test.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"


def standalone(key):
    from lib.data import run_regime_combo_backtest
    r = run_regime_combo_backtest(bull_key=key, bear_key=key, universe="KOSPI", rebal_type="monthly", regime_mode="ma")
    c = (r or {}).get("REGIME_COMBO", {})
    rets = c.get("monthly_returns", []); dates = c.get("rebalance_dates", [])
    return pd.Series(rets, index=[pd.Period(pd.Timestamp(d), freq="M").strftime("%Y-%m") for d in dates[:len(rets)]])


def main():
    print("bull/bear 단독 백테스트...", flush=True)
    bull = standalone(BULL); bear = standalone(BEAR)
    df = pd.DataFrame({"bull": bull, "bear": bear}).dropna()
    df["winner_bear"] = (df["bear"] > df["bull"]).astype(int)
    print(f"  공통 {len(df)}개월, bear승 {df['winner_bear'].sum()} ({df['winner_bear'].mean()*100:.0f}%)", flush=True)

    # feature
    breadth = pd.read_csv(A / "breadth_monthly.csv", parse_dates=['dt']).set_index('dt')['breadth']
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    def md(ind):
        d = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
        d['dt'] = pd.to_datetime(d['dt']); return d.set_index('dt')['v'].sort_index()
    kospi = md('kospi'); spx = md('sp500'); conn.close()
    ym = pd.PeriodIndex(kospi.index, freq='M'); last = {}
    for d, p in zip(kospi.index, ym): last[p] = d
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1) if len(p) and p.iloc[-1] else np.nan
    feat = {}
    for m in df.index:
        per = pd.Period(m, freq='M'); e = last.get(per - 1)  # 직전월말 기준 (lookahead 방지)
        if e is None: continue
        feat[m] = {'breadth': asof(breadth, e), 'mom_decel': pct(kospi, e, 30) - pct(kospi, e, 180),
                   'kospi_3m': pct(kospi, e, 90), 'spx_1m': pct(spx, e, 30)}
    fdf = pd.DataFrame(feat).T
    panel = df.join(fdf).dropna()
    fc = ['breadth', 'mom_decel', 'kospi_3m', 'spx_1m']; X = panel[fc].values; y = panel['winner_bear'].values; n = len(panel)
    print(f"  예측 패널 {n}개월", flush=True)

    pred = np.full(n, np.nan)
    for i in range(36, n):
        ytr = y[:i]
        if ytr.sum() < 6 or len(ytr) - ytr.sum() < 6: continue
        sc = StandardScaler().fit(X[:i]); lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000)
        lr.fit(sc.transform(X[:i]), ytr); pred[i] = lr.predict_proba(sc.transform(X[i:i+1]))[0, 1]
    m = ~np.isnan(pred); auc = roc_auc_score(y[m], pred[m]) if len(set(y[m])) > 1 else float('nan')
    print(f"\n  팩터로테이션 예측 OOS AUC = {auc:.3f}")

    ev = panel[m].copy(); ev['pred'] = (pred[m] >= 0.5).astype(int)
    ev['rot'] = np.where(ev['pred'] == 1, ev['bear'], ev['bull'])     # 예측 회전
    ev['perfect'] = np.where(ev['winner_bear'] == 1, ev['bear'], ev['bull'])  # 완벽(상한)
    def cum(r): return (1 + r / 100).prod() - 1
    print(f"\n  === 회전 PnL ({len(ev)}개월) ===")
    print(f"  항상 bull   : 누적 {cum(ev['bull'])*100:.0f}%  월평균 {ev['bull'].mean():+.2f}%")
    print(f"  항상 bear   : 누적 {cum(ev['bear'])*100:.0f}%  월평균 {ev['bear'].mean():+.2f}%")
    print(f"  예측 회전   : 누적 {cum(ev['rot'])*100:.0f}%  월평균 {ev['rot'].mean():+.2f}%")
    print(f"  완벽 회전(상한): 누적 {cum(ev['perfect'])*100:.0f}%")
    print("\n판정: AUC>0.55 & 예측회전 > 항상bull 이면 → 예측력+PnL 둘 다 (FCF 정답 타겟).")


if __name__ == "__main__":
    main()
