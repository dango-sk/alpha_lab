"""
analysis/breadth_transition_test.py

길1(breadth) + 길2(추세전환 타겟) 동시 실행.
- Feature: 시장 breadth (개별종목 MA120 상회 비율), SOX 상대강도
- 타겟A: 다음달 방향 (up) — 비교용
- 타겟B(핵심): 현재 상승추세(가격>MA150)인데 *향후 2개월 내 MA150 하향돌파* (=상승장 끝)
평가: no-fit expanding 분위수 AUC + tilt, + 최소 feature LR. 과적합 회피.

입력: Railway PG (daily_price 개별종목 2017~, macro kospi/sox)
사용: .venv/bin/python analysis/breadth_transition_test.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); load_dotenv(Path('.env'))


def main():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    print("개별종목 일별가 로드 중 (6.4M행, 수십초)...", flush=True)
    dp = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float c FROM alpha_lab.daily_price "
                     "WHERE close IS NOT NULL", conn)
    print(f"  {len(dp):,}행 로드. pivot...", flush=True)
    dp['dt'] = pd.to_datetime(dp['dt'])
    wide = dp.pivot_table(index='dt', columns='stock_code', values='c').sort_index()
    # breadth: MA120 상회 종목 비율 (해당일 데이터 있는 종목 중)
    ma120 = wide.rolling(120, min_periods=60).mean()
    above = (wide > ma120)
    valid = wide.notna() & ma120.notna()
    breadth = (above & valid).sum(axis=1) / valid.sum(axis=1).clip(lower=1)
    breadth = breadth[valid.sum(axis=1) > 200]   # 종목 충분한 날만
    print(f"  breadth 계산 완료: {breadth.index.min().date()}~{breadth.index.max().date()}", flush=True)

    # KOSPI / SOX
    def macd(ind):
        d = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators "
                        "WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
        d['dt'] = pd.to_datetime(d['dt']); return d.set_index('dt')['v'].sort_index()
    kospi = macd('kospi'); sox = macd('sox')
    conn.close()

    ym = pd.PeriodIndex(kospi.index, freq='M'); last = {}
    for d, p in zip(kospi.index, ym): last[p] = d
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]

    def asof(s, e):
        x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan

    def pct(s, e, d):
        c = asof(s, e); pa = s[s.index <= e - timedelta(days=d)]
        return (c / pa.iloc[-1] - 1) if len(pa) and pa.iloc[-1] else np.nan

    rows = []
    for i in range(len(mends) - 1):
        e = mends[i]
        ks = kospi[kospi.index <= e]
        if len(ks) < 150 or i + 2 >= len(mends):
            continue
        ma150 = ks.iloc[-150:].mean()
        up_now = asof(kospi, e) > ma150
        # 향후 2개월 내 MA150 하향돌파?
        broke = 0
        for j in [1, 2]:
            ej = mends[i + j]; maj = kospi[kospi.index <= ej].iloc[-150:].mean()
            if kospi.loc[ej] < maj:
                broke = 1
        rows.append({
            'dt': e, 'breadth': asof(breadth, e), 'sox_rs': pct(sox, e, 60) - pct(kospi, e, 60),
            'up_now': int(up_now),
            'dir_up': int(kospi.loc[mends[i + 1]] / kospi.loc[e] - 1 > 0),   # 타겟A
            'broke_2m': broke,                                              # 타겟B (전환)
        })
    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    print(f"\n표본 {len(df)}개월\n", flush=True)

    def nofit_auc(sig, label, sub=None):
        d = df if sub is None else df[sub].reset_index(drop=True)
        v = d[sig].values; y = d[label].values; n = len(d)
        sc = np.full(n, np.nan)
        for t in range(24, n):
            h = v[:t]; sc[t] = (h < v[t]).mean()
        m = ~np.isnan(sc)
        if len(set(y[m])) < 2: return None
        return roc_auc_score(y[m], sc[m])

    print("== 타겟A: 다음달 방향(up) — no-fit 분위수 AUC ==")
    for sig in ['breadth', 'sox_rs']:
        a = nofit_auc(sig, 'dir_up')
        # breadth는 높을수록 강세 → AUC가 0.5 위면 정방향
        print(f"  {sig:10} AUC {a:.3f}" if a else f"  {sig}: n/a")

    print("\n== 타겟B: 상승추세 중 향후2개월 MA150 하향돌파(전환) — *상승추세 월만* ==")
    upmask = df['up_now'] == 1
    nb = df[upmask]['broke_2m']
    print(f"  상승추세 월 {upmask.sum()}개, 그중 2개월내 돌파 {nb.sum()}개 ({nb.mean()*100:.0f}%)")
    for sig in ['breadth', 'sox_rs']:
        a = nofit_auc(sig, 'broke_2m', sub=upmask)
        # breadth 낮을수록 전환위험 → AUC<0.5면 정방향(예측력). |AUC-0.5|로 본다
        if a:
            strength = abs(a - 0.5)
            print(f"  {sig:10} AUC {a:.3f}  (전환예측력 |Δ0.5|={strength:.3f})")
    print("\n판정: 타겟B(전환)에서 breadth AUC가 0.5에서 크게 벗어나면 → 추세전환 예측 가능 = 돌파구")


if __name__ == '__main__':
    main()
