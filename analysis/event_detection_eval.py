"""
analysis/event_detection_eval.py

레짐 전환을 *이벤트 감지*로 평가 (단순 AUC 아니라 timing).
실제 약세전환 이벤트 = forward 6m 낙폭이 -15%를 처음 넘는 월(약세 진입점).
breadth 신호(천장)가 그 이벤트를 며칠(개월) 전/후에 잡는지:
  · Lead Time (이벤트월 - 신호월; + = 먼저 감지=좋음)
  · Miss (이벤트 ±6개월 내 신호 없음)
  · False Alarm (신호인데 근처에 이벤트 없음)
  · 위기별 오차

신호: 천장스코어 = ½[(1-breadth분위)+(mom_decel분위)], >=0.6 또는 breadth 급락 → Bear onset.
입력: kospi_stocks_2000_2016.csv + daily_price + kospi index.
사용: .venv/bin/python analysis/event_detection_eval.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
WIN = 6   # 이벤트 매칭 윈도우(개월)


def main():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    k = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators WHERE indicator='kospi' AND freq='D' ORDER BY period", conn); conn.close()
    d["dt"] = pd.to_datetime(d["dt"]); k["dt"] = pd.to_datetime(k["dt"]); kospi = k.set_index("dt")["v"].sort_index()
    wide = pd.concat([h, d[["dt", "stock_code", "close"]]], ignore_index=True).drop_duplicates(["dt", "stock_code"]).pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    ma = wide.rolling(120, min_periods=60).mean(); val = wide.notna() & ma.notna()
    breadth = ((wide > ma) & val).sum(axis=1) / val.sum(axis=1).clip(lower=1); breadth = breadth[val.sum(axis=1) > 100]

    ym = pd.PeriodIndex(kospi.index, freq="M"); last = {}
    for dt_, p in zip(kospi.index, ym): last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1) if len(p) and p.iloc[-1] else np.nan

    rows = []
    for i in range(len(mends) - 1):
        e = mends[i]
        dd6 = np.nan
        if i + 6 < len(mends):
            path = pd.concat([pd.Series([kospi.loc[e]], index=[e]), kospi[(kospi.index > e) & (kospi.index <= mends[i+6])]])
            dd6 = (path / path.cummax() - 1).min() * 100
        rows.append({"ym": pd.Period(e, freq="M").strftime("%Y-%m"), "breadth": asof(breadth, e),
                     "br_chg": asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
                     "mom_decel": pct(kospi, e, 30) - pct(kospi, e, 180), "dd6": dd6})
    df = pd.DataFrame(rows); n = len(df)
    # 실제 약세전환 이벤트: dd6가 -15% 처음 넘는 월
    df["bear_zone"] = (df["dd6"] <= -15).astype(int)
    df["event"] = ((df["bear_zone"] == 1) & (df["bear_zone"].shift(1).fillna(0) == 0)).astype(int)
    events = df.index[df["event"] == 1].tolist()

    # 신호: 천장스코어 expanding 분위수
    def ep(arr, t): h = arr[:t]; return (h < arr[t]).mean() if t > 0 else .5
    sig = np.zeros(n, bool)
    for t in range(12, n):
        top = np.mean([1 - ep(df["breadth"].values, t), ep(np.nan_to_num(df["mom_decel"].values), t)])
        sudden = ep(np.nan_to_num(df["br_chg"].values), t) <= 0.15
        sig[t] = (top >= 0.6) or sudden
    # 신호 onset = Bull→신호 전환점
    sig_onset = [t for t in range(1, n) if sig[t] and not sig[t-1]]

    print(f"패널 {n}개월, 실제 약세전환 이벤트 {len(events)}개, 신호발생 {len(sig_onset)}개\n")
    print("=== 위기별 전환 감지 (Lead+ = 먼저 감지) ===")
    matched_sig = set(); leads = []; miss = 0
    for ev in events:
        cand = [s for s in sig_onset if abs(s - ev) <= WIN]
        if cand:
            best = min(cand, key=lambda s: abs(s - ev))
            lead = ev - best   # +면 신호가 먼저
            leads.append(lead); matched_sig.add(best)
            print(f"  이벤트 {df['ym'].iloc[ev]} (낙폭 {df['dd6'].iloc[ev]:.0f}%): 신호 {df['ym'].iloc[best]}  Lead {lead:+d}개월 {'(선행)' if lead>0 else '(지연)' if lead<0 else '(동시)'}")
        else:
            miss += 1
            print(f"  이벤트 {df['ym'].iloc[ev]} (낙폭 {df['dd6'].iloc[ev]:.0f}%): ❌ 놓침 (±{WIN}개월 내 신호없음)")
    fa = [s for s in sig_onset if all(abs(s - ev) > WIN for ev in events)]
    print(f"\n=== 요약 ===")
    print(f"  포착 {len(leads)}/{len(events)}  Miss Rate {miss/max(len(events),1)*100:.0f}%")
    if leads: print(f"  평균 Lead {np.mean(leads):+.1f}개월 (중앙 {np.median(leads):+.0f})  — +면 선행")
    print(f"  False Alarm {len(fa)}회 (신호 {len(sig_onset)}개 중)  → 정밀도 {len(matched_sig)/max(len(sig_onset),1)*100:.0f}%")


if __name__ == "__main__":
    main()
