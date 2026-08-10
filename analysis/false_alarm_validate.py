"""
analysis/false_alarm_validate.py

가설 검증: "False Alarm = 얕은 약화(mom_decel≈0 + SOX RS 강세)"가 OOS에서도 holding 하나.
라벨(6개월 FA/TA 정의)은 그대로. 시간 분할로 전반부 학습 → 후반부 OOS AUC.
OOS에서 살면 → confirmation gate/emission 반영. 무너지면 → 과적합 폐기.
사용: .venv/bin/python analysis/false_alarm_validate.py
"""
import os, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
WIN = 6


def main():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    def md(ind):
        x = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators WHERE indicator=%s AND freq='D' ORDER BY period", conn, params=(ind,))
        x["dt"] = pd.to_datetime(x["dt"]); return x.set_index("dt")["v"].sort_index()
    kospi = md("kospi"); sox = md("sox"); conn.close()
    d["dt"] = pd.to_datetime(d["dt"])
    wide = pd.concat([h, d[["dt", "stock_code", "close"]]], ignore_index=True).drop_duplicates(["dt", "stock_code"]).pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    ma = wide.rolling(120, min_periods=60).mean(); val = wide.notna() & ma.notna()
    breadth = ((wide > ma) & val).sum(axis=1) / val.sum(axis=1).clip(lower=1); breadth = breadth[val.sum(axis=1) > 100]
    rmn = wide.rolling(252, min_periods=120).min(); newlow = ((wide <= rmn) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1); newlow = newlow.reindex(breadth.index)
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
                     "mom_decel": pct(kospi, e, 30) - pct(kospi, e, 180),
                     "sox_rs": pct(sox, e, 60) - pct(kospi, e, 60), "dd6": dd6})
    df = pd.DataFrame(rows).reset_index(drop=True); n = len(df)
    df["event"] = (((df["dd6"] <= -15) & (df["dd6"].shift(1).fillna(0) > -15))).astype(int)
    events = df.index[df["event"] == 1].tolist()
    def ep(arr, t): hh = arr[:t]; return (hh < arr[t]).mean() if t > 0 else .5
    sig = np.zeros(n, bool)
    for t in range(12, n):
        top = np.mean([1 - ep(df["breadth"].values, t), ep(np.nan_to_num(df["mom_decel"].values), t)])
        sig[t] = (top >= 0.6) or (ep(np.nan_to_num(df["br_chg"].values), t) <= 0.15)
    onsets = [t for t in range(1, n) if sig[t] and not sig[t-1]]
    lab = np.array([1 if any(abs(s - ev) <= WIN for ev in events) else 0 for s in onsets])  # 1=TA
    od = df.iloc[onsets].reset_index(drop=True)
    od["TA"] = lab
    print(f"경보 {len(onsets)}개: TA {lab.sum()}, FA {(lab==0).sum()}\n")

    # 시간 2분할
    half = len(od) // 2
    early, late = od.iloc[:half], od.iloc[half:]
    print(f"전반부 {od['ym'].iloc[0]}~{od['ym'].iloc[half-1]} ({len(early)}개, FA {(early['TA']==0).sum()})")
    print(f"후반부 {od['ym'].iloc[half]}~{od['ym'].iloc[-1]} ({len(late)}개, FA {(late['TA']==0).sum()})\n")

    print("=== feature별 FA설명 AUC (FA=1로) : 전반부 vs 후반부 ===")
    for c in ["mom_decel", "sox_rs", "breadth"]:
        def au(sub):
            y = 1 - sub["TA"].values  # FA=1
            return roc_auc_score(y, -sub[c]) if len(set(y)) > 1 else np.nan  # mom_decel↑/sox_rs↑(=FA)면 -붙여 정렬
        a_e, a_l = au(early), au(late)
        print(f"  {c:10}  전반부 {a_e:.3f}   후반부(OOS) {a_l:.3f}")

    # 전반부 학습 LR([mom_decel, sox_rs]) → 후반부 OOS
    Xe = early[["mom_decel", "sox_rs"]].values; ye = 1 - early["TA"].values
    Xl = late[["mom_decel", "sox_rs"]].values; yl = 1 - late["TA"].values
    if len(set(ye)) > 1 and len(set(yl)) > 1:
        sc = StandardScaler().fit(Xe); lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Xe), ye)
        ins = roc_auc_score(ye, lr.predict_proba(sc.transform(Xe))[:, 1])
        oos = roc_auc_score(yl, lr.predict_proba(sc.transform(Xl))[:, 1])
        print(f"\n  LR[mom_decel,sox_rs] FA예측:  전반부(IS) {ins:.3f}  →  후반부(OOS) {oos:.3f}")
        print(f"  계수: mom_decel {lr.coef_[0][0]:+.2f}, sox_rs {lr.coef_[0][1]:+.2f}")
    print("\n판정: 후반부 OOS AUC > 0.6 유지면 → 패턴 진짜(gate 반영). ~0.5면 → 과적합 폐기.")


if __name__ == "__main__":
    main()
