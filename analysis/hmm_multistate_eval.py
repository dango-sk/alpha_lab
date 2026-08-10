"""
analysis/hmm_multistate_eval.py

3/4-state HMM이 "진짜 전환 vs 일시적 약화(washout)"를 hidden state로 구분.
emission: [breadth, Δbreadth, newlow, Δnewlow, mom_decel] (기존 검증 신호만).
방어(Bear) state = bear_score 최대(낮은 breadth + 악화 Δbreadth + 높은 newlow + Δnewlow↑ + mom_decel↓).
→ washout(낮은 breadth지만 Δ 회복)은 방어 제외 → False Alarm↓, Lead 유지 목표.

평가(이벤트 기반): Miss / False Alarm / Lead / Whipsaw + Bull/Bear 수익.
2-state(기존, FA17/정밀32%)와 비교.
사용: .venv/bin/python analysis/hmm_multistate_eval.py
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
MIN_TRAIN, SEED, WIN = 36, 42, 6


def main():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    k = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators WHERE indicator='kospi' AND freq='D' ORDER BY period", conn); conn.close()
    d["dt"] = pd.to_datetime(d["dt"]); k["dt"] = pd.to_datetime(k["dt"]); kospi = k.set_index("dt")["v"].sort_index()
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
        rows.append({"ym": pd.Period(e, freq="M").strftime("%Y-%m"),
                     "breadth": asof(breadth, e), "newlow": asof(newlow, e),
                     "br_chg": asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
                     "nl_chg": asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
                     "mom_decel": pct(kospi, e, 30) - pct(kospi, e, 180),
                     "ret": (kospi.loc[mends[i+1]] / kospi.loc[e] - 1) * 100 if i + 1 < len(mends) else np.nan,
                     "dd6": dd6})
    df = pd.DataFrame(rows)
    feats = ["breadth", "br_chg", "newlow", "nl_chg", "mom_decel"]
    df_f = df.dropna(subset=feats).reset_index(drop=True); n = len(df_f); X = df_f[feats].values
    # 이벤트
    df_f["bear_zone"] = (df_f["dd6"] <= -15).astype(int)
    df_f["event"] = ((df_f["bear_zone"] == 1) & (df_f["bear_zone"].shift(1).fillna(0) == 0)).astype(int)
    events = df_f.index[df_f["event"] == 1].tolist()

    def eval_regime(reg, label):
        reg = np.array(reg)
        onset = [t for t in range(1, n) if reg[t] == "Bear" and reg[t-1] != "Bear"]
        trans = int((reg[1:] != reg[:-1]).sum())
        leads = []; miss = 0; matched = set()
        for ev in events:
            cand = [s for s in onset if abs(s - ev) <= WIN]
            if cand:
                b = min(cand, key=lambda s: abs(s - ev)); leads.append(ev - b); matched.add(b)
            else:
                miss += 1
        fa = [s for s in onset if all(abs(s - ev) > WIN for ev in events)]
        ev_idx = df_f.index >= MIN_TRAIN
        br = df_f.loc[(reg == "Bull") & ev_idx, "ret"].mean(); ber = df_f.loc[(reg == "Bear") & ev_idx, "ret"].mean()
        print(f"\n  [{label}] Bear월 {(reg=='Bear').sum()}, 전환(whipsaw) {trans}회")
        print(f"    포착 {len(leads)}/{len(events)} (Miss {miss}), 평균 Lead {np.mean(leads):+.1f}개월" if leads else f"    포착 0")
        print(f"    False Alarm {len(fa)}회 (onset {len(onset)}개, 정밀도 {len(matched)/max(len(onset),1)*100:.0f}%)")
        print(f"    Bull수익 {br:+.2f}% vs Bear수익 {ber:+.2f}% → 격차 {br-ber:+.2f}%p")
        return reg

    print(f"패널 {n}개월, 이벤트 {len(events)}개")
    best = None
    for ns in [3, 4]:
        states = np.full(n, -1)
        defensive_each = []
        for t in range(MIN_TRAIN, n):
            try:
                sc = StandardScaler().fit(X[:t]); Z = sc.transform(X[:t])
                hm = GaussianHMM(n_components=ns, covariance_type="diag", n_iter=80, random_state=SEED)
                hm.fit(Z)
                mu = hm.means_   # [state, feat] in std space: [breadth, br_chg, newlow, nl_chg, mom_decel]
                bear_score = -mu[:, 0] - mu[:, 1] + mu[:, 2] + mu[:, 3] - mu[:, 4]
                defn = int(np.argmax(bear_score))
                st = hm.predict(Z)[-1]
                states[t] = 1 if st == defn else 0
            except Exception:
                states[t] = 0
        reg = np.where(states == 1, "Bear", "Bull")
        r = eval_regime(reg, f"HMM {ns}-state")
        if ns == 4:
            json.dump(dict(zip(df_f["ym"], r)), open(A / "regime_hmm_multistate_map.json", "w"), ensure_ascii=False)
    print("\n(기존 2-state 단순: FA 17, 정밀도 32%, Lead +1.8 — 이거 대비 FA 줄고 Lead 유지면 성공)")
    print("저장: regime_hmm_multistate_map.json (4-state). 다음: FCF.")


if __name__ == "__main__":
    main()
