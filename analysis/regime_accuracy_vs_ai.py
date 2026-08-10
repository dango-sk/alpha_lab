"""
analysis/regime_accuracy_vs_ai.py

레짐 *예측 정확도* 비교: 우리 후보(Rule A, HMM full/diag) vs AI v2.
AI v2 존재구간(2018~2026) 겹침에서: Recall / Lead / FA / Whipsaw + Bull/Bear 수익격차.
실제 전환 이벤트 = dd6<=-15 onset.
사용: .venv/bin/python analysis/regime_accuracy_vs_ai.py
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
OV_START = "2018-04"


def main():
    h = pd.read_csv(A / "kospi_stocks_2000_2016.csv", dtype={"stock_code": str}, parse_dates=["dt"])[["dt", "stock_code", "close"]]
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price WHERE close IS NOT NULL", conn)
    kk = pd.read_sql("SELECT period::date dt, value::float v FROM alpha_lab.macro_indicators WHERE indicator='kospi' AND freq='D' ORDER BY period", conn); conn.close()
    d["dt"] = pd.to_datetime(d["dt"]); kk["dt"] = pd.to_datetime(kk["dt"]); kospi = kk.set_index("dt")["v"].sort_index()
    wide = pd.concat([h, d[["dt", "stock_code", "close"]]], ignore_index=True).drop_duplicates(["dt", "stock_code"]).pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    ma = wide.rolling(120, min_periods=60).mean(); val = wide.notna() & ma.notna()
    breadth = ((wide > ma) & val).sum(axis=1) / val.sum(axis=1).clip(lower=1); breadth = breadth[val.sum(axis=1) > 100]
    rmn = wide.rolling(252, min_periods=120).min(); newlow = ((wide <= rmn) & wide.notna()).sum(axis=1) / wide.notna().sum(axis=1).clip(lower=1); newlow = newlow.reindex(breadth.index)
    ym = pd.PeriodIndex(kospi.index, freq="M"); last = {}
    for dt_, p in zip(kospi.index, ym): last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= breadth.index.min()]
    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pct(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c/p.iloc[-1]-1) if len(p) and p.iloc[-1] else np.nan
    n = len(mends); yms = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    F = np.nan_to_num(np.array([[asof(breadth, e), asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
        asof(newlow, e), asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
        pct(kospi, e, 30) - pct(kospi, e, 180)] for e in mends]))
    Px = np.array([asof(kospi, e) for e in mends])
    ret = np.array([(Px[i+1]/Px[i]-1)*100 if i+1 < n else np.nan for i in range(n)])
    dd6 = np.full(n, np.nan)
    for i in range(n-1):
        if i+6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path/path.cummax()-1).min()*100

    # 후보 레짐
    regs = {}
    if (A / "regime_ABC_A.json").exists():
        m = json.load(open(A / "regime_ABC_A.json")); regs["ruleA"] = [m.get(y, "Bull") for y in yms]
    for kind, cov in [("hmm_full", "full"), ("hmm_diag", "diag")]:
        reg = ["Bull"] * n
        for t in range(MIN_TRAIN, n):
            try:
                sc = StandardScaler().fit(F[:t]); Z = sc.transform(F[:t]); hm = GaussianHMM(2, cov, n_iter=60, random_state=SEED); hm.fit(Z)
                mu = hm.means_; bear = int(np.argmax(-mu[:,0]-mu[:,1]+mu[:,2]+mu[:,3]-mu[:,4]))
                reg[t] = "Bear" if hm.predict(Z)[-1] == bear else "Bull"
            except Exception: reg[t] = reg[t-1]
        regs[kind] = reg
    # AI v2 (regime_agent_results.json, er-based, 백테스트와 동일)
    aip = A / "regime_agent_results.json"
    ai_reg = {}
    if aip.exists():
        ers = {r["as_of"][:7]: r.get("expected_return", 0) for r in json.load(open(aip))}
        prev = "Bull"
        for y in sorted(ers):
            er = ers[y]; cur = ("Bear" if er <= -2 else "Bull") if prev == "Bull" else ("Bull" if er >= 1 else "Bear")
            ai_reg[y] = cur; prev = cur
    regs["ai_v2"] = [ai_reg.get(y, "Bull") for y in yms]

    # 겹침 구간 인덱스
    ov = [i for i in range(n) if yms[i] >= OV_START and (ai_reg.get(yms[i]) is not None or yms[i] >= OV_START)]
    ov = [i for i in range(n) if yms[i] >= OV_START and yms[i] <= max(ai_reg.keys() if ai_reg else [OV_START])]
    events = [i for i in ov if dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]
    print(f"겹침 {yms[ov[0]]}~{yms[ov[-1]]} ({len(ov)}개월), 실제 전환 이벤트 {len(events)}개\n")
    print(f"  {'후보':9} {'Recall':>8} {'Lead':>6} {'FA':>4} {'Whip':>5} {'Bull수익':>8} {'Bear수익':>8} {'격차':>7}")
    for name, reg in regs.items():
        onset = [t for t in ov if reg[t] == "Bear" and t-1 >= 0 and reg[t-1] != "Bear"]
        leads = []; miss = 0
        for ev in events:
            cand = [s for s in onset if abs(s-ev) <= WIN]
            if cand: leads.append(ev - min(cand, key=lambda s: abs(s-ev)))
            else: miss += 1
        fa = sum(1 for s in onset if all(abs(s-ev) > WIN for ev in events))
        whip = sum(1 for i in range(1, len(ov)) if reg[ov[i]] != reg[ov[i-1]])
        br = np.nanmean([ret[i] for i in ov if reg[i] == "Bull"]); ber = np.nanmean([ret[i] for i in ov if reg[i] == "Bear"])
        ld = f"{np.mean(leads):+.1f}" if leads else "n/a"
        print(f"  {name:9} {len(leads)}/{len(events):<6} {ld:>6} {fa:>4} {whip:>5} {br:>+7.2f}% {ber:>+7.2f}% {br-ber:>+6.2f}p")
    print("\n  (Recall·Lead 높고 FA·Whip 낮고 격차 큰 게 레짐예측 우수. AI v2 대비 우리 후보 비교)")


if __name__ == "__main__":
    main()
