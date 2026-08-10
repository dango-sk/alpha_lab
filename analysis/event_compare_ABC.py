"""
analysis/event_compare_ABC.py

A(즉시 Bear) / B(전체 1개월확인) / C(하이브리드: 강한경보 즉시+약한경보 1개월확인) 비교.
Part1: 1개월확인이 어떤 TA를 탈락시키나 (위기별, 코로나/GFC 급락형 놓치나).
Part2: 탈락/지연 TA의 비용 (1개월 대기 중 KOSPI 낙폭, 손절지연).
Part3: 강한경보 기준 OOS 선택 (Δbreadth급락 / Δnewlow급증 / KOSPI1m급락).
Part4(event): Miss/FA/Lead/Whipsaw. + A/B/C 레짐맵 저장(→FCF).
사용: .venv/bin/python analysis/event_compare_ABC.py
"""
import os, json, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
A = Path(__file__).parent
WIN = 6


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
    n = len(mends)
    Br = np.array([asof(breadth, e) for e in mends]); Nl = np.array([asof(newlow, e) for e in mends])
    Md = np.array([pct(kospi, e, 30) - pct(kospi, e, 180) for e in mends])
    Brc = np.array([asof(breadth, mends[i]) - asof(breadth, mends[i] - timedelta(days=30)) for i in range(n)])
    Nlc = np.array([(Nl[i] - Nl[i-1]) if i > 0 else 0 for i in range(n)])
    Px = np.array([asof(kospi, e) for e in mends]); Ret1 = np.array([(Px[i]/Px[i-1]-1)*100 if i > 0 else 0 for i in range(n)])
    yms = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    dd6 = np.full(n, np.nan)
    for i in range(n - 1):
        if i + 6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path / path.cummax() - 1).min() * 100
    events = [i for i in range(n) if dd6[i] <= -15 and (i == 0 or not (dd6[i-1] <= -15))]

    def ep(arr, t):
        hh = [a for a in arr[:t] if a == a]; return (np.array(hh) < arr[t]).mean() if hh else .5
    sigA = np.zeros(n, bool)  # breadth bear signal active
    for t in range(12, n):
        top = np.mean([1 - ep(Br, t), ep(np.nan_to_num(Md), t)])
        sigA[t] = (top >= 0.6) or (ep(np.nan_to_num(Brc), t) <= 0.15)
    onsets = [t for t in range(1, n) if sigA[t] and not sigA[t-1] and t + 1 < n]
    is_TA = {t: any(abs(t - ev) <= WIN for ev in events) for t in onsets}

    # ── Part3: 강한경보 기준 OOS 선택 ──
    half_t = onsets[len(onsets)//2]
    print("=== Part3: 강한경보 후보별 TA정밀도 (전반부 vs 후반부 OOS) ===")
    cands = {
        "Δbreadth급락(≤10%)": lambda t: ep(np.nan_to_num(Brc), t) <= 0.10,
        "Δnewlow급증(≥90%)": lambda t: ep(np.nan_to_num(Nlc), t) >= 0.90,
        "KOSPI1m급락(≤10%)": lambda t: ep(Ret1, t) <= 0.10,
    }
    best_strong = None; best_oos = 0
    for nm, fn in cands.items():
        e_on = [t for t in onsets if t <= half_t]; l_on = [t for t in onsets if t > half_t]
        def prec(group):
            s = [t for t in group if fn(t)]; return (np.mean([is_TA[t] for t in s]) if s else np.nan, len(s))
        pe, ne = prec(e_on); pl, nl_ = prec(l_on)
        print(f"  {nm:18} 전반부 TA정밀 {pe:.2f}(n{ne})  후반부 {pl:.2f}(n{nl_})")
        if pl == pl and pl >= best_oos and nl_ >= 2:
            best_oos = pl; best_strong = nm
    print(f"  → 강한경보 채택: {best_strong} (후반부 정밀 {best_oos:.2f})")
    strong_fn = cands[best_strong]

    # ── 레짐 구성 A/B/C ──
    def build(mode):
        reg = ["Bull"] * n; state = "Bull"
        for t in range(1, n):
            if sigA[t] and not sigA[t-1]:  # warning onset
                go_now = (mode == "A") or (mode == "C" and strong_fn(t))
                if go_now:
                    state = "Bear"
                # B 또는 C-약한: t+1에서 확인 (아래 t+1 처리)
            elif not sigA[t] and sigA[t-1]:  # signal off → 회복
                state = "Bull"
            # 1개월 확인: 직전월이 warning onset이고 아직 Bull이면, Δbreadth로 판정
            if t >= 2 and sigA[t-1] and not sigA[t-2] and state == "Bull":
                weak = not ((mode == "A") or (mode == "C" and strong_fn(t-1)))
                if mode in ("B", "C") and weak:
                    if (Br[t] - Br[t-1]) <= 0:   # 악화 지속 → 확인
                        state = "Bear"
                    # 회복 → 취소 (Bull 유지)
            reg[t] = state
        return reg

    regs = {m: build(m) for m in ["A", "B", "C"]}

    # ── Part1: B가 탈락시킨 TA ──
    print("\n=== Part1: 1개월확인(B)이 탈락/지연시킨 True Alarm ===")
    for t in onsets:
        if not is_TA[t]: continue
        a_bear = regs["A"][t] == "Bear"; b_bear_t = regs["B"][t] == "Bear"; b_bear_t1 = (t+1 < n and regs["B"][t+1] == "Bear")
        if a_bear and not b_bear_t1:
            drop_dd = dd6[t]
            print(f"  {yms[t]} (낙폭 {drop_dd:.0f}%): B 탈락 ❌ {'← 급락형!' if drop_dd<=-25 else ''}")
        elif a_bear and not b_bear_t and b_bear_t1:
            cost = (Px[t+1]/Px[t]-1)*100
            print(f"  {yms[t]} (낙폭 {dd6[t]:.0f}%): B 1개월 지연 (그동안 KOSPI {cost:+.1f}%)")

    # ── Part4: event 지표 ──
    print("\n=== Part4(event): Miss / FA / Lead / Whipsaw ===")
    for m in ["A", "B", "C"]:
        reg = regs[m]; onset_m = [t for t in range(1, n) if reg[t] == "Bear" and reg[t-1] != "Bear"]
        leads = []; miss = 0
        for ev in events:
            cand = [s for s in onset_m if abs(s - ev) <= WIN]
            if cand: leads.append(ev - min(cand, key=lambda s: abs(s-ev)))
            else: miss += 1
        fa = sum(1 for s in onset_m if all(abs(s - ev) > WIN for ev in events))
        whip = sum(1 for t in range(1, n) if reg[t] != reg[t-1])
        print(f"  {m}: 포착 {len(leads)}/{len(events)} (Miss {miss}), 평균Lead {np.mean(leads):+.1f}, FA {fa}, Whipsaw {whip}, Bear월 {sum(1 for x in reg if x=='Bear')}")
        json.dump(dict(zip(yms, reg)), open(A / f"regime_ABC_{m}.json", "w"), ensure_ascii=False)
    print("\n저장: regime_ABC_{A,B,C}.json → 다음 FCF로 CAGR/Sharpe/MDD 비교")


if __name__ == "__main__":
    main()
