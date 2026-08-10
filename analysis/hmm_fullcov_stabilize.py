"""
analysis/hmm_fullcov_stabilize.py

메인 모델 = Full-Cov HMM. 분별력(Recall/격차) 유지하며 전환 안정화(Whipsaw/FA↓).
베이스라인 = 현재 Full-Cov HMM. 4개 변형 모두 full covariance emission 유지:
  1) sticky   : transmat_prior 대각 강화(자기상태 유지 prior)
  2) penalty  : Viterbi decoding에 state-switch cost(off-diagonal log-prob에 -c)
  3) hyst     : posterior P(Bear) 비대칭 threshold (진입>0.65 / 복귀<0.35)
  4) hsmm     : explicit-duration Viterbi (state별 duration 분포 학습; full-cov emission)
지표: Recall/Lead/FA/Whipsaw/Bear월/수익격차 (full 2000~ & overlap 2018~) + FCF CAGR/Sharpe/MDD.
사용: .venv/bin/python analysis/hmm_fullcov_stabilize.py   (백그라운드 권장)
"""
import os, json, math, sys, warnings
import numpy as np, pandas as pd, psycopg2
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore"); load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))
A = Path(__file__).parent
MIN_TRAIN, SEED, WIN, OV = 36, 42, 6, "2018-04"
BULL, BEAR = "FCF_YIELD추가전략", "FCF_YIELD_BEAR전략"
STICKY_S, PENALTY_C, T_IN, T_OUT, DMAX = 50.0, 2.0, 0.65, 0.35, 48
NEG = -1e18


def build_panel():
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
    F = np.nan_to_num(np.array([[asof(breadth, e), asof(breadth, e) - asof(breadth, e - timedelta(days=30)),
        asof(newlow, e), asof(newlow, e) - asof(newlow, e - timedelta(days=30)),
        pct(kospi, e, 30) - pct(kospi, e, 180)] for e in mends]))
    Px = np.array([asof(kospi, e) for e in mends]); yms = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    ret = np.array([(Px[i+1]/Px[i]-1)*100 if i+1 < n else np.nan for i in range(n)])
    dd6 = np.full(n, np.nan)
    for i in range(n-1):
        if i+6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i+6])]])
            dd6[i] = (path/path.cummax()-1).min()*100
    return F, yms, n, ret, dd6


def bear_state(mu):
    return int(np.argmax(-mu[:, 0] - mu[:, 1] + mu[:, 2] + mu[:, 3] - mu[:, 4]))


def viterbi(logB, logstart, logT):
    """표준 Viterbi, 마지막 시점 state 반환."""
    L, S = logB.shape
    V = logstart + logB[0]; bp = np.zeros((L, S), int)
    for t in range(1, L):
        for j in range(S):
            cand = V + logT[:, j]; bp[t, j] = int(np.argmax(cand)); V = V  # noqa
        nv = np.empty(S)
        for j in range(S):
            cand = V + logT[:, j]; bp[t, j] = int(np.argmax(cand)); nv[j] = cand[bp[t, j]] + logB[t, j]
        V = nv
    return int(np.argmax(V))


def nb_logpmf_fn(durs):
    """state별 duration 분포: NB(MoM), 부족하면 완만한 geometric fallback. d>=1."""
    if len(durs) >= 3:
        m = float(np.mean(durs)); v = float(np.var(durs, ddof=1))
        if v > m > 0:
            p = m / v; r = m * m / (v - m)
            if r > 0:
                def f(d, r=r, p=p):
                    k = d - 1  # shift: 최소 duration 1
                    return math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1) + r * math.log(p) + k * math.log(1 - p)
                return f
    m = max(float(np.mean(durs)) if len(durs) else 6.0, 1.5)
    q = 1.0 / m  # geometric mean m
    return lambda d, q=q: (d - 1) * math.log(1 - q) + math.log(q)


def hsmm_last_state(logB, logstart, logD, bear):
    """explicit-duration Viterbi (2-state, 전환은 결정적). 마지막 시점 state 반환."""
    L, S = logB.shape
    cum = np.zeros((S, L + 1))
    for i in range(S):
        cum[i, 1:] = np.cumsum(logB[:, i])
    V = np.full((L, S), NEG)
    for end in range(L):
        for i in range(S):
            best = NEG
            dmax = min(DMAX, end + 1)
            for d in range(1, dmax + 1):
                a = end - d + 1
                seg = cum[i, end + 1] - cum[i, a]
                dur = logD[i](d)
                if a == 0:
                    val = logstart[i] + dur + seg
                else:
                    prev = max(V[a - 1, j] for j in range(S) if j != i)  # 전환 logA=0
                    val = prev + dur + seg
                if val > best:
                    best = val
            V[end, i] = best
    return int(np.argmax(V[L - 1]))


def gen_maps(F, yms, n):
    """walk-forward로 5개(baseline+4변형) 레짐 시퀀스 생성."""
    base, sticky, penalty, hyst, hsmm = (["Bull"] * n for _ in range(5))
    for t in range(MIN_TRAIN, n):
        sc = StandardScaler().fit(F[:t]); Z = sc.transform(F[:t])
        # --- baseline full ---
        try:
            hm = GaussianHMM(2, "full", n_iter=60, random_state=SEED); hm.fit(Z)
            b = bear_state(hm.means_)
            base[t] = "Bear" if hm.predict(Z)[-1] == b else "Bull"
            logB = hm._compute_log_likelihood(Z)
            logstart = np.log(np.clip(hm.startprob_, 1e-12, 1)); logT = np.log(np.clip(hm.transmat_, 1e-12, 1))
            # --- penalty (switch cost on baseline model) ---
            logTp = logT.copy()
            for i in range(2):
                for j in range(2):
                    if i != j: logTp[i, j] -= PENALTY_C
            penalty[t] = "Bear" if viterbi(logB, logstart, logTp) == b else "Bull"
            # --- hysteresis on posterior ---
            pbear = hm.predict_proba(Z)[-1, b]; prev = hyst[t - 1]
            if prev == "Bear": hyst[t] = "Bear" if pbear >= T_OUT else "Bull"
            else: hyst[t] = "Bear" if pbear >= T_IN else "Bull"
            # --- hsmm explicit-duration (durations from baseline Viterbi seg) ---
            path = hm.predict(Z); durs = {0: [], 1: []}
            run_s, run_l = path[0], 1
            for k in range(1, len(path)):
                if path[k] == run_s: run_l += 1
                else: durs[run_s].append(run_l); run_s, run_l = path[k], 1
            durs[run_s].append(run_l)
            logD = {i: nb_logpmf_fn(durs[i]) for i in range(2)}
            hsmm[t] = "Bear" if hsmm_last_state(logB, logstart, logD, b) == b else "Bull"
        except Exception:
            for arr in (base, penalty, hyst, hsmm): arr[t] = arr[t - 1]
        # --- sticky (full cov + diagonal transmat_prior) ---
        try:
            hs = GaussianHMM(2, "full", n_iter=60, random_state=SEED,
                             transmat_prior=np.array([[STICKY_S, 1.0], [1.0, STICKY_S]]))
            hs.fit(Z); bs = bear_state(hs.means_)
            sticky[t] = "Bear" if hs.predict(Z)[-1] == bs else "Bull"
        except Exception:
            sticky[t] = sticky[t - 1]
    return {"full(base)": base, "sticky": sticky, "penalty": penalty, "hyst": hyst, "hsmm": hsmm}


def events_in(idx, dd6):
    return [i for i in idx if dd6[i] <= -15 and (i == 0 or not (dd6[i - 1] <= -15))]


def evalr(reg, yms, n, ret, dd6, idx):
    events = events_in(idx, dd6)
    onset = [t for t in idx if reg[t] == "Bear" and t - 1 >= 0 and reg[t - 1] != "Bear"]
    leads = []
    for ev in events:
        cand = [s for s in onset if abs(s - ev) <= WIN]
        if cand: leads.append(ev - min(cand, key=lambda s: abs(s - ev)))
    fa = sum(1 for s in onset if all(abs(s - ev) > WIN for ev in events))
    whip = sum(1 for k in range(1, len(idx)) if reg[idx[k]] != reg[idx[k - 1]])
    bearm = sum(1 for i in idx if reg[i] == "Bear")
    br = np.nanmean([ret[i] for i in idx if reg[i] == "Bull"]); be = np.nanmean([ret[i] for i in idx if reg[i] == "Bear"])
    gap = (br - be) if not (np.isnan(br) or np.isnan(be)) else float("nan")
    ld = np.mean(leads) if leads else float("nan")
    return len(leads), len(events), ld, fa, whip, bearm, br, be, gap


def ai_v2_map(yms):
    aip = A / "regime_agent_results.json"
    if not aip.exists(): return None
    ers = {r["as_of"][:7]: r.get("expected_return", 0) for r in json.load(open(aip))}
    reg = {}; prev = "Bull"
    for y in sorted(ers):
        er = ers[y]; cur = ("Bear" if er <= -2 else "Bull") if prev == "Bull" else ("Bull" if er >= 1 else "Bear")
        reg[y] = cur; prev = cur
    return [reg.get(y, "Bull") for y in yms]


def main():
    from lib.data import run_regime_combo_backtest
    F, yms, n, ret, dd6 = build_panel()
    print(f"패널 {n}개월. walk-forward 맵 생성 중...", flush=True)
    maps = gen_maps(F, yms, n)
    aiv = ai_v2_map(yms)
    full_idx = list(range(n)); ov_idx = [i for i in range(n) if yms[i] >= OV]

    def row(name, reg, idx):
        rc, ne, ld, fa, whip, bm, br, be, gap = evalr(reg, yms, n, ret, dd6, idx)
        lds = f"{ld:>+5.1f}" if not np.isnan(ld) else "  n/a"
        return f"  {name:12} {rc}/{ne:<5} {lds} {fa:>4} {whip:>5} {bm:>5} {br:>+6.2f}% {be:>+6.2f}% {gap:>+6.2f}p"

    hdr = f"  {'변형':12} {'Recall':>7} {'Lead':>5} {'FA':>4} {'Whip':>5} {'Bear월':>5} {'Bull평균':>7} {'Bear평균':>7} {'격차':>7}"
    print("\n=== 이벤트 지표 (full 2000~, 베이스라인=full(base)) ===")
    print(hdr)
    for k, v in maps.items(): print(row(k, v, full_idx))
    print("\n=== 이벤트 지표 (overlap 2018~2026, AI v2 겹침 구간) ===")
    print(hdr)
    for k, v in maps.items(): print(row(k, v, ov_idx))
    if aiv: print(row("ai_v2", aiv, ov_idx))

    # === 위기별 detection table ===
    NAMES = {"2018-04": "2018 하락", "2019-09": "2019 급락", "2020-02": "2020 코로나",
             "2021-07": "2021 하락", "2021-11": "2021~22 하락", "2024-02": "2024 하락", "2025-09": "2025 하락"}
    print("\n=== 위기별 detection (full 2000~; 값=선행개월 +빠름/-늦음, X=놓침, ()안=±6M whipsaw수) ===")
    evs = events_in(full_idx, dd6)
    cols = list(maps.items()) + ([("ai_v2", aiv)] if aiv else [])
    print(f"  {'위기':14} {'낙폭':>5}  " + " ".join(f"{name:>12}" for name, _ in cols))
    for ev in evs:
        ymv = yms[ev]; label = NAMES.get(ymv, ymv); cells = []
        for _, reg in cols:
            onset = [s for s in full_idx if reg[s] == "Bear" and s - 1 >= 0 and reg[s - 1] != "Bear"]
            cand = [s for s in onset if abs(s - ev) <= WIN]
            wp = sum(1 for k in range(max(1, ev - WIN), min(n, ev + WIN + 1)) if reg[k] != reg[k - 1])
            if cand:
                s0 = min(cand, key=lambda s: abs(s - ev)); cells.append(f"{ev - s0:>+3d}({wp})")
            else:
                cells.append(f"  X({wp})")
        print(f"  {label:14} {dd6[ev]:>4.0f}%  " + " ".join(f"{c:>12}" for c in cells))

    # --- FCF ---
    slot = A / "regime_rf_map.json"; bak = A / "regime_rf_map.json.stbak"
    if slot.exists() and not bak.exists(): bak.write_text(slot.read_text())
    fcf = {}
    print("\n=== FCF 백테스트 ===", flush=True)
    for k, v in maps.items():
        slot.write_text(json.dumps(dict(zip(yms, v)), ensure_ascii=False))
        r = run_regime_combo_backtest(bull_key=BULL, bear_key=BEAR, universe="KOSPI", rebal_type="monthly", regime_mode="rf")
        c = (r or {}).get("REGIME_COMBO", {}); fcf[k] = c
        print(f"  {k:12} done", flush=True)
    if bak.exists(): slot.write_text(bak.read_text())  # 원복

    print(f"\n  {'변형':12} {'CAGR':>7} {'Sharpe':>7} {'MDD':>7} {'Calmar':>7}")
    for k, c in fcf.items():
        if not c: print(f"  {k:12} (no result)"); continue
        cg, md, sh = c.get("cagr"), c.get("mdd"), c.get("sharpe")
        cal = (cg / abs(md)) if (cg is not None and md) else float("nan")
        print(f"  {k:12} {cg*100:>6.1f}% {sh:>7.2f} {md*100:>6.1f}% {cal:>7.2f}")
    print("\n목표: full(base) 대비 Recall·격차 유지 + FA·Whipsaw↓. (sticky_S={}, penalty_c={}, hyst {}/{})".format(STICKY_S, PENALTY_C, T_IN, T_OUT))


if __name__ == "__main__":
    main()
