"""
analysis/hsmm_dualchannel.py

이중 채널 실험 — P_bear = max(P_shock, P_trend).  production 미수정.

■ 왜 이 구조인가
  2022 slow bear 미탐지의 원인은 emission **내부의 경쟁**이다.
  2022-01에 breadth·trend는 crash 상태와 사실상 동일(|z| 0.07 / 0.01)했는데
  newlow가 5.59σ로 단독 veto를 행사해 진입이 막혔다(가우시안 우도는 z²에 비례).

  지금까지의 처방은 전부 '같은 emission 안에서' 풀려다 실패했다:
    · newlow 교체(ever_newlow_20)     → veto는 사라지나 신호도 소실(Bear비율 78%, 리프트 1.04)
    · newlow 압축(zclip2/pct60)       → veto 85%→34%로 완화됐으나 crash 상태 자체가 소멸
    · ma200_slope emission 추가       → 짧은 패널은 13개월 창에서 4피처 추정 붕괴(NaN),
                                        긴 패널은 리프트 0.67로 실패
    · 3-state                         → 세 번째 상태가 Trend Bear가 아니라 Bull 분할
                                        (S0 breadth 0.422 vs S1 0.411, S1 조건부낙폭이 오히려 양호)

  → 경쟁을 없애려면 채널을 나눠야 한다.

      P_shock  = 2-state HSMM (breadth, newlow, trend)           ← production과 동일. crash 담당
      P_trend  = 2-state HSMM (breadth, trend, ma200_slope_60)   ← newlow 없음. veto 없음
      P_bear   = max(P_shock, P_trend)

■ 이 구조의 이점
  · veto 소멸: P_trend에 newlow가 없으니 거부권이 작동할 자리가 없다.
              crash는 P_shock이 그대로 담당하므로 압축처럼 crash를 잃지 않는다.
  · 비지도 라벨링 위험 없음: 각 채널이 2-state라 Bear 상태 의미가 사전에 정해진다
    (3-state에서 세 번째 상태가 Bull을 쪼갠 문제가 발생하지 않는다).
  · 추정 가능: 각 채널이 3피처라 13개월 첫 창에서도 production과 동일하게 돌아간다.
    (4피처 시도가 NaN으로 붕괴했던 문제가 사라진다)
  · 기여도 분리: P_shock만 / P_trend만 / max 를 나란히 비교하면 각 채널 몫이 드러난다.

■ 회귀 검증
  P_shock은 production(hsmm_final)과 설정이 완전히 동일하므로 P_bear 경로가 일치해야 한다.
  일치하지 않으면 구현 오류다.

■ 사용 / 산출
  .venv/bin/python analysis/hsmm_dualchannel.py
  analysis/results/hsmm_dual_path.csv
"""
import sys
import argparse
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent.parent
A_DIR = Path(__file__).parent
OUT = A_DIR / "results"; OUT.mkdir(exist_ok=True)
PANEL = A_DIR / ".cache" / "hsmm_features.pkl"
SLOPE_CACHE = A_DIR / ".cache" / "ma200_slope_60.pkl"
sys.path.insert(0, str(BASE))
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ModuleNotFoundError:
    pass

_sp = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_sp); sys.modules["hsmm_final"] = HF; _sp.loader.exec_module(HF)

TRAN = HF.TRAN_COLS
WINDOW_M, REFIT_EVERY, HL, EPS = HF.WINDOW_M, HF.REFIT_EVERY, HF.HL, HF.EPS
PBEAR_EMA, T_IN, T_OUT, WIN = HF.PBEAR_EMA, HF.T_IN, HF.T_OUT, HF.WIN
SLOW_A, SLOW_B = "2021-11", "2022-12"

# 채널 정의. signs는 bear_score용 — 값이 클수록 약세면 +1, 강세면 -1.
CHANNELS = {
    "shock": (["breadth", "newlow", "trend"], [-1.0, +1.0, -1.0]),
    "trend": (["breadth", "trend", "ma200_slope_60"], [-1.0, -1.0, -1.0]),
}


def load_slope(yms):
    if SLOPE_CACHE.exists():
        s = pd.read_pickle(SLOPE_CACHE)
        if set(yms) <= set(s.index):
            return s.reindex(yms)
    conn = HF._connect()
    k = pd.read_sql("SELECT period p, value::float v FROM alpha_lab.macro_indicators "
                    "WHERE indicator='kospi' AND freq='D'", conn)
    conn.close()
    k["p"] = pd.to_datetime(k.p.str.slice(0, 10))
    ks = k.set_index("p")["v"].sort_index()
    ma = ks.rolling(200, min_periods=100).mean()
    sl = ma / ma.shift(60) - 1
    mend = {}
    for d, pp in zip(ks.index, pd.PeriodIndex(ks.index, freq="M")):
        mend[pp.strftime("%Y-%m")] = d
    out = pd.Series({y: (sl[sl.index <= mend[y]].iloc[-1] if y in mend else np.nan) for y in yms}).fillna(0.0)
    SLOPE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, SLOPE_CACHE)
    return out


def walk_forward(df, yms, n, cols, signs, decide_start):
    """production walk_forward와 동일 골격. EMIS만 채널별로 교체."""
    start = yms.index(decide_start) if decide_start in yms else 12
    TRz = HF.roll_z(df, TRAN).values
    EM = df[cols].values
    stress_raw = TRz[:, 0] - TRz[:, 1]
    sg = np.asarray(signs)
    pbear = np.full(n, np.nan)
    params, sc, last_refit = None, None, -10 ** 9
    orig_bs = HF.bear_score
    try:
        HF.bear_score = lambda means: means @ sg          # 채널별 부호
        for t in range(start, n):
            lo = max(0, t + 1 - WINDOW_M); Xr = EM[lo:t + 1]; nw = len(Xr)
            w = 0.5 ** ((nw - 1 - np.arange(nw)) / HL)
            if params is None or (t - last_refit) >= REFIT_EVERY:
                sc = StandardScaler().fit(Xr)
                params = HF.fit_emission(sc.transform(Xr), w,
                                         HF.cold_emission(sc.transform(Xr)) if params is None else params,
                                         40 if params is None else 10)
                last_refit = t
            Xz = sc.transform(Xr)
            logB = HF.emis_logB(Xz, params["means"], params["covs"])
            bear = int(np.argmax(HF.bear_score(params["means"])))
            gamma, _ = HF.forward_backward(logB, params["Amat"], params["pi"], w)
            durs = HF.state_durations(gamma.argmax(1))
            haz = np.vstack([HF.to_hazard(HF.dur_pmf(durs[0])), HF.to_hazard(HF.dur_pmf(durs[1]))])
            sw = stress_raw[lo:t + 1]; sw = (sw - sw.mean()) / (sw.std() + EPS)
            filt = HF.hsmm_filter(logB, sw, haz, bear, params["pi"])
            pbear[t] = filt[-1, bear]
    finally:
        HF.bear_score = orig_bs
    return pbear, start


def ema(raw, n, start):
    out = raw.copy()
    for t in range(start + 1, n):
        out[t] = PBEAR_EMA * raw[t] + (1 - PBEAR_EMA) * out[t - 1]
    return out


def regimes(pb, n, start):
    reg = ["Bull"] * n; p = "Bull"
    for t in range(start, n):
        p = ("Bear" if pb[t] >= T_OUT else "Bull") if p == "Bear" else ("Bear" if pb[t] >= T_IN else "Bull")
        reg[t] = p
    return reg


def evaluate(name, pb, reg, yms, n, start, dd6, thr, verbose=True):
    idx = list(range(start, n))
    evs = [i for i in idx if dd6[i] <= thr and (i == start or not (dd6[i - 1] <= thr))]
    onset = [t for t in idx if reg[t] == "Bear" and reg[t - 1] != "Bear"]
    leads, nhit = [], 0
    for ev in evs:
        cand = [s for s in onset if abs(s - ev) <= WIN]
        if cand:
            nhit += 1; leads.append(ev - min(cand, key=lambda s: abs(s - ev)))
    ok = [t for t in idx if not np.isnan(dd6[t])]
    tp = sum(1 for t in ok if dd6[t] <= thr and reg[t] == "Bear")
    fn = sum(1 for t in ok if dd6[t] <= thr and reg[t] != "Bear")
    fp = sum(1 for t in ok if dd6[t] > thr and reg[t] == "Bear")
    tn = sum(1 for t in ok if dd6[t] > thr and reg[t] != "Bear")
    prec = tp / (tp + fp) if tp + fp else np.nan
    base = (tp + fn) / len(ok) if ok else np.nan
    lift = prec / base if (base and not np.isnan(prec)) else np.nan
    bdd = np.mean([dd6[t] for t in ok if reg[t] == "Bear"]) if (tp + fp) else np.nan
    udd = np.mean([dd6[t] for t in ok if reg[t] != "Bear"]) if (tn + fn) else np.nan
    disc = udd - bdd if not (np.isnan(bdd) or np.isnan(udd)) else np.nan
    sb = [t for t in idx if SLOW_A <= yms[t] <= SLOW_B]
    v = pb[sb]
    m = dict(rec=nhit / len(evs) if evs else np.nan, lead=np.mean(leads) if leads else np.nan,
             lift=lift, disc=disc, fp=fp, bear_ratio=(tp + fp) / len(ok),
             sb_mean=float(np.nanmean(v)), sb_max=float(np.nanmax(v)),
             sb_bear=sum(1 for t in sb if reg[t] == "Bear"), evs=evs, onset=onset)
    if verbose:
        print(f"\n  [{name}]")
        print(f"    리프트 {lift:.2f}배   분별력 {disc:+.1f}%p   FP {fp}   Bear비율 {m['bear_ratio']:.0%}")
        print(f"    이벤트 {len(evs)}개 중 탐지 {nhit}개   평균선행 "
              f"{(f'{np.mean(leads):+.1f}개월' if leads else '-')}")
        print(f"    ★ slow bear  평균 P {m['sb_mean']:.3f}  최대 {m['sb_max']:.2f}  Bear {m['sb_bear']}/14개월")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=-15.0)
    ap.add_argument("--long", action="store_true", help="장기 패널(2004~)로 추정 표본 확대")
    args = ap.parse_args()
    thr = -abs(args.thr)

    if args.long:
        LP = A_DIR / ".cache" / "hsmm_longrun_features.pkl"
        if not LP.exists():
            print("!! 장기 패널 없음 — analysis/hsmm_longrun.py 먼저 실행"); return
        df, yms, n, ret, _rv, _dv, dd6, _raw = pd.read_pickle(LP)
        ds = "2009-01"
    else:
        if not PANEL.exists():
            print("!! production 패널 없음 — analysis/hsmm_final.py 먼저 실행"); return
        df, yms, n, ret, rvol, dvol, dd6 = pd.read_pickle(PANEL)
        ds = HF.DECIDE_START
    df = df.copy()
    df["ma200_slope_60"] = load_slope(yms).values
    print(f"패널 {yms[0]}~{yms[-1]} ({n}개월)  판정 {ds}  production 패널·초기화 그대로")
    for k, (c, s) in CHANNELS.items():
        print(f"  채널 {k:6} EMIS={c}  signs={s}")

    P = {}
    for k, (cols, signs) in CHANNELS.items():
        print(f"\n  [{k}] walk-forward...", flush=True)
        raw, start = walk_forward(df, yms, n, cols, signs, ds)
        P[k] = ema(raw, n, start)
    P["max"] = np.fmax(P["shock"], P["trend"])

    # ── 회귀 검증: shock 채널 = production ──
    if args.long:
        print("\n  (장기 패널 — production 회귀검증 생략)")
    else:
     prod = pd.read_csv(A_DIR / "hsmm_final_path.csv", encoding="utf-8-sig").set_index("ym")["pbear"]
     aligned = pd.Series(P["shock"], index=yms).reindex(prod.index)
     d = float(np.nanmax(np.abs(aligned.values - prod.values)))
     print(f"\n  회귀 검증  shock 채널 vs production P_bear  최대 절대차 {d:.2e}"
           f"   {'✓ 일치' if d < 1e-9 else '✗ 불일치'}")

    print(f"\n{'='*76}\n  탐지기 평가 (이벤트 = 향후6M 낙폭 <= {thr:.0f}%)\n{'='*76}")
    S = {}
    for k in ["shock", "trend", "max"]:
        S[k] = evaluate(k, P[k], regimes(P[k], n, start), yms, n, start, dd6, thr)

    print(f"\n{'='*76}\n  요약\n{'='*76}")
    print(f"  {'지표':20}{'shock(=현행)':>15}{'trend':>12}{'max':>12}")
    for key, lab, fmt in [("sb_mean", "★slow bear 평균P", "{:.3f}"), ("sb_bear", "★slow bear Bear월", "{:d}"),
                          ("lift", "★리프트", "{:.2f}배"), ("disc", "분별력", "{:+.1f}%p"),
                          ("lead", "평균선행(월)", "{:+.1f}"), ("rec", "이벤트Recall", "{:.0%}"),
                          ("fp", "False Positive", "{:d}"), ("bear_ratio", "Bear비율", "{:.0%}")]:
        cells = ""
        for k in ["shock", "trend", "max"]:
            v = S[k][key]
            cells += f"{'  -  ':>12}" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{fmt.format(v):>12}"
        print(f"  {lab:20}{'':>3}{cells}")

    print("\n  이벤트별 선행(+빠름):")
    NAMES = {"2018-10": "2018 하락", "2020-02": "2020 코로나", "2021-11": "2021~22 slow",
             "2024-08": "2024 급락", "2025-09": "2025 하락"}
    for ev in S["shock"]["evs"]:
        cells = ""
        for k in ["shock", "trend", "max"]:
            cand = [o for o in S[k]["onset"] if abs(o - ev) <= WIN]
            cells += f"{(f'{ev-min(cand,key=lambda o:abs(o-ev)):+d}' if cand else 'X'):>12}"
        print(f"    {NAMES.get(yms[ev], yms[ev]):16}{dd6[ev]:>6.0f}%{cells}")

    print(f"\n{'='*76}\n  2021-07 ~ 2023-03 월별\n{'='*76}")
    print(f"  {'ym':9}{'P_shock':>9}{'P_trend':>9}{'P_max':>8}{'breadth':>9}{'newlow':>8}"
          f"{'trend':>8}{'slope':>8}  판정(max)")
    rmax = regimes(P["max"], n, start)
    for i, y in enumerate(yms):
        if not ("2021-07" <= y <= "2023-03"):
            continue
        print(f"  {y:9}{P['shock'][i]:9.2f}{P['trend'][i]:9.2f}{P['max'][i]:8.2f}"
              f"{df.breadth.iloc[i]:9.3f}{df.newlow.iloc[i]:8.3f}{df.trend.iloc[i]:8.3f}"
              f"{df.ma200_slope_60.iloc[i]:8.3f}  {rmax[i]}")

    rows = []
    for i in range(n):
        rows.append(dict(ym=yms[i], P_shock=P["shock"][i], P_trend=P["trend"][i], P_max=P["max"][i],
                         regime_max=rmax[i], ret=ret[i], dd6=dd6[i],
                         breadth=df.breadth.iloc[i], newlow=df.newlow.iloc[i],
                         trend=df.trend.iloc[i], slope=df.ma200_slope_60.iloc[i]))
    pd.DataFrame(rows).to_csv(OUT / "hsmm_dual_path.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT/'hsmm_dual_path.csv'}")
    print("\n※ production 미수정. 채택 판단은 2022뿐 아니라 리프트·FP·crash 유지까지 함께 볼 것.")


if __name__ == "__main__":
    main()
