"""
analysis/diag_new_axis.py

새 '분리축' 후보 선별 진단 — 모델 변경 없음. production 미수정.

■ 왜 필요한가 (확정된 근본 원인)
  현재 emission [breadth, newlow, trend]에서 newlow가 두 역할을 겸한다:
    ① 2-state를 식별시키는 **유일한 분리축**(첨도 18.7의 스파이크형)
    ② 그 축이 조용한 국면(slow bear)을 5.6σ로 배제하는 **veto**
  하나를 얻으면 다른 하나를 잃어, 조합·변환·상태수 변경으로는 탈출 불가.
  실제로 newlow를 뺀 채널은 상태가 완전히 붕괴했다(2019-01부터 두 상태 평균 동일).

■ 그래서 찾는 것
  slow bear에서 약세이면서 **상태를 만들 만한 분포적 분리력**이 있고,
  **newlow처럼 극단적이지는 않은**(veto가 되지 않는) 피처.

  실패한 후보(전부 완만·고상관): ma200_slope_60, breadth MA6, trend 음수지속, 고점대비낙폭
  → 완만한 축은 분리를 못 만든다. 필요한 건 '레짐처럼 수준이 갈리되 꼬리는 얇은' 축.

■ 후보 (전부 기존 daily_price로 계산. 신규 수집 없음)
  newhigh   52주 신고가 비율        — newlow의 대칭. 약세장엔 0에 붙어 '지속'한다(비대칭)
  dd_med    횡단면 52주 고점대비 낙폭 중앙값 — 지속형, 레짐성
  dd_q25    같은 것의 하위 25분위    — 꼬리 쪽 관찰
  disp      횡단면 월수익률 표준편차  — 스트레스 국면에 확대
  turn      거래대금/시가총액 회전율  — slow bear의 거래 위축
  ew_vw     동일가중 − 시총가중 월수익 — 참여 폭(소형주 소외)

■ 선별 기준 (성과 최적화 아님. 구조적 요건)
  1) 분리력  : 1-D 2-component GMM 적합 → 두 성분 평균차 / 통합σ, 소수 성분 비중
  2) 꼬리    : 첨도. newlow(18.7)급이면 veto 재발 위험
  3) 지속성  : lag-1 자기상관 (slow bear는 지속형이므로 필요)
  4) 신호    : slow bear 구간 z, 코로나 구간 z (둘 다 반응하면 이상적)
  5) 중복    : 기존 breadth/newlow/trend와의 상관

■ 사용 / 산출
  .venv/bin/python analysis/diag_new_axis.py            (첫 실행 DB ~2분, 이후 캐시)
  옵션 --refresh
  analysis/results/diag_new_axis.csv
"""
import os
import sys
import warnings
import importlib.util
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent.parent
A_DIR = Path(__file__).parent
OUT = A_DIR / "results"; OUT.mkdir(exist_ok=True)
CACHE = A_DIR / ".cache" / "new_axis_candidates.pkl"
sys.path.insert(0, str(BASE))
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ModuleNotFoundError:
    pass

_sp = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_sp); sys.modules["hsmm_final"] = HF; _sp.loader.exec_module(HF)

SLOW_A, SLOW_B = "2021-11", "2022-12"
COVID_A, COVID_B = "2020-01", "2020-03"
DS = HF.DECIDE_START


def build():
    if "--refresh" not in sys.argv and CACHE.exists():
        print(f"[cache] 후보 재사용: {CACHE.relative_to(BASE)}", flush=True)
        return pd.read_pickle(CACHE)

    print("[db] daily_price 로드 (~2분)...", flush=True)
    conn = HF._connect()
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close, "
                    "trade_amount::float amt, market_cap::float mcap "
                    "FROM alpha_lab.daily_price WHERE close IS NOT NULL AND trade_date>='2017-01-01'", conn)
    snap = pd.read_sql("SELECT snapshot_date ym, stock_code FROM alpha_lab.fnspace_master "
                       "WHERE market='KOSPI' AND sec_cd_nm IS NOT NULL", conn)
    conn.close()
    d["dt"] = pd.to_datetime(d["dt"])
    close = d.pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    amt = d.pivot_table(index="dt", columns="stock_code", values="amt").sort_index()
    mcap = d.pivot_table(index="dt", columns="stock_code", values="mcap").sort_index()

    snap["code"] = snap.stock_code.str[1:]
    snap = snap[snap.code.str.match(r"^\d{5}0$")]
    by_ym = snap.groupby("ym")["code"].apply(set).to_dict(); yms_snap = sorted(by_ym)
    K = pd.DataFrame(False, index=close.index, columns=close.columns)
    day_ym = close.index.to_period("M").strftime("%Y-%m")
    for _ym in np.unique(day_ym):
        avail = [s for s in yms_snap if s <= _ym]
        if avail:
            K.loc[day_ym == _ym, close.columns.isin(by_ym[avail[-1]])] = True
    ok = close.notna() & K
    den = ok.sum(axis=1).clip(lower=1)

    print("후보 계산...", flush=True)
    rmax = close.rolling(252, min_periods=60).max()
    rmin = close.rolling(252, min_periods=60).min()
    newhigh = ((close >= rmax) & ok).sum(axis=1) / den            # 52주 신고가 비율
    dd = close / rmax - 1                                          # 종목별 고점대비 낙폭
    dd_med = dd.where(ok).median(axis=1)
    dd_q25 = dd.where(ok).quantile(0.25, axis=1)
    turn = (amt.where(ok).sum(axis=1)) / (mcap.where(ok).sum(axis=1))   # 회전율

    mend_all = close.groupby(close.index.to_period("M")).tail(1).index
    mret = close.loc[mend_all].pct_change()
    okm = ok.loc[mend_all]
    disp_m = mret.where(okm).std(axis=1)                           # 횡단면 수익률 σ
    w = mcap.loc[mend_all].where(okm)
    vw = (mret.where(okm) * w).sum(axis=1) / w.sum(axis=1)
    ew = mret.where(okm).mean(axis=1)
    ewvw_m = ew - vw

    # 월말 기준 정렬 (production build_features와 동일 정의)
    conn = HF._connect()
    kk = pd.read_sql("SELECT period p, value::float v FROM alpha_lab.macro_indicators "
                     "WHERE indicator='kospi' AND freq='D'", conn)
    conn.close()
    kk["p"] = pd.to_datetime(kk.p.str.slice(0, 10))
    kospi = kk.set_index("p")["v"].sort_index()
    last = {}
    for dt_, p in zip(kospi.index, pd.PeriodIndex(kospi.index, freq="M")):
        last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= pd.Timestamp(HF.PANEL_START)]

    def asof(s, e):
        x = s[s.index <= e]
        return x.iloc[-1] if len(x) else np.nan

    rows = []
    for e in mends:
        rows.append(dict(newhigh=asof(newhigh, e), dd_med=asof(dd_med, e), dd_q25=asof(dd_q25, e),
                         turn=asof(turn, e), disp=asof(disp_m, e), ew_vw=asof(ewvw_m, e)))
    idx = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    C = pd.DataFrame(rows, index=idx)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(C, CACHE)
    print(f"[cache] 저장: {CACHE.relative_to(BASE)}", flush=True)
    return C


def sep_power(x):
    """1-D 2-component GMM 분리력. |μ1-μ2|/통합σ 와 소수 성분 비중."""
    z = ((x - x.mean()) / (x.std() + 1e-12)).values.reshape(-1, 1)
    g = GaussianMixture(2, covariance_type="full", random_state=42, n_init=5).fit(z)
    m = g.means_.ravel(); v = g.covariances_.ravel()
    sep = abs(m[0] - m[1]) / np.sqrt((v[0] + v[1]) / 2)
    return sep, float(min(g.weights_))


def main():
    C = build()
    df, yms, n, ret, rvol, dvol, dd6 = pd.read_pickle(A_DIR / ".cache" / "hsmm_features.pkl")
    base = df[["breadth", "newlow", "trend"]].copy()
    base.index = C.index if len(C) == len(df) else base.index
    X = pd.concat([base, C], axis=1).loc[DS:]
    D6 = pd.Series(dd6, index=yms).loc[DS:]
    risk = D6 <= -15

    print(f"\n패널 {X.index[0]}~{X.index[-1]} ({len(X)}개월)")
    print(f"\n{'='*104}")
    print(f"  {'피처':10}{'분리력':>8}{'소수비중':>9}{'첨도':>8}{'자기상관':>9}"
          f"{'slowbear z':>12}{'코로나 z':>10}{'위험월 격차':>12}   기존 최대상관")
    print("=" * 104)
    rows = []
    for c in X.columns:
        s = X[c].astype(float)
        if s.std() < 1e-12:
            continue
        sep, wmin = sep_power(s)
        kurt = float(s.kurt())
        ac = float(s.autocorr(1))
        sb = (s.loc[SLOW_A:SLOW_B].mean() - s.mean()) / s.std()
        cv = (s.loc[COVID_A:COVID_B].mean() - s.mean()) / s.std()
        gap = (s[risk].mean() - s[~risk].mean()) / s.std()
        others = [o for o in ["breadth", "newlow", "trend"] if o != c]
        mx = max((abs(s.corr(X[o].astype(float))) for o in others), default=np.nan)
        tag = "  ← 기존" if c in ("breadth", "newlow", "trend") else ""
        print(f"  {c:10}{sep:8.2f}{wmin:9.2f}{kurt:8.1f}{ac:9.2f}{sb:+12.2f}{cv:+10.2f}"
              f"{gap:+12.2f}{mx:>10.2f}{tag}")
        rows.append(dict(feature=c, separation=sep, minor_w=wmin, kurt_v=kurt, autocorr=ac,
                         slowbear_z=sb, covid_z=cv, risk_gap=gap, max_corr_base=mx))
    R = pd.DataFrame(rows)

    print(f"\n{'='*104}\n  선별 — 구조적 요건 충족 여부\n{'='*104}")
    nl = R[R.feature == "newlow"].iloc[0]
    print(f"  기준점: newlow  분리력 {nl.separation:.2f}  첨도 {nl.kurt_v:.1f}(veto 원인)  "
          f"자기상관 {nl.autocorr:.2f}  slowbear z {nl.slowbear_z:+.2f}")
    print(f"  요건: ① 분리력 >= 1.5 (상태 식별 가능)  ② 첨도 <= 5 (veto 회피)")
    print(f"        ③ 자기상관 >= 0.5 (지속형)        ④ slowbear z 절대값 >= 0.5 (2022에 반응)")
    cand = R[(R.separation >= 1.5) & (R.kurt_v <= 5) & (R.autocorr >= 0.5)
             & (R.slowbear_z.abs() >= 0.5) & (~R.feature.isin(["breadth", "newlow", "trend"]))]
    if len(cand):
        print(f"\n  ★ 4요건 통과 후보:")
        print(cand[["feature", "separation", "kurt_v", "autocorr", "slowbear_z",
                    "covid_z", "max_corr_base"]].round(2).to_string(index=False))
    else:
        print("\n  4요건 동시 충족 후보 없음 — 요건별로 어디서 걸리는지 확인:")
        for _, r in R[~R.feature.isin(["breadth", "newlow", "trend"])].iterrows():
            fail = []
            if r.separation < 1.5: fail.append(f"분리력 {r.separation:.2f}")
            if r.kurt_v > 5: fail.append(f"첨도 {r.kurt_v:.1f}")
            if r.autocorr < 0.5: fail.append(f"자기상관 {r.autocorr:.2f}")
            if abs(r.slowbear_z) < 0.5: fail.append(f"slowbear {r.slowbear_z:+.2f}")
            print(f"    {r.feature:10} {'통과' if not fail else ' / '.join(fail)}")

    print(f"\n  2021-07~2023-03 후보 경로")
    cols = [c for c in C.columns]
    print(f"  {'ym':9}" + "".join(f"{c:>10}" for c in cols))
    for y in [v for v in X.index if "2021-07" <= v <= "2023-03"]:
        print(f"  {y:9}" + "".join(f"{X[c][y]:>10.3f}" for c in cols))

    R.to_csv(OUT / "diag_new_axis.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT/'diag_new_axis.csv'}")
    print("\n※ 선별 진단 전용. 모델 변경·채택 없음. production 미수정.")


if __name__ == "__main__":
    main()
