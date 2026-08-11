"""
analysis/diag_macro_axis.py

매크로 '분리축' 후보 선별 — 금리 스프레드/신용 스프레드. 모델 변경 없음. production 미수정.

■ 왜 매크로인가
  가격 파생 후보(diag_new_axis.py)는 전부 breadth와 중복이었다:
    dd_q25  분리력 2.82 · 첨도 -0.2 · 자기상관 0.80 · slowbear -1.01  → **그러나 breadth와 상관 0.91**
    dd_med  분리력 1.63                                              → 상관 0.92
    turn    상관 0.39(독립적)                                        → 그러나 slowbear -0.17(무반응)
  같은 종가 행렬에서 나오니 당연하다. 새 정보는 시장 밖에서 와야 한다.

■ 기존 이력과의 관계 (중요)
  [[project_regime_macro_features]]에 "금리 무효 확정"이 있으나, 그건 **미국채 10년 레벨**이다.
  여기서 보는 것은 **스프레드**로 레벨과 정보가 다르다(신용 사이클·기간 프리미엄).
  2022 slow bear의 실제 서사가 긴축 + 레고랜드 신용경색이므로 직접 대응하는 축이다.
  환율(Δusd_krw 3m)은 이미 transition에 채택돼 있으므로 중복 확인만 한다.

■ 후보 (FnSpace EconomyApi — 코드 실측 확인 완료)
  arKOIRKSDATB1/TB3/TB10  국고채 1·3·10년 (전부 2017-01-02~, 장기패널 불가)
  arKOIRKSDACD                       CD 금리
  arKOIRKSDACP                       CP 금리
    term_10_3  = TB10 - TB3    장단기 금리차(플래트닝/역전 = 침체 선행)
    term_3_1   = TB3  - TB1
    credit_cp_cd = CP - CD     **단기 신용경색** (기업어음 vs 은행). 2022-10 레고랜드 직격
    credit_cp_tb = CP - TB3    기업 조달 스프레드
    bank_cd_tb   = CD - TB1    은행 조달 스프레드
  레벨은 이미 무효로 확정됐으므로 Δ3m(3개월 변화)도 함께 본다.

■ 선별 기준 (diag_new_axis.py와 동일. 성과 최적화 아님)
  ① 분리력 >= 1.5 (1-D 2-component GMM 평균차/통합σ)   ② 첨도 <= 5 (newlow 18.7식 veto 회피)
  ③ 자기상관 >= 0.5 (지속형)                            ④ |slowbear z| >= 0.5
  ⑤ 기존 breadth/newlow/trend와 상관이 낮을수록 좋음(새 정보)

■ 사용 / 산출
  .venv/bin/python analysis/diag_macro_axis.py     (첫 실행 API 호출, 이후 캐시)
  옵션 --refresh
  analysis/results/diag_macro_axis.csv
"""
import os
import sys
import time
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent.parent
A_DIR = Path(__file__).parent
OUT = A_DIR / "results"; OUT.mkdir(exist_ok=True)
CACHE = A_DIR / ".cache" / "macro_rates.pkl"
sys.path.insert(0, str(BASE))
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ModuleNotFoundError:
    pass

_sp = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_sp); sys.modules["hsmm_final"] = HF; _sp.loader.exec_module(HF)

URL = "https://www.fnspace.com/Api/EconomyApi"
KEY = os.environ.get("FNSPACE_API_KEY", "D0E7A9A250B8C43545C5")
SERIES = {"TB1": "arKOIRKSDATB1", "TB3": "arKOIRKSDATB3", "TB10": "arKOIRKSDATB10",
          "CD": "arKOIRKSDACD", "CP": "arKOIRKSDACP"}
SLOW_A, SLOW_B = "2021-11", "2022-12"
COVID_A, COVID_B = "2020-01", "2020-03"
DS = HF.DECIDE_START


def fetch(code, fr="20170101", to="20260731"):
    r = requests.get(URL, params={"key": KEY, "format": "json", "item": code,
                                  "frdate": fr, "todate": to}, timeout=60)
    d = r.json()
    data = d.get("dataset", [{}])[0].get("DATA", []) if d.get("dataset") else []
    if not data:
        raise RuntimeError(f"{code}: {d.get('errmsg','데이터 없음')[:40]}")
    s = pd.Series({pd.Timestamp(f"{x['DT'][:4]}-{x['DT'][4:6]}-{x['DT'][6:8]}"): float(x["AMOUNT"])
                   for x in data if x.get("AMOUNT") not in (None, "")}).sort_index()
    return s


def load_rates():
    if "--refresh" not in sys.argv and CACHE.exists():
        print(f"[cache] 금리 재사용: {CACHE.relative_to(BASE)}", flush=True)
        return pd.read_pickle(CACHE)
    out = {}
    for k, code in SERIES.items():
        s = fetch(code)
        out[k] = s
        print(f"  {k:5} {code:16} {len(s):5d}건  {s.index[0]:%Y-%m-%d} ~ {s.index[-1]:%Y-%m-%d}"
              f"   ({s.iloc[0]:.2f} → {s.iloc[-1]:.2f})", flush=True)
        time.sleep(0.3)
    R = pd.DataFrame(out).sort_index()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(R, CACHE)
    return R


def sep_power(x):
    z = ((x - x.mean()) / (x.std() + 1e-12)).values.reshape(-1, 1)
    g = GaussianMixture(2, covariance_type="full", random_state=42, n_init=5).fit(z)
    m = g.means_.ravel(); v = g.covariances_.ravel()
    return abs(m[0] - m[1]) / np.sqrt((v[0] + v[1]) / 2), float(min(g.weights_))


def main():
    print("[FnSpace] 금리 시계열 로드...", flush=True)
    R = load_rates()

    # 스프레드 (일별) → 월말 as-of
    S = pd.DataFrame(index=R.index)
    S["term_10_3"] = R.TB10 - R.TB3
    S["term_3_1"] = R.TB3 - R.TB1
    S["credit_cp_cd"] = R.CP - R.CD
    S["credit_cp_tb"] = R.CP - R.TB3
    S["bank_cd_tb"] = R.CD - R.TB1
    S = S.ffill()

    df, yms, n, ret, rvol, dvol, dd6 = pd.read_pickle(A_DIR / ".cache" / "hsmm_features.pkl")
    conn = HF._connect()
    kk = pd.read_sql("SELECT period p, value::float v FROM alpha_lab.macro_indicators "
                     "WHERE indicator='kospi' AND freq='D'", conn)
    conn.close()
    kk["p"] = pd.to_datetime(kk.p.str.slice(0, 10))
    kospi = kk.set_index("p")["v"].sort_index()
    last = {}
    for dt_, p in zip(kospi.index, pd.PeriodIndex(kospi.index, freq="M")):
        last[p] = dt_
    mend = {pd.Period(v, freq="M").strftime("%Y-%m"): v for v in last.values()}

    rows = {}
    for y in yms:
        e = mend.get(y)
        if e is None:
            continue
        x = S[S.index <= e]
        rows[y] = x.iloc[-1] if len(x) else pd.Series(np.nan, index=S.columns)
    M = pd.DataFrame(rows).T.reindex(yms)
    # 레벨은 이미 무효 확정 → Δ3m 도 함께
    for c in list(M.columns):
        M[c + "_d3m"] = M[c] - M[c].shift(3)
    M = M.astype(float)

    base = df[["breadth", "newlow", "trend"]].copy()
    base.index = yms
    X = pd.concat([base, M], axis=1).loc[DS:].dropna(axis=1, how="all")
    D6 = pd.Series(dd6, index=yms).loc[DS:]
    risk = D6 <= -15
    cov = R.index[0]
    print(f"\n금리 커버리지 시작 {cov:%Y-%m}   패널 {X.index[0]}~{X.index[-1]} ({len(X)}개월)")

    print(f"\n{'='*106}")
    print(f"  {'피처':16}{'분리력':>8}{'첨도':>8}{'자기상관':>9}{'slowbear z':>12}"
          f"{'코로나 z':>10}{'위험월격차':>11}{'기존상관':>9}   판정")
    print("=" * 106)
    out = []
    for c in X.columns:
        s = X[c]
        if s.isna().all() or s.std() < 1e-12:
            continue
        s = s.fillna(method="ffill").fillna(0.0)
        sep, _w = sep_power(s)
        kt = float(s.kurt()); ac = float(s.autocorr(1))
        sb = (s.loc[SLOW_A:SLOW_B].mean() - s.mean()) / s.std()
        cv = (s.loc[COVID_A:COVID_B].mean() - s.mean()) / s.std()
        gap = (s[risk].mean() - s[~risk].mean()) / s.std()
        mx = max(abs(s.corr(X[o].astype(float))) for o in ["breadth", "newlow", "trend"] if o != c)
        is_base = c in ("breadth", "newlow", "trend")
        fails = []
        if sep < 1.5: fails.append("분리력")
        if kt > 5: fails.append("첨도")
        if ac < 0.5: fails.append("지속성")
        if abs(sb) < 0.5: fails.append("slowbear")
        verdict = "← 기존" if is_base else ("★ 통과" if not fails else "／".join(fails))
        print(f"  {c:16}{sep:8.2f}{kt:8.1f}{ac:9.2f}{sb:+12.2f}{cv:+10.2f}{gap:+11.2f}"
              f"{mx:>9.2f}   {verdict}")
        out.append(dict(feature=c, separation=sep, kurt_v=kt, autocorr=ac, slowbear_z=sb,
                        covid_z=cv, risk_gap=gap, corr_base=mx, base=is_base,
                        pass_all=(not fails and not is_base)))
    O = pd.DataFrame(out)

    win = O[O.pass_all]
    print(f"\n{'='*106}")
    if len(win):
        print("  ★ 4요건 통과 — 기존 상관이 낮은 순")
        print(win.sort_values("corr_base")[["feature", "separation", "kurt_v", "autocorr",
                                            "slowbear_z", "covid_z", "corr_base"]]
              .round(2).to_string(index=False))
        print("\n  ※ 통과했다고 채택이 아니다. 다음 단계에서 emission에 넣어 crash 유지·리프트를 확인해야 한다.")
    else:
        print("  4요건 통과 후보 없음 — 매크로 스프레드도 분리축이 되지 못함")

    print(f"\n  2021-07 ~ 2023-03 스프레드 경로 (%p)")
    cols = [c for c in M.columns if not c.endswith("_d3m")]
    print(f"  {'ym':9}" + "".join(f"{c:>15}" for c in cols))
    for y in [v for v in X.index if "2021-07" <= v <= "2023-03"]:
        print(f"  {y:9}" + "".join(f"{X[c][y]:>15.3f}" for c in cols))

    O.to_csv(OUT / "diag_macro_axis.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT/'diag_macro_axis.csv'}")
    print("\n※ 선별 진단 전용. 모델 변경·채택 없음. production 미수정.")


if __name__ == "__main__":
    main()
