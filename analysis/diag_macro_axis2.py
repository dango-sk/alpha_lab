"""
analysis/diag_macro_axis2.py

매크로 분리축 2차 탐색 — DB에 이미 있는 실물·심리·글로벌·수급 계열. production 미수정.

■ 1차(금리 스프레드)에서 배운 것
  · credit_cp_tb : 만기 불일치(CP 91일 vs TB3 3년)로 **기간구조 오염**. 2022 상반기 음수는
                   신용이 아니라 곡선 스티프닝(TB3-TB1 0.85~0.99). 같은 구간에서 부호 반전 → 기각
  · term_3_1     : 스티프닝(긴축)↑ / 역전(침체)↓ **양쪽 다 약세** = U자 비단조 → 선형 부호 불가
  · credit_cp_cd : 만기 정합이라 지표로는 옳으나 **후행**(KOSPI 저점 2022-09, 스프레드 정점 2022-11)
  → 이번엔 **선행성**과 **단조성**을 처음부터 요건에 넣는다.

■ 후보 (전부 alpha_lab.macro_indicators에 이미 적재됨. 신규 수집 없음)
  장기(1996~2000~) — 장기 패널에도 사용 가능
    sox_3m      필라델피아 반도체지수 3개월 수익률   한국 수출·KOSPI 선행
    sox_rs      sox/kospi 상대강도 3개월            반도체 사이클 대비 국내
    sp500_3m    S&P500 3개월 수익률                 글로벌 위험선호
    vix         VIX 레벨
    indiv_3m    개인 순매수 3개월 누적              slow bear의 '물타기' 수급
    inst_3m     기관 순매수 3개월 누적
    ind_frn     개인 − 외국인 (수급 괴리)           하락장 특유의 수급 구조
  월별(2017~) — production 패널만. **발표 지연 반영**(regime_agent_multimodel.LAG 규칙)
    leading_index(2M) / bsi_all(2M) / csi_outlook(1M) / cpi(1M) / ppi(2M)
    레벨과 함께 전년동월비·3개월 변화도 본다.

■ 선별 요건 (1차와 동일 + 선행성·단조성 추가)
  ① 분리력 >= 1.5   ② 첨도 <= 5   ③ 자기상관 >= 0.5   ④ |slowbear z| >= 0.5
  ⑤ 기존 breadth/newlow/trend 상관 낮을수록 좋음
  ⑥ ★선행성 : slow bear 시작(2021-11) '이전 3개월'에 이미 신호가 있었나
  ⑦ ★단조성 : slow bear 전반부(2021-11~2022-05) 와 후반부(2022-06~12) 의 부호가 같은가
               (credit_cp_tb처럼 구간 내 반전하면 탈락)

■ 사용 / 산출
  .venv/bin/python analysis/diag_macro_axis2.py
  analysis/results/diag_macro_axis2.csv
"""
import os
import sys
import warnings
import importlib.util
from pathlib import Path

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
sys.path.insert(0, str(BASE))
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ModuleNotFoundError:
    pass

_sp = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_sp); sys.modules["hsmm_final"] = HF; _sp.loader.exec_module(HF)

SLOW_A, SLOW_B = "2021-11", "2022-12"
SLOW_MID = "2022-05"
PRE_A, PRE_B = "2021-08", "2021-10"     # slow bear 직전 3개월 (선행성 판정)
COVID_A, COVID_B = "2020-01", "2020-03"
DS = HF.DECIDE_START
LAG_M = {"leading_index": 2, "bsi_all": 2, "csi_outlook": 1, "cpi": 1, "ppi": 2}


def load_macro():
    conn = HF._connect()
    d = pd.read_sql("SELECT indicator, period, freq, value::float v FROM alpha_lab.macro_indicators", conn)
    conn.close()
    D, M = {}, {}
    for ind, g in d.groupby("indicator"):
        if g.freq.iloc[0] == "D":
            s = g.copy(); s["p"] = pd.to_datetime(s.period.str.slice(0, 10))
            D[ind] = s.set_index("p")["v"].sort_index()
        else:
            s = g.copy()
            s["p"] = s.period.str.slice(0, 4) + "-" + s.period.str.slice(4, 6)
            M[ind] = s.set_index("p")["v"].sort_index()
    return D, M


def sep_power(x):
    z = ((x - x.mean()) / (x.std() + 1e-12)).values.reshape(-1, 1)
    g = GaussianMixture(2, covariance_type="full", random_state=42, n_init=5).fit(z)
    m = g.means_.ravel(); v = g.covariances_.ravel()
    return abs(m[0] - m[1]) / np.sqrt((v[0] + v[1]) / 2)


def main():
    print("[db] 매크로 로드...", flush=True)
    D, M = load_macro()
    df, yms, n, ret, rvol, dvol, dd6 = pd.read_pickle(A_DIR / ".cache" / "hsmm_features.pkl")

    kospi = D["kospi"]
    last = {}
    for dt_, p in zip(kospi.index, pd.PeriodIndex(kospi.index, freq="M")):
        last[p.strftime("%Y-%m")] = dt_

    def asof_d(s, ym):
        e = last.get(ym)
        if e is None:
            return np.nan
        x = s[s.index <= e]
        return float(x.iloc[-1]) if len(x) else np.nan

    def mret(s, ym, months):
        """월말 기준 n개월 수익률 (causal)"""
        e = last.get(ym)
        if e is None:
            return np.nan
        cur = s[s.index <= e]
        prv = s[s.index <= e - pd.DateOffset(months=months)]
        if not len(cur) or not len(prv) or prv.iloc[-1] == 0:
            return np.nan
        return float(cur.iloc[-1] / prv.iloc[-1] - 1)

    def msum(s, ym, days):
        e = last.get(ym)
        if e is None:
            return np.nan
        x = s[(s.index <= e) & (s.index > e - pd.Timedelta(days=days))]
        return float(x.sum()) if len(x) else np.nan

    C = {}
    C["sox_3m"] = [mret(D["sox"], y, 3) for y in yms]
    C["sp500_3m"] = [mret(D["sp500"], y, 3) for y in yms]
    C["vix"] = [asof_d(D["vix"], y) for y in yms]
    C["indiv_3m"] = [msum(D["investor_individual_kospi"], y, 90) for y in yms]
    C["inst_3m"] = [msum(D["investor_institution_kospi"], y, 90) for y in yms]
    C["frn_3m"] = [msum(D["investor_foreign_kospi"], y, 90) for y in yms]
    sox3 = np.array(C["sox_3m"], dtype=float)
    ks3 = np.array([mret(kospi, y, 3) for y in yms], dtype=float)
    C["sox_rs"] = (sox3 - ks3).tolist()
    C["ind_frn"] = (np.array(C["indiv_3m"], dtype=float) - np.array(C["frn_3m"], dtype=float)).tolist()

    # 월별 지표 — 발표 지연 반영
    for k, lag in LAG_M.items():
        if k not in M:
            continue
        s = M[k]
        vals, yoy, d3 = [], [], []
        for y in yms:
            p = (pd.Period(y, freq="M") - lag).strftime("%Y-%m")
            p12 = (pd.Period(y, freq="M") - lag - 12).strftime("%Y-%m")
            p3 = (pd.Period(y, freq="M") - lag - 3).strftime("%Y-%m")
            v = s.get(p, np.nan); v12 = s.get(p12, np.nan); v3 = s.get(p3, np.nan)
            vals.append(v)
            yoy.append(v / v12 - 1 if (v12 and not np.isnan(v12) and v12 != 0) else np.nan)
            d3.append(v - v3 if not (np.isnan(v) or np.isnan(v3)) else np.nan)
        C[k] = vals; C[k + "_yoy"] = yoy; C[k + "_d3"] = d3

    X = pd.DataFrame(C, index=yms)
    base = df[["breadth", "newlow", "trend"]].copy(); base.index = yms
    X = pd.concat([base, X], axis=1).loc[DS:]
    D6 = pd.Series(dd6, index=yms).loc[DS:]
    risk = D6 <= -15

    print(f"\n패널 {X.index[0]}~{X.index[-1]} ({len(X)}개월)   월별지표는 발표지연 {LAG_M} 반영")
    print(f"\n{'='*118}")
    print(f"  {'피처':18}{'분리력':>7}{'첨도':>7}{'자기상관':>8}{'slow z':>9}{'선행 z':>8}"
          f"{'전반':>7}{'후반':>7}{'코로나':>8}{'기존상관':>9}   판정")
    print("=" * 118)
    rows = []
    for c in X.columns:
        s = X[c].astype(float)
        if s.isna().sum() > len(s) * 0.3 or s.std() < 1e-12:
            continue
        s = s.ffill().bfill()
        sep = sep_power(s); kt = float(s.kurt()); ac = float(s.autocorr(1))
        z = lambda a, b: (s.loc[a:b].mean() - s.mean()) / s.std()
        sb, pre = z(SLOW_A, SLOW_B), z(PRE_A, PRE_B)
        h1, h2 = z(SLOW_A, SLOW_MID), z("2022-06", SLOW_B)
        cv = z(COVID_A, COVID_B)
        mx = max(abs(s.corr(X[o].astype(float))) for o in ["breadth", "newlow", "trend"] if o != c)
        is_base = c in ("breadth", "newlow", "trend")
        f = []
        if sep < 1.5: f.append("분리력")
        if kt > 5: f.append("첨도")
        if ac < 0.5: f.append("지속성")
        if abs(sb) < 0.5: f.append("slowbear")
        if abs(pre) < 0.3: f.append("선행성")
        if h1 * h2 < 0: f.append("부호반전")
        v = "← 기존" if is_base else ("★ 통과" if not f else "／".join(f))
        print(f"  {c:18}{sep:7.2f}{kt:7.1f}{ac:8.2f}{sb:+9.2f}{pre:+8.2f}{h1:+7.2f}{h2:+7.2f}"
              f"{cv:+8.2f}{mx:>9.2f}   {v}")
        rows.append(dict(feature=c, sep=sep, kurt_v=kt, ac=ac, slow_z=sb, pre_z=pre,
                         h1=h1, h2=h2, covid_z=cv, corr_base=mx, base=is_base,
                         pass_all=(not f and not is_base)))
    R = pd.DataFrame(rows)
    win = R[R.pass_all].sort_values("corr_base")
    print(f"\n{'='*118}")
    if len(win):
        print("  ★ 6요건 통과 (분리력·첨도·지속성·slowbear·선행성·단조성) — 기존 상관 낮은 순")
        print(win[["feature", "sep", "kurt_v", "ac", "slow_z", "pre_z", "h1", "h2",
                   "covid_z", "corr_base"]].round(2).to_string(index=False))
    else:
        print("  6요건 통과 없음 — 요건을 하나씩 완화해 무엇이 걸리는지 확인 필요")
    R.to_csv(OUT / "diag_macro_axis2.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT/'diag_macro_axis2.csv'}")
    print("\n※ 선별 진단 전용. 모델 변경·채택 없음. production 미수정.")
    print("※ 통과해도 다음 단계에서 emission 투입 후 crash 유지·리프트를 확인해야 채택 가능.")


if __name__ == "__main__":
    main()
