"""
analysis/diag_transition_signal.py

진단 전용 — "2022 slow bear 정보는 이미 transition 피처에 있었는데,
fflow 편향과 반복 표준화 때문에 약화된 것인가?"

파라미터 튜닝 없음(λ·KAPPA 불변). production 미수정. 원인 분해만 한다.

■ 배경
  stress = roll_z(fx3m) - roll_z(fflow) 는 2022에 분명히 반응했다
  (구간 평균 +2.14 vs 전체 +0.93, 2022-09 +4.12로 코로나 +3.04 상회).
  그런데 위험월 변별력은 0.09σ에 불과하고 z_ff는 73%의 달에서 음수다.

■ 신호 경로 3단계
  L0  원 피처            fx3m, fflow
  L1  roll_z(36M)      z_fx, z_ff  → stress_L1 = z_fx - z_ff
  L2  walk_forward      학습창(60M) 내 재표준화 → 모델이 실제로 보는 값

■ 진단 1: z_ff 음수 73% 원인 분해 (아직 처방 선택 안 함)
  H1 장기 추세      : fflow 자체가 우하향 → 이동평균 대비 지속 미달
  H2 분포 비대칭    : 우측 꼬리(간헐적 대규모 순매수)로 평균>중앙값
                     → 평균 기준 z는 절반 넘는 달에서 음수 (추세 없어도 발생)
  H3 창 정의        : 36M 창/min_periods 때문
  H4 구조 변화      : 특정 시점 전후 레짐 이동
  H5 스케일 드리프트 : 변동성 자체가 커져 z가 압축

■ 진단 2: 재표준화가 지속형 stress를 스스로 지우는가
  같은 raw stress가 2020-02 vs 2022-01에서 왜 다른 L2가 되는지
  '창 평균 이동'과 '창 표준편차 변화'로 분해하고,
  stress 고조 지속개월(run-length)과 감쇠율의 관계를 측정한다.

■ 사용 / 산출
  .venv/bin/python analysis/diag_transition_signal.py
  analysis/results/diag_transition.csv
"""
import sys
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent.parent
A_DIR = Path(__file__).parent
OUT = A_DIR / "results"; OUT.mkdir(exist_ok=True)
PANEL = A_DIR / ".cache" / "hsmm_features.pkl"
sys.path.insert(0, str(BASE))

_sp = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_sp); sys.modules["hsmm_final"] = HF; _sp.loader.exec_module(HF)

DS = HF.DECIDE_START          # 2018-01
WINDOW_M = HF.WINDOW_M        # 60
SLOW_A, SLOW_B = "2021-11", "2022-12"


def hr(t):
    print(f"\n{'='*84}\n  {t}\n{'='*84}")


def main():
    if not PANEL.exists():
        print("!! production 패널 없음 — analysis/hsmm_final.py 먼저 실행"); return
    df, yms, n, ret, rvol, dvol, dd6 = pd.read_pickle(PANEL)
    idx = pd.Index(yms)
    fx, ff = df["fx3m"].copy(), df["fflow"].copy()
    fx.index = ff.index = idx
    TRz = HF.roll_z(df, HF.TRAN_COLS); TRz.index = idx
    zfx, zff = TRz["fx3m"], TRz["fflow"]
    sL1 = zfx - zff
    D = pd.Series(dd6, index=idx)

    print(f"패널 {yms[0]}~{yms[-1]} ({n}개월)   판정 {DS}   fflow 단위: 백만원(90일 누적 순매수)")

    # ══════════════════ 진단 1 : z_ff 음수 73% 원인 ══════════════════
    hr("진단 1  z_ff가 73% 음수인 이유 — 가설별 분해")
    S = ff.loc[DS:]
    print(f"  z_ff 음수 비율 {(zff.loc[DS:] < 0).mean():.0%}   평균 {zff.loc[DS:].mean():+.2f}")

    print("\n  [H2 분포 비대칭] 평균 기준 z는 우측꼬리 분포에서 구조적으로 음수가 많다")
    print(f"    fflow 왜도 {S.skew():+.2f}   첨도 {S.kurt():+.2f}")
    print(f"    평균 {S.mean()/1e6:+.3f}백만   중앙값 {S.median()/1e6:+.3f}백만"
          f"   (평균>중앙값이면 우측꼬리)")
    # 이동'중앙값' 기준으로 바꿔보면 음수 비율이 50%에 수렴하는가?
    med = ff.rolling(36, min_periods=12).median()
    below_mean = (ff < ff.rolling(36, min_periods=12).mean()).loc[DS:].mean()
    below_med = (ff < med).loc[DS:].mean()
    print(f"    이동평균 미달 비율 {below_mean:.0%}   이동중앙값 미달 비율 {below_med:.0%}")
    print(f"    → 중앙값 기준이 50%에 가까우면 '비대칭'이 주원인, 둘 다 높으면 '추세'가 주원인")

    print("\n  [H1 장기 추세] 시간에 대한 회귀 + 연도별 평균")
    x = np.arange(len(S)); y = S.values
    b1, b0 = np.polyfit(x, y, 1)
    resid = y - (b0 + b1 * x)
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
    print(f"    기울기 {b1/1e6:+.4f}백만/월   t={b1/se:+.2f}   "
          f"{'유의한 추세' if abs(b1/se) > 2 else '유의한 추세 아님'}")
    yr = S.groupby(S.index.str[:4]).agg(["mean", "median", "std", "count"])
    print(f"    {'연도':6}{'평균(백만)':>10}{'중앙(백만)':>10}{'σ(백만)':>9}{'개월':>5}")
    for y_, r in yr.iterrows():
        print(f"    {y_:6}{r['mean']/1e6:>10.3f}{r['median']/1e6:>10.3f}{r['std']/1e6:>9.3f}{int(r['count']):>5}")

    print("\n  [H3 창 정의] 창 길이를 바꾸면 음수 비율이 달라지는가")
    print(f"    {'창':>6}{'음수비율':>10}{'평균z':>9}")
    for wlen in (24, 36, 60, 120):
        m = ff.rolling(wlen, min_periods=12).mean()
        sd = ff.rolling(wlen, min_periods=12).std().replace(0, np.nan)
        z = ((ff - m) / sd).fillna(0.0).clip(-4, 4).loc[DS:]
        print(f"    {wlen:>6}{(z<0).mean():>9.0%}{z.mean():>9.2f}")

    print("\n  [H5 스케일 드리프트] 변동성이 커지면 z가 압축된다")
    print(f"    {'연도':6}{'|fflow| 평균(백만)':>16}{'롤링σ(백만)':>12}")
    rs = ff.rolling(36, min_periods=12).std()
    for y_ in sorted(set(S.index.str[:4])):
        m = S.index.str[:4] == y_
        print(f"    {y_:6}{np.abs(S[m]).mean()/1e6:>16.3f}{rs.loc[DS:][m].mean()/1e6:>12.3f}")

    print("\n  [H4 구조 변화] 전후 비교 (2018~2021 vs 2022~)")
    a, b = ff.loc["2018-01":"2021-12"], ff.loc["2022-01":]
    print(f"    2018~2021  평균 {a.mean()/1e6:+.3f}백만  중앙 {a.median()/1e6:+.3f}백만  σ {a.std()/1e6:.3f}")
    print(f"    2022~      평균 {b.mean()/1e6:+.3f}백만  중앙 {b.median()/1e6:+.3f}백만  σ {b.std()/1e6:.3f}")

    # ══════════════════ 진단 2 : 재표준화 ══════════════════
    hr("진단 2  walk-forward 내 재표준화가 지속형 stress를 약화시키는가")

    L2, wmean, wstd = {}, {}, {}
    start = yms.index(DS)
    for t in range(start, n):
        lo = max(0, t + 1 - WINDOW_M)
        w = sL1.values[lo:t + 1]
        wmean[yms[t]] = w.mean(); wstd[yms[t]] = w.std()
        L2[yms[t]] = (w[-1] - w.mean()) / (w.std() + HF.EPS)
    L2 = pd.Series(L2); wmean = pd.Series(wmean); wstd = pd.Series(wstd)

    print("\n  같은 raw가 왜 다른 z가 되는가 — 창 평균/표준편차로 분해")
    print(f"    {'시점':10}{'stress_L1':>11}{'창평균':>9}{'창σ':>8}{'stress_L2':>11}   해석")
    for y_ in ["2020-02", "2020-03", "2022-01", "2022-08", "2022-09"]:
        if y_ not in L2.index:
            continue
        print(f"    {y_:10}{sL1[y_]:>11.2f}{wmean[y_]:>9.2f}{wstd[y_]:>8.2f}{L2[y_]:>11.2f}")
    if "2020-02" in L2.index and "2022-01" in L2.index:
        d_num = sL1["2022-01"] - sL1["2020-02"]
        d_mean = wmean["2022-01"] - wmean["2020-02"]
        d_std = wstd["2022-01"] / wstd["2020-02"]
        print(f"\n    2020-02 → 2022-01:  L1 차이 {d_num:+.2f}   창평균 {d_mean:+.2f} 상승"
              f"   창σ {d_std:.2f}배")
        print(f"      L2가 {L2['2020-02']:.2f} → {L2['2022-01']:.2f}로 축소된 주원인은 "
              f"{'창평균 상승' if abs(d_mean) > abs(wstd['2022-01']*(d_std-1)) else '창σ 확대'}")

    print("\n  ★ 지속 자기소거 검정 — stress 고조가 이어질수록 L2가 깎이는가")
    hi = sL1 > sL1.loc[DS:].median()
    run = np.zeros(len(sL1))
    for i in range(len(sL1)):
        run[i] = (run[i - 1] + 1) if (i > 0 and hi.iloc[i]) else (1.0 if hi.iloc[i] else 0.0)
    R = pd.Series(run, index=idx)
    tab = pd.DataFrame({"L1": sL1, "L2": L2, "run": R}).dropna()
    tab["atten"] = tab.L2 / tab.L1.replace(0, np.nan)
    print(f"    {'고조 지속개월':>12}{'개월수':>7}{'L1평균':>9}{'L2평균':>9}{'L2/L1':>9}")
    for lo_, hi_, lab in [(1, 1, "1"), (2, 3, "2~3"), (4, 6, "4~6"), (7, 99, "7+")]:
        m = (tab.run >= lo_) & (tab.run <= hi_) & (tab.L1 > 0)
        if m.sum() == 0:
            continue
        print(f"    {lab:>12}{int(m.sum()):>7}{tab.L1[m].mean():>9.2f}{tab.L2[m].mean():>9.2f}"
              f"{tab.atten[m].mean():>9.2f}")
    print("    → 지속개월이 길수록 L2/L1이 낮아지면 '지속형 신호 자기소거' 확인")

    print(f"\n  2021-01 ~ 2023-03 월별 3단계 경로")
    print(f"    {'ym':9}{'fx3m':>8}{'fflow(백만)':>11}{'z_fx':>7}{'z_ff':>7}"
          f"{'L1':>7}{'창평균':>8}{'창σ':>7}{'L2':>7}  slow")
    rows = []
    for y_ in [v for v in yms if "2021-01" <= v <= "2023-03"]:
        mark = "★" if SLOW_A <= y_ <= SLOW_B else ""
        print(f"    {y_:9}{fx[y_]:>8.3f}{ff[y_]/1e6:>11.2f}{zfx[y_]:>7.2f}{zff[y_]:>7.2f}"
              f"{sL1[y_]:>7.2f}{wmean.get(y_, np.nan):>8.2f}{wstd.get(y_, np.nan):>7.2f}"
              f"{L2.get(y_, np.nan):>7.2f}  {mark}")
        rows.append(dict(ym=y_, fx3m=fx[y_], fflow=ff[y_], z_fx=zfx[y_], z_ff=zff[y_],
                         stress_L1=sL1[y_], win_mean=wmean.get(y_, np.nan),
                         win_std=wstd.get(y_, np.nan), stress_L2=L2.get(y_, np.nan)))

    # ══════════════════ 종합 ══════════════════
    hr("종합 — 2022 정보는 있었는데 약화된 것인가")
    risk = D.loc[DS:] <= -15
    for lab, s in [("stress_L1", sL1.loc[DS:]), ("stress_L2", L2)]:
        r_, p_ = s[risk.reindex(s.index).fillna(False)], s[~risk.reindex(s.index).fillna(False)]
        print(f"  {lab}  위험월 {r_.mean():+.2f}  평시 {p_.mean():+.2f}"
              f"  격차 {abs(r_.mean()-p_.mean())/s.std():.2f}σ")
    sb = [y_ for y_ in yms if SLOW_A <= y_ <= SLOW_B]
    print(f"\n  slow bear 구간   L1 평균 {sL1[sb].mean():+.2f}   L2 평균 {L2.reindex(sb).mean():+.2f}"
          f"   축소율 {L2.reindex(sb).mean()/sL1[sb].mean():.2f}배")
    cov = [y_ for y_ in yms if "2020-01" <= y_ <= "2020-04"]
    print(f"  코로나 구간      L1 평균 {sL1[cov].mean():+.2f}   L2 평균 {L2.reindex(cov).mean():+.2f}"
          f"   축소율 {L2.reindex(cov).mean()/sL1[cov].mean():.2f}배")

    pd.DataFrame(rows).to_csv(OUT / "diag_transition.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT/'diag_transition.csv'}")
    print("\n※ 이 스크립트는 진단 전용이다. λ·KAPPA 불변, production 미수정, 처방 미채택.")


if __name__ == "__main__":
    main()
