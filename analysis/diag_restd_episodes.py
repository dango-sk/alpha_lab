"""
analysis/diag_restd_episodes.py

진단 전용 — walk-forward 재표준화의 '지속형 stress 자기소거'가
2022만의 현상인지, 여러 위기에서 반복되는 구조적 문제인지 장기 패널(2004~)에서 검증.

■ 검증 가설
  stress가 높은 상태가 오래 지속될수록 학습창 평균이 그것을 따라 올라가면서
  L2/L1(재표준화 후/전) 축소율이 체계적으로 낮아지는가?

  production 패널(2018~)에서 이미 관측된 값:
    2022 slow bear  L1 +2.14 → L2 +0.56   축소율 0.26
    2020 코로나      L1 +3.04 → L2 +2.26   축소율 0.74

■ 신호 경로 (production과 동일, 변경 없음)
  L1 = roll_z(fx3m, 36M) - roll_z(fflow, 36M)
  L2 = (L1[t] - mean(L1[t-59..t])) / std(L1[t-59..t])      ← walk_forward 내부와 동일

■ 이번 단계에서 하지 않는 것
  고정 스케일 / expanding window / 창 길이 튜닝 / fflow detrending / λ 재튜닝 /
  새 transition 피처 / production 변경 — 전부 없음. 진단만.

■ 결론 형식
  ① 구조적 자기소거 확인 / ② 2022 특이현상 / ③ 판단 불충분

■ 사용 / 산출
  .venv/bin/python analysis/diag_restd_episodes.py
  analysis/results/diag_restd_episodes.csv
"""
import sys
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent.parent
A_DIR = Path(__file__).parent
OUT = A_DIR / "results"; OUT.mkdir(exist_ok=True)
LONG_PANEL = A_DIR / ".cache" / "hsmm_longrun_features.pkl"
sys.path.insert(0, str(BASE))

_sp = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_sp); sys.modules["hsmm_final"] = HF; _sp.loader.exec_module(HF)

WINDOW_M = HF.WINDOW_M              # 60 — production과 동일
DIAG_START = "2006-01"              # 창 24개월 이상 확보되는 시점부터 진단

# 위기 구간 (사후 성과 최적화용이 아니라 서술용 라벨. 경계는 통상적 정의)
CRISES = [
    ("2008 금융위기", "2008-06", "2009-02"),
    ("2011 유럽",     "2011-08", "2011-12"),
    ("2015 차이나",   "2015-06", "2015-09"),
    ("2018 하락",     "2018-06", "2019-01"),
    ("2020 코로나",   "2020-01", "2020-04"),
    ("2022 slow bear", "2021-11", "2022-12"),
]


def hr(t):
    print(f"\n{'='*92}\n  {t}\n{'='*92}")


def main():
    if not LONG_PANEL.exists():
        print("!! 장기 패널 없음 — analysis/hsmm_longrun.py 먼저 실행"); return
    df, yms, n, ret, _rv, _dv, dd6, _raw = pd.read_pickle(LONG_PANEL)
    idx = pd.Index(yms)
    TRz = HF.roll_z(df, HF.TRAN_COLS); TRz.index = idx
    sL1 = TRz["fx3m"] - TRz["fflow"]

    # L2 = walk_forward 내부 재표준화와 동일 계산
    L2, wmean, wstd, wlen = {}, {}, {}, {}
    s0 = yms.index(DIAG_START)
    for t in range(s0, n):
        lo = max(0, t + 1 - WINDOW_M)
        w = sL1.values[lo:t + 1]
        wmean[yms[t]] = w.mean(); wstd[yms[t]] = w.std(); wlen[yms[t]] = len(w)
        L2[yms[t]] = (w[-1] - w.mean()) / (w.std() + HF.EPS)
    L2 = pd.Series(L2); wmean = pd.Series(wmean); wstd = pd.Series(wstd); wlen = pd.Series(wlen)

    print(f"장기 패널 {yms[0]}~{yms[-1]} ({n}개월)   진단 구간 {DIAG_START}~{yms[-1]} ({len(L2)}개월)")
    print(f"창 {WINDOW_M}개월 (production 동일).  L1 = z(fx3m)-z(fflow),  L2 = 창내 재표준화")
    print(f"L1 전체 평균 {sL1.loc[DIAG_START:].mean():+.2f}  σ {sL1.loc[DIAG_START:].std():.2f}")

    # ══════════════ 1. 위기별 비교 ══════════════
    hr("1. 위기별  stress_L1 → stress_L2  축소율")
    med = sL1.loc[DIAG_START:].median()
    hi = sL1 > med
    run = np.zeros(len(sL1))
    for i in range(len(sL1)):
        run[i] = (run[i - 1] + 1) if (i > 0 and hi.iloc[i]) else (1.0 if hi.iloc[i] else 0.0)
    R = pd.Series(run, index=idx)

    print(f"  {'위기':16}{'개월':>5}{'L1평균':>9}{'L2평균':>9}{'L2/L1':>8}"
          f"{'고조지속':>9}{'창평균':>9}{'창σ':>8}   성격")
    crows = []
    for name, a, b in CRISES:
        sel = [y for y in yms if a <= y <= b and y in L2.index]
        if not sel:
            print(f"  {name:16}   — 진단 구간 밖")
            continue
        l1 = sL1[sel].mean(); l2 = L2[sel].mean()
        ratio = l2 / l1 if abs(l1) > 1e-9 else np.nan
        runmax = R[sel].max()
        kind = "짧은 crash" if runmax <= 3 else ("중간" if runmax <= 6 else "지속형 하락")
        print(f"  {name:16}{len(sel):>5}{l1:>9.2f}{l2:>9.2f}{ratio:>8.2f}"
              f"{int(runmax):>9}{wmean[sel].mean():>9.2f}{wstd[sel].mean():>8.2f}   {kind}")
        crows.append(dict(kind="crisis", name=name, months=len(sel), L1=l1, L2=l2,
                          ratio=ratio, run_max=int(runmax),
                          win_mean=wmean[sel].mean(), win_std=wstd[sel].mean(), label=kind))

    # ══════════════ 2. 전체 episode ══════════════
    hr("2. 전체 stress 고조 episode (L1 > 전체 중앙값, 연속 구간)")
    eps, cur = [], []
    for y in L2.index:
        if hi[y]:
            cur.append(y)
        elif cur:
            eps.append(cur); cur = []
    if cur:
        eps.append(cur)
    erows = []
    print(f"  {'시작':9}{'종료':9}{'개월':>5}{'L1평균':>9}{'L2평균':>9}{'L2/L1':>8}{'창평균':>9}{'창σ':>8}")
    for e in eps:
        l1 = sL1[e].mean(); l2 = L2[e].mean()
        ratio = l2 / l1 if abs(l1) > 1e-9 else np.nan
        print(f"  {e[0]:9}{e[-1]:9}{len(e):>5}{l1:>9.2f}{l2:>9.2f}{ratio:>8.2f}"
              f"{wmean[e].mean():>9.2f}{wstd[e].mean():>8.2f}")
        erows.append(dict(kind="episode", name=f"{e[0]}~{e[-1]}", months=len(e), L1=l1, L2=l2,
                          ratio=ratio, run_max=len(e),
                          win_mean=wmean[e].mean(), win_std=wstd[e].mean(), label=""))
    E = pd.DataFrame(erows)
    print(f"\n  episode 총 {len(E)}개   평균 지속 {E.months.mean():.1f}개월   최장 {E.months.max()}개월")

    # ══════════════ 3. 지속기간 버킷 ══════════════
    hr("3. 지속기간별 축소율 (진단용 구분 — 성과 최적화에 사용 금지)")
    print(f"  {'지속':>8}{'episode수':>10}{'L1평균':>9}{'L2평균':>9}{'L2/L1 평균':>12}{'L2/L1 중앙':>12}")
    buckets = [(1, 1, "1개월"), (2, 3, "2~3"), (4, 6, "4~6"), (7, 999, "7+")]
    brows = []
    for lo_, hi_, lab in buckets:
        m = (E.months >= lo_) & (E.months <= hi_)
        if m.sum() == 0:
            print(f"  {lab:>8}{0:>10}"); continue
        print(f"  {lab:>8}{int(m.sum()):>10}{E.L1[m].mean():>9.2f}{E.L2[m].mean():>9.2f}"
              f"{E.ratio[m].mean():>12.2f}{E.ratio[m].median():>12.2f}")
        brows.append(dict(bucket=lab, n=int(m.sum()), L1=E.L1[m].mean(), L2=E.L2[m].mean(),
                          ratio_mean=E.ratio[m].mean(), ratio_med=E.ratio[m].median()))

    # ══════════════ 4. 관계 검정 ══════════════
    hr("4. 지속기간 ↔ 축소율 관계")
    # ★ L1이 0 근처면 L2/L1이 폭발한다(19.16, -33.53, 29.21 등). 비율이 정의되는
    #   '실제로 stress가 높았던' episode로 제한해야 관계를 볼 수 있다. 사후선택이 아니라
    #   지표 정의상의 요건이므로 임계값(L1>=1.0 = 약 0.5σ)을 고정해 둔다.
    L1_MIN = 1.0
    ok = E.dropna(subset=["ratio"])
    ok = ok[ok.L1 >= L1_MIN]
    print(f"  비율 유효 episode (L1 >= {L1_MIN}): {len(ok)}개 / 전체 {len(E)}개")
    print(f"  {'구간':22}{'개월':>5}{'L1':>7}{'L2':>7}{'L2/L1':>8}")
    for _, r in ok.sort_values('months').iterrows():
        print(f"  {r['name']:22}{int(r['months']):>5}{r['L1']:>7.2f}{r['L2']:>7.2f}{r['ratio']:>8.2f}")
    if len(ok) >= 4:
        rho, p = spearmanr(ok.months, ok.ratio)
        print(f"\n  Spearman(지속개월, L2/L1)  ρ = {rho:+.3f}   p = {p:.4f}   n = {len(ok)}")
        rho2, p2 = spearmanr(ok.months, ok.win_mean)
        print(f"  Spearman(지속개월, 창평균)  ρ = {rho2:+.3f}   p = {p2:.4f}   ← 창평균 추종 여부")
    else:
        rho, p = np.nan, np.nan
        print("  episode 수 부족")

    # 월 단위 보조 검정 — episode 표본이 적으므로 월 단위로도 확인
    print(f"\n  [월 단위] L1 >= {L1_MIN}인 달을 고조 지속개월로 묶어 비교")
    M = pd.DataFrame({"L1": sL1.reindex(L2.index), "L2": L2, "run": R.reindex(L2.index),
                      "wmean": wmean}).dropna()
    M = M[M.L1 >= L1_MIN]
    print(f"    {'지속':>8}{'개월수':>7}{'L1평균':>9}{'L2평균':>9}{'L2-L1':>9}{'창평균':>9}")
    for lo_, hi_, lab in buckets:
        m = (M.run >= lo_) & (M.run <= hi_)
        if m.sum() == 0:
            continue
        print(f"    {lab:>8}{int(m.sum()):>7}{M.L1[m].mean():>9.2f}{M.L2[m].mean():>9.2f}"
              f"{(M.L2[m]-M.L1[m]).mean():>9.2f}{M.wmean[m].mean():>9.2f}")
    if len(M) >= 10:
        rho3, p3 = spearmanr(M.run, M.L2 - M.L1)
        print(f"    Spearman(지속개월, L2-L1)  ρ = {rho3:+.3f}  p = {p3:.4f}  n = {len(M)}")
    else:
        rho3, p3 = np.nan, np.nan

    # ══════════════ 5. 결론 ══════════════
    hr("5. 결론")
    C = pd.DataFrame(crows)
    short = C[C.run_max <= 3]["ratio"]
    long_ = C[C.run_max >= 7]["ratio"]
    print(f"  짧은 crash 축소율 평균 {short.mean():.2f} (n={len(short)})"
          f"   /   지속형 축소율 평균 {long_.mean():.2f} (n={len(long_)})")
    c2022 = C[C.name.str.contains("2022")]["ratio"]
    others_long = long_[C[C.run_max >= 7].name.str.contains("2022") == False] if len(long_) else long_
    print(f"  2022 축소율 {float(c2022.iloc[0]):.2f}"
          f"   /   2022 제외 지속형 평균 "
          f"{(others_long.mean() if len(others_long) else float('nan')):.2f}")

    structural = (len(long_) >= 2 and long_.mean() < 0.7
                  and ((not np.isnan(rho) and rho < -0.3)
                       or (not np.isnan(rho3) and rho3 < -0.3 and p3 < 0.05)))
    only2022 = (len(long_) >= 2 and not structural
                and float(c2022.iloc[0]) < (others_long.mean() - 0.15 if len(others_long) else 99))
    print()
    if structural:
        print("  ▶ ① 구조적 자기소거 확인")
        print("     지속기간이 길수록 L2/L1이 체계적으로 낮아지고, 창평균이 stress를 따라 오른다.")
        print("     2022만의 현상이 아니라 재표준화 구조의 일반적 문제로 판단.")
    elif only2022:
        print("  ▶ ② 2022 특이현상")
        print("     다른 지속형 위기에서는 같은 정도의 소거가 나타나지 않는다.")
        print("     재표준화를 바로 수정하기보다 stress = z(fx3m)-z(fflow) 설계 자체를 재검토할 것.")
    else:
        print("  ▶ ③ 판단 불충분")
        print("     episode 수 또는 지속형 위기 표본이 부족하거나 관계가 일관되지 않는다.")

    pd.concat([C, E], ignore_index=True).to_csv(OUT / "diag_restd_episodes.csv",
                                                index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT/'diag_restd_episodes.csv'}")
    print("\n※ 진단 전용. 처방(고정 스케일/expanding/창 튜닝/detrending/λ) 일절 미적용, production 미수정.")
    print("※ 2004~2016 구간은 생존편향이 있으나 stress는 매크로(환율·외국인)만 쓰므로 이 진단에는 무관.")


if __name__ == "__main__":
    main()
