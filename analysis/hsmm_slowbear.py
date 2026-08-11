"""
analysis/hsmm_slowbear.py

slow bear(2021~23) 탐지 개선 — emission에 ma200_slope_60 **1개만 추가**.

■ 문제 (진단 완료)
  2021-11~2022-12 내내 breadth는 하위 10분위(0.10~0.30), trend는 최저치 부근(-0.19)이었는데
  pbear는 newlow가 튄 달에만 0.50으로 올랐다 곧 0.01로 꺼졌다.
  원인은 스케일: newlow는 첨도 18.7의 스파이크형이라 z가 +3.3까지 가는데,
  breadth는 분포가 얌전해 바닥이어도 -1.3σ가 한계다.
  가우시안 emission은 z²에 비례하므로 newlow 스파이크 한 방이 breadth의 6배 발언권을 갖는다.
  → 패닉 없는 slow bear는 breadth·trend가 눌려 있어도 Bear로 안 넘어간다.

■ 처방 (이번 실험)
  newlow를 건드리지 않고(교체 실험은 이미 기각 — [[project_ever_newlow_rejected]])
  **지속성 있는 피처 1개만 추가**한다.

      ma200_slope_60 = KOSPI 200일선의 60거래일 변화율 = MA200[t]/MA200[t-60] - 1

  trend(=log(price/MA200), '가격이 선 위냐 아래냐')와 달리 slope는 '선 자체가 어디로 가느냐'다.
  slow bear에서 trend는 가격이 200일선을 들락거려 0 근처로 되돌아오지만
  (2022-11 -0.011, 2023-01 -0.001, 2023-02 +0.003),
  slope는 2021-12~2023-03 **16개월 연속 음수**를 유지한다. HSMM의 Bear 지속에 필요한 성질.

■ 위험이 낮은 이유
  - KOSPI 지수(macro_indicators, 1996~)만으로 계산 → 종목 유니버스·생존편향과 무관
  - 새 데이터 수집 없음, DB 주가 재조회 없음(캐시 패널 재사용)
  - production 파일 미수정

■ 유의점
  - trend와 상관 0.83으로 높다(중복 위험). 단독 변별력도 0.11σ로 약하다.
    기대는 '변별력'이 아니라 '지속성'에 있다.
  - emission이 3→4개가 되므로 bear_score를 일반화해야 한다(원본은 3개 하드코딩).
      원본:  -means[:,0] + means[:,1] - means[:,2]      = means @ [-1, +1, -1]
      확장:  means @ [-1, +1, -1, -1]   (slope는 낮을수록 약세 → 음수 부호)

■ 사용
  .venv/bin/python analysis/hsmm_slowbear.py
  옵션: --thr -15
■ 산출
  analysis/results/hsmm_slowbear_compare.csv
"""
import os
import sys
import argparse
import warnings
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
PANEL = A_DIR / ".cache" / "hsmm_evernewlow_panel.pkl"     # 기존 실험 캐시 재사용
SLOPE_CACHE = A_DIR / ".cache" / "ma200_slope_60.pkl"
LONG_PANEL = A_DIR / ".cache" / "hsmm_longrun_features.pkl"

sys.path.insert(0, str(BASE))
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ModuleNotFoundError:
    _env = BASE / ".env"
    if _env.exists():
        for _ln in _env.read_text(encoding="utf-8").splitlines():
            _ln = _ln.strip()
            if _ln and not _ln.startswith("#") and "=" in _ln:
                _k, _v = _ln.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import importlib.util
_spec = importlib.util.spec_from_file_location("hsmm_final", A_DIR / "hsmm_final.py")
HF = importlib.util.module_from_spec(_spec)
sys.modules["hsmm_final"] = HF
_spec.loader.exec_module(HF)

SLOPE_WIN = 60          # 거래일


def load_slope(yms):
    """KOSPI 200일선의 60거래일 변화율을 월말 기준으로."""
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
    slope = ma / ma.shift(SLOPE_WIN) - 1
    mend = {}
    for d, pp in zip(ks.index, pd.PeriodIndex(ks.index, freq="M")):
        mend[pp.strftime("%Y-%m")] = d
    s = pd.Series({y: (slope[slope.index <= mend[y]].iloc[-1] if y in mend else np.nan) for y in yms})
    s = s.fillna(0.0)
    SLOPE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(s, SLOPE_CACHE)
    return s


def cold_emission_fixed(X):
    """★ production 버그 수정판.

    원본(hsmm_final.py:230):
        covs = np.diag(np.asarray(hm.covars_[k]).reshape(-1)[:d])
    hmlearn의 covars_[k]는 이미 (d,d) 행렬이라 reshape(-1)[:d]는 '첫 번째 행'을 집는다.
    대각행렬의 첫 행 = [var0, 0, 0, ...] → 초기 공분산이 diag(var0, 0, 0)이 되어
    breadth 외 피처의 분산이 0이 된다(1e-6 ridge로 연산만 통과). d=4면 아예 터진다.
    올바른 대각 추출은 np.diag(np.diag(M)).
    """
    from hmmlearn.hmm import GaussianHMM
    d = X.shape[1]
    hm = GaussianHMM(2, "diag", n_iter=50, random_state=HF.SEED)
    hm.fit(X)
    covs = np.array([np.diag(np.diag(np.asarray(hm.covars_[k]))) + 1e-6 * np.eye(d) for k in range(2)])
    return dict(means=hm.means_.copy(), covs=covs,
                Amat=np.array([[0.9, 0.1], [0.2, 0.8]]), pi=np.array([0.85, 0.15]))



def cold_emission_anchor(X):
    """★ production의 앵커-바닥 초기화를 임의의 d로 일반화한 것.

    production(hsmm_final.cold_emission)은 앵커 피처(EMIS_COLS[0]=breadth)에만 적합 분산을
    주고 나머지는 바닥값(1e-6)으로 고정해 초기 상태 분리를 강제한다(13개월 창 전용 정규화).
    d=3에서는 production과 **완전히 동일**하고, d=4에서도 같은 정책을 그대로 적용한다.
    → '정상 covariance'로 바꾸면 EM이 붕괴하지만(2026-08-10 검증), 이 정책은 d를 늘려도 안전하다.
    """
    from hmmlearn.hmm import GaussianHMM
    d = X.shape[1]
    hm = GaussianHMM(2, "diag", n_iter=50, random_state=HF.SEED)
    hm.fit(X)
    covs = np.zeros((2, d, d))
    for k in range(2):
        var = np.zeros(d)
        var[0] = float(np.asarray(hm.covars_[k])[0, 0])
        covs[k] = np.diag(var) + 1e-6 * np.eye(d)
    return dict(means=hm.means_.copy(), covs=covs,
                Amat=np.array([[0.9, 0.1], [0.2, 0.8]]), pi=np.array([0.85, 0.15]))


_GUARD_HITS = [0]


def make_safe_m_step(orig):
    """m_step_emis의 수치 가드.

    첫 창이 13개월뿐이라(판정 2018-01, 패널 2017-01 시작) 상태 하나가 붕괴하면
    R≈EPS로 나눠져 공분산에 NaN/inf가 섞이고 cholesky가 터진다.
    모델을 바꾸는 게 아니라 NaN 제거 + 최소 고윳값 바닥만 보장한다.
    정상 케이스에선 아무것도 하지 않는다(no-op) → 기존 결과와 비교 가능.
    """
    def _f(X, gamma, w):
        means, covs = orig(X, gamma, w)
        means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        d = covs.shape[-1]
        for k in range(covs.shape[0]):
            C = np.nan_to_num(covs[k], nan=0.0, posinf=0.0, neginf=0.0)
            C = (C + C.T) / 2.0
            ev = np.linalg.eigvalsh(C)
            if not np.all(np.isfinite(ev)) or ev.min() < 1e-6:
                _GUARD_HITS[0] += 1
                bump = max(1e-6 - min(ev.min(), 0.0), 1e-6)
                C = C + bump * np.eye(d)
            covs[k] = C
        return means, covs
    return _f


def make_bear_score(signs):
    """emission 개수에 무관한 bear_score. 원본(3개)과 수식이 동일하도록 부호를 준다."""
    sg = np.asarray(signs, dtype=float)

    def _f(means):
        return means @ sg
    return _f


def regimes(pbear, n, start):
    reg = ["Bull"] * n; p = "Bull"
    for t in range(start, n):
        p = ("Bear" if pbear[t] >= HF.T_OUT else "Bull") if p == "Bear" else ("Bear" if pbear[t] >= HF.T_IN else "Bull")
        reg[t] = p
    return reg


def evaluate(name, pbear, reg, yms, n, start, dd6, thr):
    idx = list(range(start, n))
    evs = [i for i in idx if dd6[i] <= thr and (i == start or not (dd6[i - 1] <= thr))]
    onset = [t for t in idx if reg[t] == "Bear" and reg[t - 1] != "Bear"]
    hits, leads, matched = [], [], set()
    for ev in evs:
        cand = [s for s in onset if abs(s - ev) <= HF.WIN]
        if cand:
            best = min(cand, key=lambda s: abs(s - ev))
            hits.append(ev); leads.append(ev - best); matched.add(best)
        else:
            hits.append(None)
    rec = sum(h is not None for h in hits) / len(evs) if evs else np.nan
    ok = [t for t in idx if not np.isnan(dd6[t])]
    tp = sum(1 for t in ok if dd6[t] <= thr and reg[t] == "Bear")
    fn = sum(1 for t in ok if dd6[t] <= thr and reg[t] != "Bear")
    fp = sum(1 for t in ok if dd6[t] > thr and reg[t] == "Bear")
    tn = sum(1 for t in ok if dd6[t] > thr and reg[t] != "Bear")
    mprec = tp / (tp + fp) if tp + fp else np.nan
    base = (tp + fn) / len(ok) if ok else np.nan
    bear_ratio = (tp + fp) / len(ok) if ok else np.nan
    lift = mprec / base if (base and not np.isnan(mprec) and base > 0) else np.nan
    bdd = np.mean([dd6[t] for t in ok if reg[t] == "Bear"]) if (tp + fp) else np.nan
    udd = np.mean([dd6[t] for t in ok if reg[t] != "Bear"]) if (tn + fn) else np.nan
    disc = udd - bdd if not (np.isnan(bdd) or np.isnan(udd)) else np.nan

    # slow bear 전용: 2021-11 ~ 2022-12 중 Bear로 판정한 개월수
    sb = [t for t in idx if "2021-11" <= yms[t] <= "2022-12"]
    sb_bear = sum(1 for t in sb if reg[t] == "Bear")

    print(f"\n  [{name}]")
    print(f"    이벤트 {len(evs)}개 중 탐지 {sum(h is not None for h in hits)}개  Recall {rec:.0%}"
          f"   평균선행 {(f'{np.mean(leads):+.1f}개월' if leads else '-')}")
    print(f"    Bear비율 {bear_ratio:.0%} (기저 {base:.0%})   리프트 {lift:.2f}배   분별력 {disc:+.1f}%p")
    print(f"    ★ slow bear(2021-11~2022-12) {len(sb)}개월 중 Bear 판정 {sb_bear}개월")
    NAMES = {"2018-10": "2018 하락", "2020-02": "2020 코로나", "2021-11": "2021~22 하락",
             "2022-01": "2022 하락", "2024-08": "2024 급락", "2025-09": "2025 하락"}
    for ev, h in zip(evs, hits):
        if h is None:
            tag = "X(놓침)"
        else:
            cand = [s for s in onset if abs(s - ev) <= HF.WIN]
            tag = f"{ev - min(cand, key=lambda s: abs(s-ev)):+d}개월"
        print(f"      {NAMES.get(yms[ev], yms[ev]):16}{dd6[ev]:>6.0f}%   {tag}")
    return dict(recall=rec, lead=np.mean(leads) if leads else np.nan, lift=lift,
                disc=disc, bear_ratio=bear_ratio, sb=sb_bear, sb_tot=len(sb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=-15.0)
    ap.add_argument("--long", action="store_true",
                    help="장기 패널(2004~) 사용. 짧은 패널은 첫 창 13개월이라 4피처 추정 불가")
    ap.add_argument("--decide-start", default=None, help="판정 시작월(장기 기본 2009-01)")
    args = ap.parse_args()
    thr = -abs(args.thr)

    if args.long:
        if not LONG_PANEL.exists():
            print(f"!! 장기 패널 캐시가 없습니다: {LONG_PANEL}")
            print("   먼저 .venv/bin/python analysis/hsmm_longrun.py 를 실행하세요.")
            return
        df, yms, n, ret, _rv, _dv, dd6, _raw = pd.read_pickle(LONG_PANEL)
        HF.DECIDE_START = args.decide_start or "2009-01"
        df = df.copy()
        # 장기 패널엔 newlow가 이미 emission에 있다(컬럼명 동일). 그대로 쓴다.
    else:
        if not PANEL.exists():
            print(f"!! 패널 캐시가 없습니다: {PANEL}")
            print("   먼저 .venv/bin/python analysis/hsmm_evernewlow.py 를 한 번 실행하세요.")
            return
        df, yms, n, ret, dd6 = pd.read_pickle(PANEL)
        if args.decide_start:
            HF.DECIDE_START = args.decide_start
        df = df.copy()
    df["ma200_slope_60"] = load_slope(yms).values

    print(f"패널 {yms[0]} ~ {yms[-1]} ({n}개월)   판정 시작 {HF.DECIDE_START}")
    S = df.loc[HF.DECIDE_START:]
    print(f"\nma200_slope_60  평균 {S.ma200_slope_60.mean():+.4f}  σ {S.ma200_slope_60.std():.4f}"
          f"  왜도 {S.ma200_slope_60.skew():.2f}  첨도 {S.ma200_slope_60.kurt():.2f}")
    print(f"  trend와 상관 {S.trend.corr(S.ma200_slope_60):.2f}  (높으면 중복 위험)")
    neg = (S.ma200_slope_60 < 0)
    print(f"  음수 개월 {neg.sum()}/{len(S)}   최장 연속 음수 "
          f"{max((sum(1 for _ in g) for k_, g in __import__('itertools').groupby(neg) if k_), default=0)}개월")

    # 3원 비교: 버그 수정 효과와 새 피처 효과를 분리한다.
    VARIANTS = [
        ("A. production", ["breadth", "newlow", "trend"], [-1, +1, -1], "anchor"),
        ("H. 앵커+slope", ["breadth", "newlow", "trend", "ma200_slope_60"], [-1, +1, -1, -1], "anchor"),
        ("A'. 정상cov", ["breadth", "newlow", "trend"], [-1, +1, -1], "normal"),
        ("H'. 정상cov+slope", ["breadth", "newlow", "trend", "ma200_slope_60"], [-1, +1, -1, -1], "normal"),
    ]
    orig_bear_score, orig_cold = HF.bear_score, HF.cold_emission
    orig_m_step = HF.m_step_emis
    HF.m_step_emis = make_safe_m_step(orig_m_step)   # 전 변형 공통(정상시 no-op)
    res = {}
    try:
        for name, cols, signs, fix in VARIANTS:
            HF.EMIS_COLS = cols
            HF.bear_score = make_bear_score(signs)
            HF.cold_emission = {"anchor": cold_emission_anchor,
                                "normal": cold_emission_fixed}[fix]
            _GUARD_HITS[0] = 0
            sub = df[cols + HF.TRAN_COLS].copy()
            print(f"\n{'='*72}\n  {name}   EMIS={cols}   cold={fix}\n{'='*72}")
            pbear_raw, start = HF.walk_forward(sub, yms, n)
            pbear = pbear_raw.copy()
            for t in range(start + 1, n):
                pbear[t] = HF.PBEAR_EMA * pbear_raw[t] + (1 - HF.PBEAR_EMA) * pbear[t - 1]
            if _GUARD_HITS[0]:
                print(f"  (수치 가드 발동 {_GUARD_HITS[0]}회 — 공분산 NaN/비양정 보정)")
            res[name] = (pbear, regimes(pbear, n, start), start)
    finally:
        HF.bear_score, HF.cold_emission = orig_bear_score, orig_cold
        HF.m_step_emis = orig_m_step

    print(f"\n{'='*72}\n  탐지기 평가 (이벤트 = 향후6M 낙폭 <= {thr:.0f}%)\n{'='*72}")
    summ = {nm: evaluate(nm, p, r, yms, n, st, dd6, thr) for nm, (p, r, st) in res.items()}

    print(f"\n{'='*72}\n  요약\n{'='*72}")
    names = [v[0] for v in VARIANTS]
    print(f"  {'지표':22}" + "".join(f"{_x:>18}" for _x in names))
    for k, lab, fmt in [("sb", "★slow bear Bear월", "{:d}"), ("lift", "★리프트", "{:.2f}배"),
                        ("disc", "★분별력", "{:+.1f}%p"), ("recall", "이벤트Recall", "{:.0%}"),
                        ("lead", "평균선행(월)", "{:+.1f}"), ("bear_ratio", "Bear비율", "{:.0%}")]:
        cells = ""
        for _x in names:
            v = summ[_x][k]
            cells += f"{'  -  ':>18}" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{fmt.format(v):>18}"
        print(f"  {lab:22}{cells}")

    a, h = summ[names[0]], summ[names[-1]]
    af = summ[names[1]]
    print(f"\n  판정:")
    print(f"    slow bear 판정  현행 {a['sb']} → 버그수정 {af['sb']} → +slope {h['sb']}  (전체 {a['sb_tot']}개월)")
    print(f"    버그수정 단독 효과: 리프트 {a['lift']:.2f} → {af['lift']:.2f},  분별력 {a['disc']:+.1f} → {af['disc']:+.1f}%p")
    if h["sb"] > a["sb"] and h["lift"] >= a["lift"] * 0.95 and h["bear_ratio"] < 0.55:
        print("    → ★ 채택 검토: slow bear를 더 잡으면서 리프트·Bear비율이 무너지지 않음")
    elif h["bear_ratio"] >= 0.55:
        print("    → 기각: Bear비율 과다(포화). '항상 Bear'로 얻은 개선은 무의미")
    elif h["sb"] <= a["sb"]:
        print("    → 기각: slow bear 판정이 늘지 않음. 이 피처로는 안 됨")
    else:
        print("    → 보류: slow bear는 늘었으나 리프트가 하락. 득실 판단 필요")

    rows = []
    for nm, (p, r, st) in res.items():
        for t in range(n):
            rows.append(dict(variant=nm, ym=yms[t], pbear=p[t], regime=r[t], ret=ret[t], dd6=dd6[t],
                             ma200_slope_60=df["ma200_slope_60"].iloc[t]))
    pd.DataFrame(rows).to_csv(OUT / "hsmm_slowbear_compare.csv", index=False, encoding="utf-8-sig")
    print(f"\n  경로 → {OUT / 'hsmm_slowbear_compare.csv'}")


if __name__ == "__main__":
    main()
