"""
analysis/hsmm_evernewlow.py

newlow → ever_newlow_20 교체 A/B 실험 (production 기간 그대로: 패널 2017-01~, 판정 2018-01~)

■ 무엇을 고치는가 (측정 버그)
  현재 production의 newlow는 '월말 그날 하루' 신저가 종목 비율이다.

      rmn    = wide.rolling(252).min()
      newlow = ((wide <= rmn) & okK).sum(axis=1) / okK.sum(axis=1)
      ...  asof(newlow, e)          ← 월말 '하루'만 샘플링

  신저가는 특정 하루에만 성립하는 점(point) 이벤트다. 그런데 월말 하루를 뽑아 쓰므로
  그 달 내내 신저가가 쏟아졌어도 월말이 반등일이면 값이 0에 가깝게 나온다.
  급락장(하루에 몰림)에서는 우연히 잡히지만, **슬로우 베어처럼 신저가가 여러 날에
  흩어져 발생하면 대부분 놓친다** — 2021~23 미탐지의 후보 원인.

  ever_newlow_20 = '최근 20거래일 중 한 번이라도' 신저가를 찍은 종목 비율.
  월말 하루가 아니라 그 달 전체를 본다. 정의만 바꾸는 것이고 새 데이터는 쓰지 않는다.

■ 실험 설계
  - EMIS_COLS의 newlow를 ever_newlow_20으로 **교체**(추가 아님) → 피처 개수 동일(3개)
  - 나머지(breadth/trend/fx3m/fflow, 창 60월, HSMM, 시드)는 전부 동일
  - 같은 실행에서 A(기존)·B(교체)를 모두 돌려 조건을 맞춘다
  - production 파일은 수정하지 않는다(모델 코드는 hsmm_final에서 import해 재사용)

■ 평가 = 위험국면 탐지기
  1개월 방향 적중률이 아니라 Recall / Lead / False alarm / Precision / F1 / confusion으로 본다.
  이벤트 정의는 production과 동일: dd6(향후 6개월 최대낙폭) <= -15%.

■ 사용
  .venv/bin/python analysis/hsmm_evernewlow.py
  옵션: --refresh   피처 캐시 재생성 (DB 재조회 ~1.5분)
        --thr -15   이벤트 임계 낙폭(%)

■ 산출
  analysis/.cache/hsmm_evernewlow_panel.pkl
  analysis/results/hsmm_evernewlow_compare.csv
"""
import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import timedelta

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
CACHE = A_DIR / ".cache" / "hsmm_evernewlow_panel.pkl"

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

EVER_WIN = 20            # 거래일 (약 1개월)


# ─────────────────────────── 패널 ───────────────────────────
def build_panel(use_cache=True):
    """production과 동일한 정의로 패널을 만들되 newlow / ever_newlow_20 두 컬럼을 모두 산출."""
    if use_cache and "--refresh" not in sys.argv and CACHE.exists():
        print(f"[cache] 패널 재사용: {CACHE.relative_to(BASE)} (갱신 --refresh)", flush=True)
        return pd.read_pickle(CACHE)

    print("[db] 일별 주가 로드 (658만행, ~1.5분)...", flush=True)
    conn = HF._connect()

    def mac(ind):
        x = pd.read_sql(f"SELECT period p, value::float v FROM alpha_lab.macro_indicators "
                        f"WHERE indicator='{ind}' AND freq='D'", conn)
        x["p"] = pd.to_datetime(x["p"].str.slice(0, 10))
        return x.set_index("p")["v"].sort_index()

    kospi, usdkrw, frn = mac("kospi"), mac("usd_krw"), mac("investor_foreign_kospi")
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close "
                    "FROM alpha_lab.daily_price WHERE close IS NOT NULL AND trade_date>='2017-01-01'", conn)
    snap = pd.read_sql("SELECT snapshot_date ym, stock_code FROM alpha_lab.fnspace_master "
                       "WHERE market='KOSPI' AND sec_cd_nm IS NOT NULL", conn)
    conn.close()

    d["dt"] = pd.to_datetime(d["dt"])
    wide = d.pivot_table(index="dt", columns="stock_code", values="close").sort_index()
    snap["code"] = snap.stock_code.str[1:]
    snap = snap[snap.code.str.match(r"^\d{5}0$")]
    by_ym = snap.groupby("ym")["code"].apply(set).to_dict(); yms_snap = sorted(by_ym)
    K = pd.DataFrame(False, index=wide.index, columns=wide.columns)
    day_ym = wide.index.to_period("M").strftime("%Y-%m")
    for _ym in np.unique(day_ym):
        avail = [s for s in yms_snap if s <= _ym]
        if avail:
            K.loc[day_ym == _ym, wide.columns.isin(by_ym[avail[-1]])] = True

    print("피처 계산 (breadth / newlow / ever_newlow_20)...", flush=True)
    ma = wide.rolling(200, min_periods=100).mean()
    vld = wide.notna() & ma.notna() & K
    breadth = ((wide > ma) & vld).sum(axis=1) / vld.sum(axis=1).clip(lower=1)
    breadth = breadth[vld.sum(axis=1) > 50]

    rmn = wide.rolling(252, min_periods=60).min()
    okK = wide.notna() & K
    hit = ((wide <= rmn) & okK)                       # 그날 신저가를 찍은 종목
    den = okK.sum(axis=1).clip(lower=1)

    newlow = hit.sum(axis=1) / den                                        # A: 기존(그날 하루)
    ever = hit.astype("int8").rolling(EVER_WIN, min_periods=1).max()      # 최근 20일 중 1회라도
    ever_newlow_20 = (ever.astype(bool) & okK).sum(axis=1) / den          # B: 교체안
    newlow = newlow[okK.sum(axis=1) > 50]
    ever_newlow_20 = ever_newlow_20[okK.sum(axis=1) > 50]

    # ── 이하 production(hsmm_final.build_features)과 동일 정의 ──
    lr = np.log(kospi / kospi.shift(1))
    kma = kospi.rolling(200, min_periods=100).mean()
    fx_m1y = usdkrw.rolling(252, min_periods=120).mean()
    ym = pd.PeriodIndex(kospi.index, freq="M"); last = {}
    for dt_, p in zip(kospi.index, ym):
        last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= pd.Timestamp(HF.PANEL_START)]

    def asof(s, e):
        x = s[s.index <= e]
        return x.iloc[-1] if len(x) else np.nan

    def pctc(s, e, dy):
        c0 = asof(s, e); p0 = s[s.index <= e - timedelta(days=dy)]
        return (c0 / p0.iloc[-1] - 1) if len(p0) and p0.iloc[-1] else np.nan

    def rv(e, win):
        r = lr[lr.index <= e].iloc[-win:]
        return r.std() * np.sqrt(252) if len(r) > 3 else np.nan

    def dv(e, win):
        r = lr[lr.index <= e].iloc[-win:]
        return np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2) * 252) if len(r) > 3 else np.nan

    def flow(e, dy):
        x = frn[(frn.index <= e) & (frn.index > e - timedelta(days=dy))]
        return x.sum() if len(x) else np.nan

    rows, Px, rvol_l, dvol_l = [], [], [], []
    for e in mends:
        km, kp = asof(kma, e), asof(kospi, e)
        trend = np.log(kp / km) if (km and kp and not np.isnan(km) and km > 0 and kp > 0) else 0.0
        fx_chg = pctc(usdkrw, e, 90); lvl, lvlm = asof(usdkrw, e), asof(fx_m1y, e)
        fx_ctx = (fx_chg * (lvl / lvlm) if (lvlm and not np.isnan(lvlm) and lvlm > 0 and not np.isnan(fx_chg))
                  else (fx_chg if not np.isnan(fx_chg) else 0.0))
        rows.append(dict(breadth=asof(breadth, e), newlow=asof(newlow, e),
                         ever_newlow_20=asof(ever_newlow_20, e), trend=trend,
                         fx3m=fx_ctx, fflow=flow(e, 90)))
        Px.append(kp); rvol_l.append(rv(e, 20)); dvol_l.append(dv(e, 60))

    idx = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    df = pd.DataFrame(rows, index=idx).fillna(0.0)
    Px = np.array(Px); n = len(df); yms = list(df.index)
    ret = np.array([(Px[i + 1] / Px[i] - 1) if i + 1 < n else np.nan for i in range(n)])
    dd6 = np.full(n, np.nan)
    for i in range(n - 1):
        if i + 6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]),
                              kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i + 6])]])
            dd6[i] = (path / path.cummax() - 1).min() * 100

    out = (df, yms, n, ret, dd6)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, CACHE)
    print(f"[cache] 패널 저장: {CACHE.relative_to(BASE)}", flush=True)
    return out


# ─────────────────────────── 탐지기 평가 ───────────────────────────
def regimes(pbear, n, start):
    """production과 동일한 히스테리시스(0.6/0.4)로 이산 레짐."""
    reg = ["Bull"] * n; p = "Bull"
    for t in range(start, n):
        p = ("Bear" if pbear[t] >= HF.T_OUT else "Bull") if p == "Bear" else ("Bear" if pbear[t] >= HF.T_IN else "Bull")
        reg[t] = p
    return reg


def evaluate(name, pbear, reg, yms, n, start, dd6, thr):
    """위험국면 탐지기로 평가 (1개월 방향 적중률이 아님)."""
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
    fa = [s for s in onset if s not in matched]
    prec = len(matched) / len(onset) if onset else np.nan
    f1 = 2 * rec * prec / (rec + prec) if (rec and prec and rec + prec > 0) else np.nan

    # 월단위 confusion: 위험월(향후6M 낙폭<=thr) 대비 Bear 판정
    ok = [t for t in idx if not np.isnan(dd6[t])]
    tp = sum(1 for t in ok if dd6[t] <= thr and reg[t] == "Bear")
    fn = sum(1 for t in ok if dd6[t] <= thr and reg[t] != "Bear")
    fp = sum(1 for t in ok if dd6[t] > thr and reg[t] == "Bear")
    tn = sum(1 for t in ok if dd6[t] > thr and reg[t] != "Bear")
    mrec = tp / (tp + fn) if tp + fn else np.nan
    mprec = tp / (tp + fp) if tp + fp else np.nan

    # ★ 착시 방지 지표 — 이게 없으면 '항상 Bear' 모델이 F1 최고점을 받는다.
    #   리프트 = Precision / 위험월 기저율. 1.0이면 무작위와 동일.
    #   분별력 = Bull월 평균 향후낙폭 - Bear월 평균 향후낙폭 (클수록 두 판정이 실제로 다름).
    base = (tp + fn) / len(ok) if ok else np.nan
    bear_ratio = (tp + fp) / len(ok) if ok else np.nan
    lift = mprec / base if (base and not np.isnan(mprec) and base > 0) else np.nan
    bear_dd = np.mean([dd6[t] for t in ok if reg[t] == "Bear"]) if (tp + fp) else np.nan
    bull_dd = np.mean([dd6[t] for t in ok if reg[t] != "Bear"]) if (tn + fn) else np.nan
    disc = bull_dd - bear_dd if not (np.isnan(bear_dd) or np.isnan(bull_dd)) else np.nan

    def pct(v):
        return "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0%}"

    nhit = sum(h is not None for h in hits)
    lead_s = f"{np.mean(leads):+.1f}개월" if leads else "-"
    f1_s = "  -  " if np.isnan(f1) else f"{f1:.2f}"
    print(f"\n  [{name}]")
    print(f"    이벤트 {len(evs)}개 중 탐지 {nhit}개   Recall {pct(rec)}   평균 선행 {lead_s}")
    print(f"    Bear 진입 {len(onset)}회 중 오경보 {len(fa)}회   Precision {pct(prec)}   F1 {f1_s}")
    print(f"    월단위  TP {tp}  FN {fn}  FP {fp}  TN {tn}   Recall {pct(mrec)}  Precision {pct(mprec)}")
    print(f"    Bear 개월 {sum(1 for t in idx if reg[t]=='Bear')}/{len(idx)}")
    lift_s = "  -  " if np.isnan(lift) else f"{lift:.2f}배"
    disc_s = "  -  " if np.isnan(disc) else f"{disc:+.1f}%p"
    print(f"    ★ Bear비율 {pct(bear_ratio)} (기저율 {pct(base)})   리프트 {lift_s}   분별력 {disc_s}")
    if not np.isnan(lift) and lift < 1.10:
        print(f"       → 리프트 {lift:.2f}배: 사실상 '항상 Bear'. Recall/F1이 높아도 신호 없음.")

    NAMES = {"2018-10": "2018 하락", "2020-02": "2020 코로나", "2021-11": "2021~22 하락",
             "2022-01": "2022 하락", "2024-08": "2024 급락", "2025-09": "2025 하락"}
    print(f"    {'이벤트':16}{'낙폭':>7}   결과")
    for ev, h in zip(evs, hits):
        if h is None:
            tag = "X(놓침)"
        else:
            cand = [s for s in onset if abs(s - ev) <= HF.WIN]
            tag = f"{ev - min(cand, key=lambda s: abs(s-ev)):+d}개월"
        print(f"    {NAMES.get(yms[ev], yms[ev]):16}{dd6[ev]:>6.0f}%   {tag}")
    return dict(recall=rec, prec=prec, f1=f1, lead=np.mean(leads) if leads else np.nan,
                fa=len(fa), bear=sum(1 for t in idx if reg[t] == "Bear"),
                tp=tp, fn=fn, fp=fp, tn=tn,
                bear_ratio=bear_ratio, lift=lift, disc=disc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--thr", type=float, default=-15.0, help="이벤트 임계 낙폭(%%)")
    args = ap.parse_args()
    thr = -abs(args.thr)

    df, yms, n, ret, dd6 = build_panel()
    print(f"\n패널 {yms[0]} ~ {yms[-1]} ({n}개월)   판정 시작 {HF.DECIDE_START}")

    print("\n피처 분포 비교 (교체 대상):")
    for c in ["newlow", "ever_newlow_20"]:
        s = df[c]
        print(f"  {c:16s} 평균 {s.mean():.3f}  σ {s.std():.3f}  최대 {s.max():.3f}  "
              f"0에 가까운 달(<0.01) {int((s < 0.01).sum())}/{n}")
    print(f"  상관계수 {df.newlow.corr(df.ever_newlow_20):.3f}")

    # 변형 정의. 전부 '교체'이며 emission 피처는 항상 3개로 고정한다.
    #   bear_score = -breadth + [1] - trend  → 인덱스 1은 '높을수록 약세'여야 한다.
    #   ever_newlow_20 / z / Δ 모두 값이 클수록 신저가가 늘어난 것이므로 방향이 성립한다.
    df = df.copy()
    df["ever_z36"] = HF.roll_z(df[["ever_newlow_20"]], ["ever_newlow_20"])["ever_newlow_20"]
    df["ever_d1"] = df["ever_newlow_20"].diff().fillna(0.0)

    VARIANTS = [
        ("A. 기존 newlow", "newlow"),            # 현행 production
        ("B. ever20 (원시)", "ever_newlow_20"),   # 측정창만 확대 — 포화로 기각됨
        ("C. ever20 Z36", "ever_z36"),           # 36개월 롤링 Z — 레벨 드리프트 제거
        # ("D. ever20 Δ", "ever_d1"),            # 전월대비 변화(PPT 성과모델 방식) — 보류
    ]

    res = {}
    for name, col in VARIANTS:
        HF.EMIS_COLS = ["breadth", col, "trend"]
        sub = df[["breadth", col, "trend"] + HF.TRAN_COLS].copy()
        print(f"\n{'='*74}\n  {name}   EMIS={HF.EMIS_COLS}\n{'='*74}")
        pbear_raw, start = HF.walk_forward(sub, yms, n)
        pbear = pbear_raw.copy()
        for t in range(start + 1, n):
            pbear[t] = HF.PBEAR_EMA * pbear_raw[t] + (1 - HF.PBEAR_EMA) * pbear[t - 1]
        reg = regimes(pbear, n, start)
        res[name] = (pbear, reg, start)

    print(f"\n{'='*74}\n  탐지기 평가 (이벤트 = 향후6M 낙폭 <= {thr:.0f}%)\n{'='*74}")
    summ = {}
    for name, (pbear, reg, start) in res.items():
        summ[name] = evaluate(name, pbear, reg, yms, n, start, dd6, thr)

    print(f"\n{'='*74}\n  요약\n{'='*74}")
    names = [v[0] for v in VARIANTS]
    hdr = "".join(f"{_nm.split('.')[0] + '.' + _nm.split('.')[1][:9]:>15}" for _nm in names)
    print(f"  {'지표':16}{hdr}")
    ROWS = [("recall", "이벤트Recall", "{:.0%}"), ("lead", "평균선행(월)", "{:+.1f}"),
            ("f1", "F1", "{:.2f}"), ("bear_ratio", "Bear비율", "{:.0%}"),
            ("lift", "★리프트", "{:.2f}배"), ("disc", "★분별력", "{:+.1f}%p")]
    for k, lab, fmt in ROWS:
        cells = ""
        for _nm in names:
            v = summ[_nm][k]
            cells += f"{'  -  ':>15}" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{fmt.format(v):>15}"
        print(f"  {lab:16}{cells}")

    print("\n  판정 기준:")
    print("    · 리프트 ≈ 1.0  → '항상 Bear'. Recall/F1이 높아도 신호 없음 (B가 이 함정에 빠졌다)")
    print("    · 분별력       → Bear/Bull 판정이 실제로 다른 미래를 가리키는가. 클수록 좋음")
    print("    · 평균선행 음수 → 낙폭이 시작된 뒤 탐지. 방어에 쓸 수 없음")
    best = max(names, key=lambda _x: (summ[_x]["lift"] if not np.isnan(summ[_x]["lift"]) else 0)
               * (summ[_x]["disc"] if not np.isnan(summ[_x]["disc"]) else 0))
    print(f"\n  리프트×분별력 최고: {best}")

    rows = []
    for name, (pbear, reg, start) in res.items():
        for t in range(n):
            rows.append(dict(variant=name, ym=yms[t], pbear=pbear[t], regime=reg[t],
                             ret=ret[t], dd6=dd6[t]))
    pd.DataFrame(rows).to_csv(OUT / "hsmm_evernewlow_compare.csv", index=False, encoding="utf-8-sig")
    print(f"\n  경로 → {OUT / 'hsmm_evernewlow_compare.csv'}")
    print("\n※ 판정 기준: Recall·Lead가 오르고 오경보가 크게 늘지 않으면 채택 검토.")
    print("  특히 '2021~22 하락'이 X(놓침)에서 탐지로 바뀌는지가 이 실험의 핵심.")


if __name__ == "__main__":
    main()
