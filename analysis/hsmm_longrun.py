"""
analysis/hsmm_longrun.py

HSMM 레짐을 2004년부터 장기 재현 — "코로나를 진짜 맞춘 건가?"에 답하기 위한 분석.

■ 왜 필요한가
  production(hsmm_final.py)은 판단 시작이 2018-01인데 롤링창이 60개월이라,
  2018~2022 판정은 학습데이터가 창을 못 채운 워밍업 상태에서 나왔다.
  실제로 2018·2019는 24개월 내리 Bear였고 2020-02(코로나 폭락월) P_bear 상승폭은
  +0.042에 불과했다(이미 0.9 포화). → 코로나 방어가 '탐지'인지 '상시 저노출'인지 구분 불가.

  2004년부터 학습하면 2020년 시점엔 창이 꽉 찬 상태 → 코로나에 진짜 반응하는지 깨끗하게 보인다.
  덤으로 2008 금융위기·2011 유럽재정위기가 들어와 Bear 표본이 늘어난다
  (현재 Bear 구간은 2개뿐 → HSMM duration 분포를 표본 2개로 추정 중).

■ 데이터 (DB 적재 없음. CSV + DB 읽기 전용)
  2000-01~2017-01  analysis/kospi_stocks_2000_2016.csv  (yfinance, close만)
  2017-01~         alpha_lab.daily_price                (DB)
  매크로           kospi(1996~) / usd_krw(2003~) / investor_foreign_kospi(2000~)
  → 환율이 2003-12부터라 실질 패널 시작 2004-01.

■ ★ 한계 (결과 해석 시 반드시 감안)
  1) 생존편향: CSV는 "2017년 현재 상장 종목"을 역으로 받은 것 → 과거 상폐종목 누락.
     종목수 693(2000)→2037(2016) 단조증가가 증거. 2000년대 초 누락률 50%+.
     pykrx는 2015-01이 한계라 2000~2014 상폐 시세는 보완 경로가 없다(확인 완료).
     → 2004~2016 구간의 CAGR/MDD는 성과로 인용하지 말 것. 국면·duration 학습용으로만.
  2) PIT 유니버스 부재: fnspace_master가 2017-01부터라 그 이전 소속 마스크가 없다.
     hsmm_final.build_features()를 그대로 쓰면 K가 전부 False가 되어
     breadth는 vld>50 필터에서 탈락→fillna(0), newlow는 0/clip(1)=0 → **두 피처가 조용히 0**이 된다.
     여기서는 '가격 존재 ∩ 끝자리0(보통주) ∩ KOSPI소속' 프록시로 대체한다(아래 KOSPI_SET).
  3) 정의 불일치: 2)의 프록시와 2017년 이후 마스터 기반 정의가 다르다.
     → 경계(2016-12 vs 2017-01)에서 레벨 점프가 생기면 모델이 국면전환으로 오인한다.
     그래서 --diag 를 먼저 돌려 점프 여부를 확인하고, 통과할 때만 본 분석으로 간다.

■ 사용
  .venv/bin/python analysis/hsmm_longrun.py --diag     # 1단계: 경계 진단만 (먼저 이것부터)
  .venv/bin/python analysis/hsmm_longrun.py            # 2단계: 전체 walk-forward + 코로나 리포트
  옵션: --decide-start 2009-01  (판정 시작월, 기본 2009-01 = 창 60개월 충족)
        --refresh                (피처 캐시 재생성)

■ 산출
  analysis/.cache/hsmm_longrun_features.pkl     피처 패널 캐시
  analysis/results/hsmm_longrun_path.csv        월별 P_bear·익스포저
  analysis/results/hsmm_longrun_boundary.png    2017년 경계 진단 차트
"""
import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent.parent
A = Path(__file__).parent
OUT = A / "results"; OUT.mkdir(exist_ok=True)
CACHE = A / ".cache" / "hsmm_longrun_features.pkl"
CSV = A / "kospi_stocks_2000_2016.csv"

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

# 한글 폰트 (OS별 자동 선택)
_HAVE = {f.name for f in font_manager.fontManager.ttflist}
KFONT = next((c for c in ("AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic")
              if c in _HAVE), "DejaVu Sans")
plt.rcParams["font.family"] = KFONT
plt.rcParams["axes.unicode_minus"] = False

# hsmm_final의 모델 코드를 그대로 재사용 (정의 불일치 방지)
import importlib.util
_spec = importlib.util.spec_from_file_location("hsmm_final", A / "hsmm_final.py")
HF = importlib.util.module_from_spec(_spec)
sys.modules["hsmm_final"] = HF
_spec.loader.exec_module(HF)

PANEL_START = "2004-01"        # 환율(usd_krw) 2003-12~ 이라 3M 변화율이 서는 첫 달
BOUNDARY = "2017-01"           # CSV → DB 전환 지점


# ─────────────────────────── 데이터 ───────────────────────────
def load_prices():
    """CSV(2000~2017-01) + DB(2017~)를 이어붙여 일별 종가 wide 테이블을 만든다."""
    print("CSV 로드 (2000~2016, 548만행 ~30초)...", flush=True)
    c = pd.read_csv(CSV, dtype={"stock_code": str})[["dt", "stock_code", "close"]]
    c = c[c.dt < BOUNDARY]                       # 겹치는 2017-01 앞부분은 DB 쪽을 신뢰
    print(f"  CSV {len(c):,}행 {c.stock_code.nunique()}종목  {c.dt.min()} ~ {c.dt.max()}", flush=True)

    print("DB 로드 (2017~, 658만행 ~1.5분)...", flush=True)
    conn = HF._connect()
    d = pd.read_sql("SELECT trade_date dt, stock_code, close::float close FROM alpha_lab.daily_price "
                    "WHERE close IS NOT NULL", conn)
    snap = pd.read_sql("SELECT snapshot_date ym, stock_code FROM alpha_lab.fnspace_master "
                       "WHERE market='KOSPI' AND sec_cd_nm IS NOT NULL", conn)
    conn.close()
    print(f"  DB  {len(d):,}행 {d.stock_code.nunique()}종목", flush=True)

    px = pd.concat([c, d], ignore_index=True)
    px["dt"] = pd.to_datetime(px.dt.str.slice(0, 10))
    wide = px.pivot_table(index="dt", columns="stock_code", values="close").sort_index()

    snap["code"] = snap.stock_code.str[1:]
    snap = snap[snap.code.str.match(r"^\d{5}0$")]
    return wide, snap


def build_mask(wide, snap):
    """소속 마스크 K (일×종목).
       2017-01~ : fnspace_master 월별 PIT 스냅샷 (production과 동일)
       ~2016-12 : 마스터가 없으므로 '가격 존재 ∩ 끝자리0 ∩ KOSPI소속' 프록시.
                  KOSPI소속은 2017년 이후 마스터에 한 번이라도 등장한 종목으로 근사한다.
                  (2017 전에 사라진 종목은 CSV에도 없으므로 어차피 불가 — 생존편향 항목 1)"""
    by_ym = snap.groupby("ym")["code"].apply(set).to_dict()
    yms_snap = sorted(by_ym)
    kospi_ever = set().union(*by_ym.values())          # 2017~ 마스터에 등장한 KOSPI 보통주
    day_ym = wide.index.to_period("M").strftime("%Y-%m")

    K = pd.DataFrame(False, index=wide.index, columns=wide.columns)
    proxy_cols = [c for c in wide.columns if c.endswith("0") and c in kospi_ever]
    for _ym in np.unique(day_ym):
        rows = day_ym == _ym
        avail = [s for s in yms_snap if s <= _ym]
        if avail:                                      # 2017-01 이후: 실제 PIT 스냅샷
            K.loc[rows, wide.columns.isin(by_ym[avail[-1]])] = True
        else:                                          # 2016-12 이전: 프록시
            K.loc[rows, proxy_cols] = True
    K &= wide.notna()                                  # 가격 없는 날은 미상장/거래정지 취급
    return K


def build_features(use_cache=True):
    """월별 피처 패널. hsmm_final.build_features()와 같은 정의를 쓰되 유니버스만 장기용으로 교체."""
    if use_cache and "--refresh" not in sys.argv and CACHE.exists():
        print(f"[cache] 피처 재사용: {CACHE.relative_to(BASE)} (갱신하려면 --refresh)", flush=True)
        return pd.read_pickle(CACHE)

    wide, snap = load_prices()
    K = build_mask(wide, snap)

    print("breadth / newlow 계산...", flush=True)
    ma = wide.rolling(200, min_periods=100).mean()
    vld = wide.notna() & ma.notna() & K
    breadth = ((wide > ma) & vld).sum(axis=1) / vld.sum(axis=1).clip(lower=1)
    breadth = breadth[vld.sum(axis=1) > 50]
    rmn = wide.rolling(252, min_periods=60).min()
    okK = wide.notna() & K
    newlow = ((wide <= rmn) & okK).sum(axis=1) / okK.sum(axis=1).clip(lower=1)
    newlow = newlow[okK.sum(axis=1) > 50]              # ★ production엔 없는 가드.
    #   production은 okK 합이 0이어도 clip(lower=1) 때문에 0/1=0으로 조용히 통과한다.
    #   장기 패널에선 초기 구간이 그 상태가 될 수 있어 명시적으로 잘라낸다.

    print("매크로 로드...", flush=True)
    conn = HF._connect()

    def mac(ind):
        x = pd.read_sql(f"SELECT period p, value::float v FROM alpha_lab.macro_indicators "
                        f"WHERE indicator='{ind}' AND freq='D'", conn)
        x["p"] = pd.to_datetime(x["p"].str.slice(0, 10))
        return x.set_index("p")["v"].sort_index()

    kospi, usdkrw, frn = mac("kospi"), mac("usd_krw"), mac("investor_foreign_kospi")
    conn.close()

    # ↓ 여기부터는 hsmm_final.build_features()와 동일한 정의 (복제 시 값이 갈리지 않도록 그대로 옮김)
    from datetime import timedelta
    lr = np.log(kospi / kospi.shift(1))
    kma = kospi.rolling(200, min_periods=100).mean()
    fx_m1y = usdkrw.rolling(252, min_periods=120).mean()

    ym = pd.PeriodIndex(kospi.index, freq="M"); last = {}
    for dt_, p in zip(kospi.index, ym):
        last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= pd.Timestamp(PANEL_START)]

    def asof(s, e):
        x = s[s.index <= e]
        return x.iloc[-1] if len(x) else np.nan

    def pctc(s, e, dy):
        cur = asof(s, e); prv = s[s.index <= e - timedelta(days=dy)]
        return (cur / prv.iloc[-1] - 1) if len(prv) and prv.iloc[-1] else np.nan

    def rv(e, win):
        r = lr[lr.index <= e].iloc[-win:]
        return r.std() * np.sqrt(252) if len(r) > 3 else np.nan

    def dv(e, win):
        r = lr[lr.index <= e].iloc[-win:]
        return np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2) * 252) if len(r) > 3 else np.nan

    def flow(e, dy):
        x = frn[(frn.index <= e) & (frn.index > e - timedelta(days=dy))]
        return x.sum() if len(x) else np.nan

    base, Px, rvol_l, dvol_l = [], [], [], []
    for e in mends:
        km, kp = asof(kma, e), asof(kospi, e)
        trend = np.log(kp / km) if (km and kp and not np.isnan(km) and km > 0 and kp > 0) else 0.0
        fx_chg = pctc(usdkrw, e, 90); lvl, lvlm = asof(usdkrw, e), asof(fx_m1y, e)
        fx_ctx = (fx_chg * (lvl / lvlm) if (lvlm and not np.isnan(lvlm) and lvlm > 0 and not np.isnan(fx_chg))
                  else (fx_chg if not np.isnan(fx_chg) else 0.0))
        base.append(dict(breadth=asof(breadth, e), newlow=asof(newlow, e), trend=trend,
                         fx3m=fx_ctx, fflow=flow(e, 90)))
        Px.append(kp); rvol_l.append(rv(e, 20)); dvol_l.append(dv(e, 60))

    idx = [pd.Period(e, freq="M").strftime("%Y-%m") for e in mends]
    raw = pd.DataFrame(base, index=idx)                     # fillna 전 — 결측 진단용
    df = raw[HF.EMIS_COLS + HF.TRAN_COLS].fillna(0.0)
    Px = np.array(Px); n = len(df); yms = list(df.index)
    ret = np.array([(Px[i + 1] / Px[i] - 1) if i + 1 < n else np.nan for i in range(n)])
    rvol = np.nan_to_num(np.array(rvol_l), nan=HF.TARGET_VOL)
    dvol = np.nan_to_num(np.array(dvol_l), nan=HF.TARGET_VOL)
    dd6 = np.full(n, np.nan)
    for i in range(n - 1):
        if i + 6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]),
                              kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i + 6])]])
            dd6[i] = (path / path.cummax() - 1).min() * 100

    out = (df, yms, n, ret, rvol, dvol, dd6, raw)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, CACHE)
    print(f"[cache] 피처 저장: {CACHE.relative_to(BASE)}", flush=True)
    return out


# ─────────────────────────── 1단계: 경계 진단 ───────────────────────────
def diagnose(df, yms, raw):
    """2017-01 경계에서 breadth/newlow 레벨이 튀는지. 튀면 본 분석으로 가면 안 된다."""
    print("\n" + "=" * 78)
    print("  경계 진단 — CSV(~2016-12) → DB(2017-01~) 전환 지점")
    print("=" * 78)

    miss = raw[HF.EMIS_COLS].isna()
    if miss.any().any():
        print("\n  [결측] fillna(0) 되기 전 NaN 개월수:")
        for c in HF.EMIS_COLS:
            m = miss[c]
            if m.any():
                print(f"    {c:9s} {m.sum():3d}개월   최초 {list(raw.index[m])[:3]}")
        print("    ※ NaN은 fillna(0.0)으로 0이 된다. 0이 많으면 모델이 그걸 학습한다.")
    else:
        print("\n  [결측] 없음 — 전 구간 breadth/newlow 산출됨")

    i = yms.index(BOUNDARY) if BOUNDARY in yms else None
    if i is None:
        print(f"\n  !! {BOUNDARY}이 패널에 없다"); return False

    # ★ 유니버스에 의존하는 피처만 검사한다.
    #   trend는 macro_indicators의 KOSPI 지수(1996~)로만 계산되어 CSV/DB 경계와 무관하다
    #   (build_features의 trend = log(kospi/kospi_MA200) — wide/K를 쓰지 않음).
    #   실제로 2017년 KOSPI 상승 때문에 trend가 오르는데, 이걸 경계 아티팩트로 잡으면 오탐이다.
    UNIVERSE_COLS = ["breadth", "newlow"]

    pre, post = df.iloc[max(0, i - 12):i], df.iloc[i:i + 12]
    print(f"\n  전후 12개월 평균 (경계 {BOUNDARY}):")
    print(f"    {'피처':10s}{'이전':>10s}{'이후':>10s}{'차이':>10s}{'이전σ':>10s}{'차이/σ':>9s}   판정")
    ok = True
    for c in HF.EMIS_COLS:
        a, b = pre[c].mean(), post[c].mean()
        sd = pre[c].std()
        z = abs(b - a) / sd if sd > 1e-9 else np.nan
        if c not in UNIVERSE_COLS:
            note = "   (유니버스 무관 — 검사 제외)"
        elif np.isnan(z) or z < 1.0:
            note = "   통과"
        else:
            note = "   ← 점프"; ok = False
        print(f"    {c:10s}{a:10.3f}{b:10.3f}{b - a:+10.3f}{sd:10.3f}{z:9.2f}{note}")

    # 계단 판별: 정의 불일치면 '경계 한 달'에 튄다. 여러 달에 걸친 변화는 시장 움직임이다.
    print(f"\n  계단 검사 (경계 직전월 → 경계월, 국소 변동 대비):")
    for c in UNIVERSE_COLS:
        step = df[c].iloc[i] - df[c].iloc[i - 1]
        local = df[c].iloc[max(0, i - 12):i].diff().abs().mean()
        r = abs(step) / local if local > 1e-9 else np.nan
        print(f"    {c:10s} 변화 {step:+.3f}   평소 월변화 {local:.3f}   배율 {r:.2f}"
              f"{'   ← 이상' if (not np.isnan(r) and r > 2.5) else '   정상'}")

    print("\n  월별 상세 (2016-07 ~ 2017-06):")
    for y in [v for v in yms if "2016-07" <= v <= "2017-06"]:
        r = df.loc[y]
        mark = "  ← 경계" if y == BOUNDARY else ""
        print(f"    {y}  breadth {r.breadth:6.3f}   newlow {r.newlow:6.3f}   trend {r.trend:+6.3f}{mark}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    x = pd.PeriodIndex(yms, freq="M").to_timestamp()
    for ax, c, lab in zip(axes, ["breadth", "newlow"], ["breadth (200일선 위 비율)", "newlow (52주 신저가 비율)"]):
        ax.plot(x, df[c].values, lw=1.3, color="#2a78d6")
        ax.axvline(pd.Timestamp("2017-01-01"), color="#eb6834", ls="--", lw=1.6)
        ax.text(pd.Timestamp("2017-02-01"), ax.get_ylim()[1] * 0.92, " CSV→DB 경계",
                color="#eb6834", fontsize=9)
        ax.set_ylabel(lab); ax.grid(alpha=.3)
    axes[0].set_title("경계 진단 — 2017-01에서 레벨 점프가 있으면 정의 불일치", fontsize=12)
    fig.tight_layout()
    p = OUT / "hsmm_longrun_boundary.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\n  차트 → {p}")

    print("\n  " + ("판정: 점프 없음 — 본 분석 진행 가능" if ok else
                    "판정: ★ 레벨 점프 감지 — 정의 불일치. 본 분석 결과를 신뢰하지 말 것"))
    return ok


# ─────────────────────────── 2단계: 코로나 리포트 ───────────────────────────
def covid_report(yms, pbear, ret, start):
    """이 스크립트의 존재 이유. 코로나에 P_bear이 '움직였는지'를 본다.
       production(2018 시작)에서는 2020-02 Δpbear이 +0.042뿐이었다(이미 0.9 포화)."""
    print("\n" + "=" * 78)
    print("  ★ 코로나 구간 (2019-06 ~ 2020-10) — 신호가 실제로 움직였는가")
    print("=" * 78)
    s = pd.Series(pbear, index=yms)
    print(f"    {'월':9s}{'KOSPI':>9s}{'P_bear':>9s}{'Δ':>8s}   판정")
    prev = np.nan
    for i, y in enumerate(yms):
        if not ("2019-06" <= y <= "2020-10"):
            continue
        pb, r = pbear[i], ret[i]
        d = pb - prev if not np.isnan(prev) else np.nan
        mark = "  ← 코로나 폭락" if y == "2020-02" else ""
        rs = f"{r * 100:+8.1f}%" if not np.isnan(r) else "       -"
        ds = f"{d:+8.3f}" if not np.isnan(d) else "       -"
        print(f"    {y:9s}{rs}{pb:9.3f}{ds}{mark}")
        prev = pb

    pre = s.loc["2019-08":"2020-01"].mean()
    jump = s.get("2020-02", np.nan) - s.get("2020-01", np.nan)
    print(f"\n    폭락 직전 6개월 평균 P_bear : {pre:.3f}")
    print(f"    2020-01 → 2020-02 상승폭     : {jump:+.3f}")
    if pre > 0.80:
        print("    → 이미 포화(>0.80). 코로나 이전부터 Bear였으므로 '탐지'로 볼 수 없다.")
    elif jump > 0.30:
        print("    → ★ 낮은 수준에서 급등. 코로나에 실제로 반응했다 = 탐지 성과.")
    else:
        print("    → 반응이 미미하다. 탐지로 보기 어렵다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true", help="경계 진단만 하고 종료")
    ap.add_argument("--decide-start", default="2009-01", help="판정 시작월(기본 2009-01: 창 60개월 충족)")
    ap.add_argument("--refresh", action="store_true", help="피처 캐시 재생성")
    args = ap.parse_args()

    df, yms, n, ret, rvol, dvol, dd6, raw = build_features()
    print(f"\n패널: {yms[0]} ~ {yms[-1]}  ({n}개월)")

    ok = diagnose(df, yms, raw)
    if args.diag:
        print("\n--diag 모드 종료. 점프가 없으면 옵션 없이 다시 실행하세요.")
        return
    if not ok:
        print("\n★ 경계 점프가 감지되어 중단합니다. 정의를 맞춘 뒤 진행하세요.")
        print("   (그래도 강행하려면 이 체크를 주석 처리하되, 결과를 성과로 인용하지 마세요.)")
        return

    HF.DECIDE_START = args.decide_start                  # walk_forward가 참조하는 전역
    pbear_raw, start = HF.walk_forward(df, yms, n)
    pbear = pbear_raw.copy()                             # EMA 스무딩 — hsmm_final.main()과 동일 구현
    for t in range(start + 1, n):
        pbear[t] = HF.PBEAR_EMA * pbear_raw[t] + (1 - HF.PBEAR_EMA) * pbear[t - 1]

    covid_report(yms, pbear, ret, start)

    # 이산 레짐(히스테리시스) — 위기 구간 요약용
    reg = ["Bull"] * n; p = "Bull"
    for t in range(start, n):
        p = ("Bear" if pbear[t] >= HF.T_OUT else "Bull") if p == "Bear" else ("Bear" if pbear[t] >= HF.T_IN else "Bull")
        reg[t] = p

    print("\n" + "=" * 78)
    print(f"  Bear 구간 (판정 {yms[start]}~{yms[-1]})")
    print("=" * 78)
    segs, s0 = [], None
    for t in range(start, n):
        if reg[t] == "Bear" and s0 is None:
            s0 = t
        if (reg[t] != "Bear" or t == n - 1) and s0 is not None:
            e = t if reg[t] != "Bear" else t + 1
            segs.append((yms[s0], yms[e - 1], e - s0)); s0 = None
    for a, z, ln in segs:
        m = np.nanmean(ret[yms.index(a):yms.index(z) + 1]) * 100
        print(f"    {a} ~ {z}   {ln:3d}개월   KOSPI 평균 {m:+.2f}%/월")
    print(f"\n    Bear 구간 {len(segs)}개 / Bear 개월 {sum(s[2] for s in segs)}개월 / 전체 {n - start}개월")
    print("    ※ production(2018~)은 Bear 구간이 2개뿐이었다. duration 추정 표본 확보가 이 분석의 목적.")

    pd.DataFrame({"ym": yms, "ret": ret, "pbear": pbear, "regime": reg,
                  "rvol": rvol, "dvol": dvol, "dd6": dd6}).to_csv(
        OUT / "hsmm_longrun_path.csv", index=False, encoding="utf-8-sig")
    print(f"\n  경로 → {OUT / 'hsmm_longrun_path.csv'}")
    print("\n★ 주의: 2004~2016은 생존편향(상폐종목 누락)이 커서 CAGR/MDD를 성과로 인용하지 말 것.")
    print("        국면 판정·duration 학습·코로나 반응 확인 용도로만 사용.")


if __name__ == "__main__":
    main()
