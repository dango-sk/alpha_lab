"""
analysis/hsmm_exposure_volvariants.py

HSMM P_bear / 익스포저 / BM(KODEX 200) 차트.

핵심: **P_bear과 BM은 고정**이고, 익스포저만 vol-타겟 분모로 무엇을 쓰느냐에 따라 갈린다.
  분모 4종 = 20일 실현 / 60일 실현 / 20일 하방 / 60일 하방
  ※ 하방 = **semi-deviation**: 양(+)의 수익률 날을 0으로 바꿔 계산에 포함 sqrt(mean(min(r,0)^2)*252).
     (hsmm_final.py의 dv()는 음수일만 골라 std를 내는 다른 정의 → 여기와 값이 다름)

익스포저 식(hsmm_final.py §5와 동일, 래칫 없음):
  tgt   = 해당 측정치의 '판단시작~당월' 확장평균      (룩어헤드 없음)
  cut   = 1 - min(1, tgt / max(measure, VOL_FLOOR))
  raw   = clip( (1 - P_bear) x (1 - P_bear x cut), EXP_FLOOR, 1 )
  최종  = 리밸밴드 0.15 넘을 때만 0.05 스텝으로 갱신
※ 격주 하방 래칫은 적용하지 않음(요청).

P_bear은 analysis/hsmm_final_path.csv(월말 확정, EMA 스무딩 후)를 그대로 사용 → 4종 모두 동일.

사용: python analysis/hsmm_exposure_volvariants.py
산출: analysis/results/hsmm_exposure_volvariants.png / .csv
      analysis/results/macro_rate_fx_bm.png / .csv        (금리·환율·BM 한 그래프, FnSpace)
      analysis/results/hsmm_emission_features.png / .csv  (emission 피처 월별 표)
"""
import os, sys, json, warnings
import numpy as np, pandas as pd, psycopg2, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

warnings.filterwarnings("ignore")

# 한글 폰트: OS별로 있는 것을 고른다 (mac=AppleGothic, Windows=Malgun Gothic).
# 고정하면 없는 OS에서 findfont 경고 수천 줄 + 한글이 □로 깨진다.
_HAVE = {f.name for f in font_manager.fontManager.ttflist}
KFONT = next((c for c in ("AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic")
              if c in _HAVE), "DejaVu Sans")
plt.rcParams["font.family"] = KFONT
plt.rcParams["axes.unicode_minus"] = False       # 마이너스 기호가 □로 깨지는 것 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")     # cp949 콘솔에서 —, ※ 등 깨짐 방지
except Exception:
    pass
BASE = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ModuleNotFoundError:                      # dotenv 미설치 파이썬에서도 동작(.env 직접 파싱)
    _env = BASE / ".env"
    if _env.exists():
        for _ln in _env.read_text(encoding="utf-8").splitlines():
            _ln = _ln.strip()
            if not _ln or _ln.startswith("#") or "=" not in _ln:
                continue
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
A = Path(__file__).parent
OUT = A / "results"; OUT.mkdir(exist_ok=True)

STRATEGY = "FCF_YIELD추가전략"                       # 오버레이를 씌울 대상 전략(backtest_cache)
BM_CODE = "069500"                                  # KODEX 200
EXP_FLOOR, VOL_FLOOR, REBAL_BAND, STEP = 0.20, 0.08, 0.15, 0.05
PERIOD_START, PERIOD_END = "2018-04-01", "2026-07-23"   # 백테스트 기간(고정). END가 월중이면 마지막 달은 부분월 평가
FALLBACK_VOL = 0.25                                 # 측정 불가 월 대체 상수(hsmm_final TARGET_VOL과 동일)


def _connect():
    """Railway 퍼블릭 프록시가 유휴 연결을 끊어도 무한 대기하지 않도록 keepalive 지정."""
    return psycopg2.connect(
        os.environ["DATABASE_URL"], connect_timeout=15,
        keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5,
    )

ANN = np.sqrt(252)
MEAS = ["20일 실현", "60일 실현", "20일 하방", "60일 하방"]

# FnSpace EconomyApi — 금리·환율·BM 차트용 (macro_rate_fx_bm.png)
FNSPACE_URL = "https://www.fnspace.com/Api/EconomyApi"
FNSPACE_KEY = os.environ.get("FNSPACE_API_KEY", "D0E7A9A250B8C43545C5")
FN_ITEMS = {"bond_10y": ("arKOIRKSDATB10", "국고채10년"),
            "usd_krw":  ("arKOFXUSDCD",    "원/달러환율")}

# 리포지토리 공통 팔레트(_biweekly_ratchet_test.py와 동일) + 색 이외 식별자(선 스타일)
COL_S = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
DASH = [(0, ()), (0, (5, 2)), (0, (1, 1.6)), (0, (7, 2, 1.5, 2))]
LW = [3.2, 2.4, 1.9, 1.4]          # 4종이 거의 겹치므로 두께를 내림차순으로 → 겹쳐도 전부 보임
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_RATE, C_FX, C_BM = "#2a78d6", "#eb6834", "#0b0b0b"   # 금리·환율·BM 대비 3색


# ─────────────────────────── 데이터 ───────────────────────────
def load():
    conn = _connect()
    ks = pd.read_sql("SELECT period p, value::float v FROM alpha_lab.macro_indicators "
                     "WHERE indicator='kospi' AND freq='D'", conn)
    bm = pd.read_sql("SELECT trade_date::date dt, adj_close::float p FROM alpha_lab.daily_price "
                     f"WHERE stock_code='{BM_CODE}' AND adj_close>0 ORDER BY 1", conn)
    conn.close()
    ks["p"] = pd.to_datetime(ks.p.str.slice(0, 10))
    kospi = ks.set_index("p")["v"].sort_index()
    bm["dt"] = pd.to_datetime(bm.dt)
    kodex = bm.set_index("dt")["p"].sort_index()

    P = pd.read_csv(A / "hsmm_final_path.csv", encoding="utf-8-sig").set_index("ym")
    lr = np.log(kospi / kospi.shift(1))

    # 각 ym의 월말 거래일 (hsmm_final과 동일 정의: KOSPI 지수의 그 달 마지막 관측일)
    mend = {}
    for dt_, pr in zip(kospi.index, pd.PeriodIndex(kospi.index, freq="M")):
        mend[pr.strftime("%Y-%m")] = dt_

    def rv(e, w):
        r = lr[lr.index <= e].iloc[-w:]
        return float(r.std() * ANN) if len(r) > 3 else np.nan

    def dv(e, w):
        """하방변동성 = semi-deviation. 양(+)의 수익률 날은 0으로 바꿔 '계산에 포함'한다.
           (음수일만 골라 std를 내는 방식이 아님 → 양수일도 분모에 남고, 중심은 0)"""
        r = lr[lr.index <= e].iloc[-w:]
        if len(r) < 4:
            return np.nan
        return float(np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2) * 252))

    def asof(s, e):
        x = s[s.index <= e]
        return float(x.iloc[-1]) if len(x) else np.nan

    rows = []
    for ym in P.index:
        e = mend.get(ym)
        if e is None:
            continue
        rows.append(dict(ym=ym, date=e, pbear=float(P.pbear[ym]), ret=float(P.ret[ym]),
                         bm=asof(kodex, e),
                         **{"20일 실현": rv(e, 20), "60일 실현": rv(e, 60),
                            "20일 하방": dv(e, 20), "60일 하방": dv(e, 60)}))
    X = pd.DataFrame(rows).set_index("ym")
    for k in MEAS:
        X[k] = X[k].fillna(FALLBACK_VOL)
    return X, kodex


def load_strategy(kodex):
    """backtest_cache에서 대상 전략의 월수익률 + 같은 구간 KODEX 200 수익률.
       기간 = PERIOD_START ~ PERIOD_END 고정. PERIOD_END가 월중에 걸리면 마지막 달은
       holdings_json 보유종목을 직접 평가한 부분월 수익으로 대체(회전비용은 캐시와의 차로 반영)."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT results_json, holdings_json FROM alpha_lab.backtest_cache "
                "WHERE name=%s AND universe='KOSPI' AND rebal_type='monthly'", (STRATEGY,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise SystemExit(f"backtest_cache에 '{STRATEGY}'(KOSPI/monthly)가 없습니다.")
    rj = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    hj = row[1] if isinstance(row[1], dict) else json.loads(row[1])
    rd, mr = rj["rebalance_dates"], rj["monthly_returns"]
    P0, P1 = pd.Timestamp(PERIOD_START), pd.Timestamp(PERIOD_END)

    def asof(s, e):
        x = s[s.index <= e]
        return float(x.iloc[-1]) if len(x) else np.nan

    out = []
    for i in range(len(mr)):
        d0, d1 = pd.Timestamp(rd[i]), pd.Timestamp(rd[i + 1])
        if d0 < P0 or d0 >= P1:
            continue
        strat, dend, partial = float(mr[i]), d1, False
        if d1 > P1:                                    # 마지막 부분월: d0 ~ PERIOD_END
            partial = True
            hold = hj[rd[i]]
            codes = tuple({h["종목코드"] for h in hold})
            px = pd.read_sql("SELECT stock_code, trade_date::date dt, adj_close::float p "
                             "FROM alpha_lab.daily_price WHERE stock_code IN %(c)s "
                             "AND adj_close>0 AND trade_date>=%(d0)s AND trade_date<=%(d1)s "
                             "ORDER BY 1,2", conn, params={"c": codes, "d0": str(d0.date()), "d1": str(d1.date())})
            px["dt"] = pd.to_datetime(px.dt)
            W = px.pivot_table(index="dt", columns="stock_code", values="p")

            def pv(d):                                 # d0→d 보유가치 변화(asof, 비중 재정규화)
                s = wsum = 0.0
                for h in hold:
                    c, w = h["종목코드"], h["비중(%)"] / 100
                    if c not in W.columns:
                        continue
                    s0 = W[c].loc[:d0].dropna(); s1 = W[c].loc[:d].dropna()
                    if not len(s0) or not len(s1):
                        continue
                    p0, p1 = float(s0.iloc[-1]), float(s1.iloc[-1])
                    if p0 > 0 and p1 > 0:
                        s += w * (p1 / p0); wsum += w
                return (s / wsum - 1.0) if wsum > 0 else np.nan

            r_cut, r_full = pv(P1), pv(d1)
            if np.isnan(r_cut):
                continue
            cost = (r_full - float(mr[i])) if not np.isnan(r_full) else 0.0   # 전략 자체 회전비용
            strat = r_cut - cost
            dend = kodex.index[kodex.index <= P1][-1]  # 실제 평가 거래일
        b0, b1 = asof(kodex, d0), asof(kodex, dend)
        # 익스포저는 리밸일이 속한 달의 '직전 월말'에 확정된 값 → 룩어헤드 없음
        # ※ d0 - MonthBegin(1)은 리밸일이 1일이 아니면 같은 달 1일로만 롤백(룩어헤드 버그) → Period-1로 계산
        out.append(dict(rebal=rd[i], d0=d0, d1=dend, strat=strat, partial=partial,
                        bm=(b1 / b0 - 1) if (b0 > 0 and b1 > 0) else np.nan,
                        prev_ym=(pd.Period(d0, freq="M") - 1).strftime("%Y-%m")))
    conn.close()
    return pd.DataFrame(out)


def fetch_fnspace(code, name):
    """FnSpace EconomyApi 일별 시계열 (PERIOD_START~PERIOD_END)."""
    fr, to = PERIOD_START.replace("-", ""), PERIOD_END.replace("-", "")
    r = requests.get(FNSPACE_URL, params={"key": FNSPACE_KEY, "format": "json",
                                          "item": code, "frdate": fr, "todate": to}, timeout=30)
    d = r.json()
    if d.get("errcd"):
        raise SystemExit(f"FnSpace API 오류 [{name} {code}]: {d.get('errmsg', '')}")
    data = d.get("dataset", [{}])[0].get("DATA", []) if d.get("dataset") else []
    if not data:
        raise SystemExit(f"FnSpace 데이터 없음 [{name} {code}]")
    s = pd.Series({pd.Timestamp(f"{x['DT'][:4]}-{x['DT'][4:6]}-{x['DT'][6:8]}"): float(x["AMOUNT"])
                   for x in data if x.get("AMOUNT") not in (None, "")}).sort_index()
    print(f"  {name}({code}): {len(s)}건  {s.index[0]:%Y-%m-%d} ~ {s.index[-1]:%Y-%m-%d}"
          f"  ({s.iloc[0]:,.2f} → {s.iloc[-1]:,.2f})")
    return s


# ─────────────────────────── 익스포저 ───────────────────────────
def exposure_path(pbear, meas):
    """vol-타겟(소프트 비대칭) + 리밸밴드. 래칫 없음."""
    tgt = pd.Series(meas).expanding(1).mean().values          # 판단시작~당월 확장평균
    cur = np.maximum(meas, VOL_FLOOR)
    cut = 1.0 - np.minimum(1.0, tgt / cur)
    raw = np.clip((1 - pbear) * (1.0 - pbear * cut), EXP_FLOOR, 1.0)
    out, held = raw.copy(), None
    for t in range(len(raw)):
        if held is None or abs(raw[t] - held) >= REBAL_BAND:
            held = round(raw[t] / STEP) * STEP
        out[t] = min(max(held, EXP_FLOOR), 1.0)
    return raw, out, tgt


def perf(r):
    r = np.asarray([x for x in r if not np.isnan(x)])
    if len(r) == 0:
        return {}
    eq = np.cumprod(1 + r); yrs = len(r) / 12
    cagr = eq[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(cagr=cagr, vol=vol, sharpe=(r.mean() * 12 / vol if vol else np.nan),
                mdd=mdd, calmar=(cagr / abs(mdd) if mdd else np.nan))


# ─────────────────────────── 차트 ───────────────────────────
def chrome(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)


def chart(X, EXPO, S, SR, kodex):
    """P_bear + 익스포저(60일 하방) + BM(KODEX 200) + 오버레이 낙폭(DD)을 한 그래프에.
       BM은 일별 시세를 로그 min-max 정규화(0~1)로 같은 축에. DD는 0선 아래 음수 영역에 실제 값으로."""
    plt.rcParams["font.family"] = KFONT
    plt.rcParams["axes.unicode_minus"] = False
    # 표시 구간도 백테스트 기간에 맞춤 (월말 패널이므로 기간 내 월말만)
    mask = (pd.to_datetime(X.date) >= pd.Timestamp(PERIOD_START)) & \
           (pd.to_datetime(X.date) <= pd.Timestamp(PERIOD_END))
    Xp = X[mask.values]
    xs = pd.to_datetime(Xp.date)
    sx = pd.to_datetime(S.d1)

    fig = plt.figure(figsize=(14, 9.6))
    fig.patch.set_facecolor("#fcfcfb")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.24], hspace=0.16)
    ax = fig.add_subplot(gs[0])
    chrome(ax)

    # BM: 일별 시세의 로그 가격을 0~1로 min-max 정규화 → 등락 모양을 전체 높이로 사용
    bmd = kodex[(kodex.index >= pd.Timestamp(PERIOD_START)) & (kodex.index <= pd.Timestamp(PERIOD_END))]
    lb = np.log(bmd.values)
    bn = (lb - lb.min()) / (lb.max() - lb.min())
    cum = bmd.iloc[-1] / bmd.iloc[0] - 1
    ax.plot(bmd.index, bn, color="#0b0b0b", lw=1.7,
            label=f"KODEX 200 (BM, 일별)   기간 누적 {cum:+.0%} · 로그 정규화")

    # P_bear: 파랑 + 옅은 면
    ax.fill_between(xs, 0, Xp.pbear, color="#2a78d6", alpha=0.12, lw=0)
    ax.plot(xs, Xp.pbear, color="#2a78d6", lw=2.0, label="HSMM P_bear")

    # 익스포저(60일 하방): 주황 계단
    e = EXPO["60일 하방"]["exp"][mask.values]
    ax.step(xs, e, where="post", color="#eb6834", lw=2.2, ls=(0, (5, 2)),
            label=f"익스포저 · 60일 하방   평균 {e.mean():.2f}")
    ax.axhline(EXP_FLOOR, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.annotate(f"하한 {EXP_FLOOR:.0%}", (xs.iloc[0], EXP_FLOOR), xytext=(3, 4),
                textcoords="offset points", fontsize=8, color=MUTED)

    # 오버레이(60일 하방) 낙폭: 0선 아래 음수 영역에 실제 값(-0.345 = -34.5%)으로
    r = np.nan_to_num(SR["60일 하방"]["ret"], nan=0.0)
    eq = np.cumprod(1 + r)
    ddv = eq / np.maximum.accumulate(eq) - 1
    ax.fill_between(sx, ddv, 0, color="#c81e1e", alpha=0.14, lw=0)
    ax.plot(sx, ddv, color="#c81e1e", lw=1.8,
            label=f"오버레이 낙폭(DD)   MDD {ddv.min()*100:.1f}%")
    im = int(np.argmin(ddv))
    ax.annotate(f"{ddv[im]*100:.1f}%", (sx.iloc[im], ddv[im]), xytext=(6, -11),
                textcoords="offset points", fontsize=9, color="#c81e1e", fontweight="bold")
    ax.axhline(0, color=INK, lw=0.9)

    ax.set_ylim(-0.40, 1.06)
    ax.set_ylabel("P_bear · 익스포저 (0~1) / BM 정규화 / 아래 음수영역 = DD", color=INK2, fontsize=9.5)
    lg = ax.legend(frameon=False, fontsize=10, loc="upper left")
    for t in lg.get_texts():
        t.set_color(INK)

    # ── 하단 성과표: BM / 원본 / 오버레이(60일 하방) ──
    axt = fig.add_subplot(gs[1]); axt.axis("off")
    def fmt(nm, m, ex=None):
        return [nm, (f"{ex:.2f}" if ex is not None else "—"), f"{m['cagr']*100:.1f}%",
                f"{m['sharpe']:.2f}", f"{m['mdd']*100:.1f}%", f"{m['calmar']:.2f}", f"{m['vol']*100:.1f}%"]
    e60 = SR["60일 하방"]["exp"]
    cell = [fmt("KODEX 200 (BM)", perf(S.bm.values)),
            fmt("오버레이 · 60일 하방", SR["60일 하방"]["perf"], float(np.nanmean(e60)))]
    t = axt.table(cellText=cell, colLabels=["계열", "평균 익스포저", "CAGR", "Sharpe", "MDD", "Calmar", "Vol"],
                  cellLoc="center", loc="upper center",
                  colWidths=[0.24, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.55)
    for (ri, ci), c in t.get_celld().items():
        c.set_edgecolor(GRID)
        if ri == 0:
            c.set_facecolor("#f0efec"); c.set_text_props(color=INK, fontweight="bold")
        elif ri == 2:                                       # 오버레이 행 강조
            c.set_facecolor("#fdf1ec"); c.set_text_props(color=INK, fontweight="bold")
        else:
            c.set_facecolor("#fcfcfb"); c.set_text_props(color=INK2)

    fig.suptitle("① 레짐 익스포저 — P_bear · 익스포저(60일 하방) · BM · 낙폭",
                 fontsize=13.5, fontweight="bold", color=INK, x=0.008, y=0.978, ha="left")
    fig.text(0.008, 0.948, f"{S.d0.iloc[0]:%Y-%m-%d} ~ {sx.iloc[-1]:%Y-%m-%d}",
             fontsize=10, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.004, 1, 0.936))
    p = OUT / "hsmm_exposure_volvariants.png"
    fig.savefig(p, dpi=140, facecolor=fig.get_facecolor())
    return p


def chart_macro(kodex):
    """금리(국고채10년) · 환율(원/달러) · BM(KODEX 200)을 한 그래프에.
       단위가 달라(%, 원, 지수) 시작=100 지수화, 원 수치는 범례 표기."""
    print("\n금리·환율 로드 (FnSpace)...")
    rate = fetch_fnspace(*FN_ITEMS["bond_10y"])
    fx = fetch_fnspace(*FN_ITEMS["usd_krw"])
    bm = kodex[(kodex.index >= pd.Timestamp(PERIOD_START)) & (kodex.index <= pd.Timestamp(PERIOD_END))]

    plt.rcParams["font.family"] = KFONT
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor("#fcfcfb")
    chrome(ax)
    for s, c, lab in [
        (rate, C_RATE, f"국고채 10년   {rate.iloc[0]:.2f}% → {rate.iloc[-1]:.2f}%"),
        (fx,   C_FX,   f"원/달러 환율   {fx.iloc[0]:,.0f}원 → {fx.iloc[-1]:,.0f}원"),
        (bm,   C_BM,   f"KODEX 200 (BM)   {bm.iloc[-1]/bm.iloc[0]-1:+.0%}"),
    ]:
        ax.plot(s.index, s / s.iloc[0] * 100, color=c, lw=1.9, label=lab)
    ax.axhline(100, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_ylabel("시작=100 지수화", color=INK2, fontsize=10)
    lg = ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    for t in lg.get_texts():
        t.set_color(INK)
    fig.suptitle("금리 · 환율 · BM(KODEX 200)", fontsize=13.5, fontweight="bold",
                 color=INK, x=0.008, y=0.975, ha="left")
    fig.text(0.008, 0.935, f"{PERIOD_START} ~ {PERIOD_END}", fontsize=10, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.008, 1, 0.92))
    p = OUT / "macro_rate_fx_bm.png"
    fig.savefig(p, dpi=140, facecolor=fig.get_facecolor())
    pd.DataFrame({"bond_10y": rate, "usd_krw": fx, "kodex200": bm}).to_csv(
        OUT / "macro_rate_fx_bm.csv", encoding="utf-8-sig")
    return p


def chart_features():
    """HSMM emission 피처(breadth·newlow·trend) 월별 표를 이미지로.
       hsmm_final.build_features()를 그대로 불러 모델과 동일한 값 보장."""
    import importlib.util
    print("\nEmission 피처 로드 (hsmm_final.build_features)...")
    spec = importlib.util.spec_from_file_location("hsmm_final", A / "hsmm_final.py")
    hf = importlib.util.module_from_spec(spec); spec.loader.exec_module(hf)
    df = hf.build_features()[0][["breadth", "newlow", "trend"]].round(4)
    ym0 = pd.Timestamp(PERIOD_START).strftime("%Y-%m")
    ym1 = pd.Timestamp(PERIOD_END).strftime("%Y-%m")
    d = df.loc[ym0:ym1]                                    # 백테스트 기간과 동일 구간
    d.index.name = "ym"
    d.to_csv(OUT / "hsmm_emission_features.csv", encoding="utf-8-sig")
    # Bull 판정 구간(스무딩 P_bear < 0.2) = 표에서 빨간 볼드
    Pb = pd.read_csv(A / "hsmm_final_path.csv", encoding="utf-8-sig").set_index("ym")["pbear"]
    bull = {ym for ym in d.index if float(Pb.get(ym, 1.0)) < 0.20}

    plt.rcParams["font.family"] = KFONT
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(13.5, 16.5)); fig.patch.set_facecolor("#fcfcfb")
    half = (len(d) + 1) // 2
    for j, h in enumerate([d.iloc[:half], d.iloc[half:]]):
        ax = fig.add_subplot(1, 2, j + 1); ax.axis("off")
        cell = [[ym, f"{r.breadth:.4f}", f"{r.newlow:.4f}", f"{r.trend:.4f}"] for ym, r in h.iterrows()]
        t = ax.table(cellText=cell, colLabels=["월", "breadth", "newlow", "trend"],
                     cellLoc="center", loc="upper center")
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.32)
        for (ri, ci), c in t.get_celld().items():
            c.set_edgecolor(GRID)
            if ri == 0:
                c.set_facecolor("#f0efec"); c.set_text_props(color=INK, fontweight="bold")
            else:
                c.set_facecolor("#fcfcfb" if ri % 2 else "#f6f5f2")
                if h.index[ri - 1] in bull:
                    c.set_text_props(color="#c81e1e", fontweight="bold")
                else:
                    c.set_text_props(color=INK2)
    fig.suptitle("HSMM Emission 피처 — breadth · 52주 신저가비율(newlow) · 추세(trend)",
                 fontsize=13.5, fontweight="bold", color=INK, x=0.008, y=0.985, ha="left")
    fig.text(0.008, 0.968, f"{ym0} ~ {ym1}", fontsize=10, color=INK2, ha="left")
    fig.text(0.992, 0.968, "빨간 볼드 = Bull 판정 구간 (P_bear < 0.2)", fontsize=9.5,
             color="#c81e1e", ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.962))
    p = OUT / "hsmm_emission_features.png"
    fig.savefig(p, dpi=140, facecolor=fig.get_facecolor())
    return p


# ─────────────────────────── main ───────────────────────────
def main():
    X, kodex = load()
    pbear = X.pbear.values
    EXPO = {}
    for k in MEAS:
        raw, exp, tgt = exposure_path(pbear, X[k].values)
        EXPO[k] = dict(raw=raw, exp=exp, tgt=tgt)

    # ── 전략에 익스포저 적용 (현금 0% 가정) ──
    S = load_strategy(kodex)
    S = S[S.prev_ym.isin(X.index)].reset_index(drop=True)      # 익스포저가 확정된 달만
    emap = {k: dict(zip(X.index, EXPO[k]["exp"])) for k in MEAS}
    SR = {}
    for k in MEAS:
        e = S.prev_ym.map(emap[k]).values
        r = e * S.strat.values
        SR[k] = dict(exp=e, ret=r, perf=perf(r))

    bar = "=" * 104
    print(bar)
    print(f"  HSMM 익스포저 오버레이 — {STRATEGY}   ({S.d0.iloc[0]:%Y-%m-%d} ~ {S.d1.iloc[-1]:%Y-%m-%d}, {len(S)}개월"
          + (", 마지막 달은 부분월" if bool(S.partial.iloc[-1]) else "") + ")")
    print(bar)
    print(f"  P_bear은 4종 공통. 익스포저만 vol-타겟 분모에 따라 갈림. 래칫 미적용, 월 1회 리밸.")
    print(f"  하한 {EXP_FLOOR:.0%} · 분모하한 {VOL_FLOOR:.0%} · 리밸밴드 {REBAL_BAND} · 스텝 {STEP}"
          f" · 하방 = semi-deviation(양수일 0 처리)")

    print("\n" + "-" * 104)
    print(f"  {'계열':30}{'평균익스포저':>12}   {'CAGR':>8}{'Sharpe':>9}{'MDD':>10}{'Calmar':>9}{'Vol':>8}")
    print("-" * 104)

    def row(nm, m, ex=None):
        print(f"  {nm:32}{(f'{ex:.3f}' if ex is not None else '—'):>10}   "
              f"{m['cagr']*100:>7.1f}%{m['sharpe']:>9.2f}{m['mdd']*100:>9.1f}%"
              f"{m['calmar']:>9.2f}{m['vol']*100:>7.1f}%")
        return dict(계열=nm, 평균익스포저=(round(ex, 3) if ex is not None else None),
                    CAGR=round(m["cagr"] * 100, 2), Sharpe=round(m["sharpe"], 3),
                    MDD=round(m["mdd"] * 100, 2), Calmar=round(m["calmar"], 3),
                    Vol=round(m["vol"] * 100, 2))

    rows = [row("KODEX 200 (BM)", perf(S.bm.values)),
            row(f"{STRATEGY} 원본", perf(S.strat.values))]
    for k in MEAS:
        rows.append(row(f"오버레이 · {k}", SR[k]["perf"], float(np.nanmean(SR[k]["exp"]))))
    print("-" * 104)
    print("  ※ 오버레이 = 전략수익 x 익스포저 (나머지는 현금, 수익 0% 가정). 오버레이 자체 거래비용 미반영.")

    out = pd.DataFrame({"rebal": S.rebal, "d_end": S.d1, "prev_ym": S.prev_ym,
                        "pbear": S.prev_ym.map(dict(zip(X.index, pbear))),
                        "strat_ret": S.strat, "bm_ret": S.bm})
    for k in MEAS:
        out[f"exp_{k}"] = SR[k]["exp"]
        out[f"ret_{k}"] = SR[k]["ret"]
    csvp = OUT / "hsmm_exposure_volvariants.csv"
    out.to_csv(csvp, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "hsmm_exposure_volvariants_summary.csv",
                              index=False, encoding="utf-8-sig")

    png = chart(X, EXPO, S, SR, kodex)
    png2 = chart_macro(kodex)
    png3 = chart_features()
    print(f"\n  차트 → {png}")
    print(f"  차트 → {png2}")
    print(f"  표   → {png3}")
    print(f"  경로 → {csvp}")


if __name__ == "__main__":
    main()
