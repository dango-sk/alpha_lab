"""
analysis/hsmm_final.py

최종 레짐 모델 = HSMM (Hidden Semi-Markov / 명시적 지속기간 모델), 박사님 설계 반영.

HSMM 핵심 (IO-HMM 대비 차이):
  - 각 상태에 **명시적 지속기간 분포(음이항 NB)**를 학습 → 기하분포(HMM)와 달리 최소 체류기간이
    생겨 whipsaw를 구조적으로 억제. (전이가 "매달 동전던지기"가 아니라 "얼마나 머물지"를 모델링)
  - 디코딩은 Viterbi(이산)가 아니라 **확장상태(state×경과기간) forward 필터** → filtered 연속확률(§5).

피처 역할 분리 (박사님 §3):
  - Emission (상태=시장 내부 건전성+방향) : breadth(200일선 위 종목비율) / 52주신저가비율 / 추세(log KOSPI÷200일선)
      → 2-state Gaussian (Ledoit-Wolf 슈링키지). breadth·신저가=참여도, trend=방향 앵커. 변동성은 제외(§1·§5).
  - Transition (전환 방아쇠·변화율 Z): 환율3M변화(레벨스트레스) / 외국인 월순매수
      → 스트레스 스칼라가 **지속기간 exit hazard를 변조**(입력의존 duration). 스트레스↑ → Bull 조기이탈.

학습 (박사님 §2): emission=가중 Baum-Welch(시간감쇠·Ledoit-Wolf), duration=세그먼트 길이서 NB 적합.
  2-state · 5년 롤링창 · 순수 웜스타트(연1회만 재추정, 나머지 달은 파라미터 고정하고 필터만).
출력 (박사님 §5·§1·§6): HSMM filtered 연속확률 → EMA스무딩 → vol-targeting 익스포저(하한0.2,
  하방변동성 반영) → Z단계 리밸밴드 → MDD 중심 자체 리포트.
데이터: 외국인 순매수 DB(2000~), breadth/신저가용 종목시세 2017~ 병목 → 판단 2018-01부터.
유니버스: breadth/신저가 = **KOSPI 상장기업만** (fnspace_master 월별 스냅샷 point-in-time,
  우선주·ETF·리츠·스팩 제외, 최근 ~800종목. 2026-08 변경 — 이전엔 KOSPI+KOSDAQ 전 시장).

사용: python analysis/hsmm_final.py   (백그라운드 권장)
"""
import os, sys, warnings
import numpy as np, pandas as pd, psycopg2
from datetime import timedelta
from pathlib import Path
from scipy.stats import nbinom
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings("ignore")
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
sys.path.insert(0, str(BASE))
A = Path(__file__).parent

PANEL_START, DECIDE_START = "2017-01", "2018-01"
WINDOW_M, REFIT_EVERY = 60, 12                         # §2 5년 롤링창 / 순수 웜스타트: 연1회 재추정, 나머지는 필터만
DMAX, KAPPA = 60, 0.80                                 # HSMM 최대 지속기간(월) / 방아쇠→전이 변조 민감도
USE_DURATION = True                                    # True =HSMM 명시적 지속기간(음이항 NB) — 채택본
                                                       # False=plain 2-state HMM(지속기간 제약 없음, 대조군)
                                                       # ※이산 성과는 두 경우 동일, 연속은 HSMM이 소폭 열위(회전↑). 명시적 지속기간 = 진짜 HSMM.
TARGET_VOL, EXP_FLOOR, VOL_FLOOR = 0.25, 0.20, 0.08    # §5 vol-targeting (하방변동성 기준, 소프트 비대칭)
# ※ TARGET_VOL은 이제 목표값이 아니라 '결측 대체용 상수'다. 실제 목표변동성은 main()에서
#    백테스트 시작~당월의 하방변동성 확장평균(tgt_vol)으로 매월 새로 계산한다.
HL = 36.0                                              # 시간감쇠 half-life(월)
PBEAR_EMA = 0.5                                        # §5 P_bear EMA 스무딩
REBAL_BAND = 0.15                                      # §6 익스포저 변화가 밴드 넘을 때만 리밸(0.05 스텝)
T_IN, T_OUT = 0.60, 0.40                               # 이산 레짐 히스테리시스(리포트용)
WIN, SEED, EPS = 6, 42, 1e-9
EMIS_COLS = ["breadth", "newlow", "trend"]            # 건전성(breadth 200일선·52주신저가) + 방향(추세)
TRAN_COLS = ["fx3m", "fflow"]                         # 변화율 방아쇠 (환율 레벨보정, 외국인 3M 누적 순매수)


# ─────────────────────────── 데이터/피처 ───────────────────────────
FEAT_CACHE = A / ".cache" / "hsmm_features.pkl"   # build_features() 결과 캐시(체인 3개 스크립트가 공유)


def _connect():
    """Railway 퍼블릭 프록시는 유휴 연결을 조용히 끊는다. keepalive가 없으면 죽은 소켓의
    read에서 무한 대기(CPU 0%로 영원히 멈춤)하므로 반드시 지정한다."""
    return psycopg2.connect(
        os.environ['DATABASE_URL'],
        connect_timeout=15,
        keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5,
    )


def build_features(use_cache=True):
    """DB에서 월별 레짐 피처 패널을 만든다. 결정론적이므로 디스크에 캐시한다.
    갱신하려면 --refresh 플래그 또는 analysis/.cache/hsmm_features.pkl 삭제."""
    if use_cache and "--refresh" not in sys.argv and FEAT_CACHE.exists():
        print(f"[cache] 피처 재사용: {FEAT_CACHE.relative_to(BASE)} "
              f"(갱신하려면 --refresh)", flush=True)
        return pd.read_pickle(FEAT_CACHE)
    print("[db] 일별 주가 로드 중(658만 행, ~1.5분)...", flush=True)
    conn = _connect()

    def mac(ind):
        x = pd.read_sql(f"SELECT period p, value::float v FROM alpha_lab.macro_indicators WHERE indicator='{ind}' AND freq='D'", conn)
        x['p'] = pd.to_datetime(x['p'].str.slice(0, 10)); return x.set_index('p')['v'].sort_index()

    kospi, usdkrw = mac('kospi'), mac('usd_krw')
    frn = mac('investor_foreign_kospi')  # 외국인 순매수(일별, DB 2000~)
    d = pd.read_sql("SELECT trade_date::date dt, stock_code, close::float close FROM alpha_lab.daily_price "
                    "WHERE close IS NOT NULL AND trade_date>='2017-01-01'", conn)
    # breadth/신저가 유니버스 = KOSPI 상장기업만 (2026-08 변경. 이전엔 daily_price 전체 = KOSPI+KOSDAQ+ETF·우선주)
    #   fnspace_master 월별 스냅샷에서 market='KOSPI' AND 섹터 있음 → ETF(섹터 NULL)·우선주·리츠·스팩(마스터 부재) 제외
    #   point-in-time 소속(이전상장 반영). 판정(P_bear)은 전 시장 대비 사실상 동일, MDD 소폭 개선 확인 후 채택.
    snap = pd.read_sql("SELECT snapshot_date ym, stock_code FROM alpha_lab.fnspace_master "
                       "WHERE market='KOSPI' AND sec_cd_nm IS NOT NULL", conn)
    conn.close()
    d['dt'] = pd.to_datetime(d['dt'])
    wide = d.pivot_table(index='dt', columns='stock_code', values='close').sort_index()
    snap['code'] = snap.stock_code.str[1:]                              # 'A005930' → '005930'
    snap = snap[snap.code.str.match(r'^\d{5}0$')]                       # 보통주 코드(끝자리 0)만 — 우선주 잔재 방지
    by_ym = snap.groupby('ym')['code'].apply(set).to_dict(); yms_snap = sorted(by_ym)
    K = pd.DataFrame(False, index=wide.index, columns=wide.columns)     # 그 달 KOSPI 기업 소속 마스크(일×종목)
    day_ym = wide.index.to_period('M').strftime('%Y-%m')
    for _ym in np.unique(day_ym):
        avail = [s for s in yms_snap if s <= _ym]
        if avail: K.loc[day_ym == _ym, wide.columns.isin(by_ym[avail[-1]])] = True
    ma = wide.rolling(200, min_periods=100).mean(); vld = wide.notna() & ma.notna() & K  # breadth: 200일선 기준
    breadth = ((wide > ma) & vld).sum(axis=1) / vld.sum(axis=1).clip(lower=1); breadth = breadth[vld.sum(axis=1) > 50]
    rmn = wide.rolling(252, min_periods=60).min(); okK = wide.notna() & K
    newlow = ((wide <= rmn) & okK).sum(axis=1) / okK.sum(axis=1).clip(lower=1)
    lr = np.log(kospi / kospi.shift(1))
    kma = kospi.rolling(200, min_periods=100).mean()          # 추세용 200일선(지수)
    fx_m1y = usdkrw.rolling(252, min_periods=120).mean()      # 환율 레벨 기준(1년 평균)

    ym = pd.PeriodIndex(kospi.index, freq='M'); last = {}
    for dt_, p in zip(kospi.index, ym): last[p] = dt_
    mends = [last[p] for p in sorted(last) if last[p] >= pd.Timestamp(PANEL_START)]

    def asof(s, e): x = s[s.index <= e]; return x.iloc[-1] if len(x) else np.nan
    def pctc(s, e, dy): c = asof(s, e); p = s[s.index <= e - timedelta(days=dy)]; return (c / p.iloc[-1] - 1) if len(p) and p.iloc[-1] else np.nan
    def rv(e, win): r = lr[lr.index <= e].iloc[-win:]; return r.std() * np.sqrt(252) if len(r) > 3 else np.nan
    def dv(e, win):
        """하방변동성 = semi-deviation. 양(+)수익일을 0으로 바꿔 계산에 포함(분모에 남고 중심은 0)."""
        r = lr[lr.index <= e].iloc[-win:]
        return np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2) * 252) if len(r) > 3 else np.nan
    def flow(e, dy): x = frn[(frn.index <= e) & (frn.index > e - timedelta(days=dy))]; return x.sum() if len(x) else np.nan

    base, Px, rvol_l, dvol_l = [], [], [], []
    for e in mends:
        km, kp = asof(kma, e), asof(kospi, e)
        trend = np.log(kp / km) if (km and kp and not np.isnan(km) and km > 0 and kp > 0) else 0.0   # §3 방향 앵커(로그)
        fx_chg = pctc(usdkrw, e, 90); lvl, lvlm = asof(usdkrw, e), asof(fx_m1y, e)                    # §3 환율=변화율
        fx_ctx = fx_chg * (lvl / lvlm) if (lvlm and not np.isnan(lvlm) and lvlm > 0 and not np.isnan(fx_chg)) else (fx_chg if not np.isnan(fx_chg) else 0.0)  # 레벨 스트레스 가중
        base.append(dict(breadth=asof(breadth, e), newlow=asof(newlow, e), trend=trend,
                         fx3m=fx_ctx, fflow=flow(e, 90)))   # 외국인 3M(90일) 누적 순매수 (기존 30일→장기)
        Px.append(kp); rvol_l.append(rv(e, 20)); dvol_l.append(dv(e, 60))
    df = pd.DataFrame(base, index=[pd.Period(e, freq='M').strftime('%Y-%m') for e in mends])[EMIS_COLS + TRAN_COLS].fillna(0.0)
    Px = np.array(Px); n = len(df); yms = list(df.index)
    ret = np.array([(Px[i + 1] / Px[i] - 1) if i + 1 < n else np.nan for i in range(n)])   # t→t+1 월수익
    rvol = np.nan_to_num(np.array(rvol_l), nan=TARGET_VOL)                                  # 실현변동성(§5 분모)
    dvol = np.nan_to_num(np.array(dvol_l), nan=TARGET_VOL)                                  # 하방변동성(§7-② 분모 결합)
    dd6 = np.full(n, np.nan)                                                                # 향후6M 최대낙폭(이벤트 정의)
    for i in range(n - 1):
        if i + 6 < n:
            path = pd.concat([pd.Series([Px[i]], index=[mends[i]]), kospi[(kospi.index > mends[i]) & (kospi.index <= mends[i + 6])]])
            dd6[i] = (path / path.cummax() - 1).min() * 100
    out = (df, yms, n, ret, rvol, dvol, dd6)
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, FEAT_CACHE)
    print(f"[cache] 피처 저장: {FEAT_CACHE.relative_to(BASE)}", flush=True)
    return out


def roll_z(df, cols, win=36):
    """전이 피처: 후행 rolling Z-score(스케일 드리프트 견딤, 아웃라이어 clip). min_periods=12."""
    out = df.copy()
    for c in cols:
        m = df[c].rolling(win, min_periods=12).mean()
        s = df[c].rolling(win, min_periods=12).std().replace(0, np.nan)
        out[c] = ((df[c] - m) / s).fillna(0.0).clip(-4, 4)
    return out


# ─────────────────────────── HSMM 엔진 ───────────────────────────
def bear_score(means):
    """emission 평균으로 Bear state 판정: breadth↓ newlow↑ 추세↓ (약한/하방 시장)."""
    return -means[:, 0] + means[:, 1] - means[:, 2]


def emis_logB(X, means, covs):
    n, d = X.shape; logB = np.zeros((n, 2))
    for k in range(2):
        L = np.linalg.cholesky(covs[k])
        sol = np.linalg.solve(L, (X - means[k]).T)
        logB[:, k] = -0.5 * ((sol ** 2).sum(0) + 2 * np.log(np.diag(L)).sum() + d * np.log(2 * np.pi))
    return logB


def forward_backward(logB, A, pi, w):
    """고정 전이행렬 A(2×2) Baum-Welch (emission 추정용). gamma, xi누적 반환."""
    n, K = logB.shape
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    alpha = np.zeros((n, K)); c = np.zeros(n)
    alpha[0] = pi * B[0]; c[0] = alpha[0].sum() + EPS; alpha[0] /= c[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ A) * B[t]; c[t] = alpha[t].sum() + EPS; alpha[t] /= c[t]
    beta = np.zeros((n, K)); beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta; gamma /= gamma.sum(axis=1, keepdims=True) + EPS
    xi = np.zeros((K, K))
    for t in range(n - 1):
        m = (alpha[t][:, None] * A) * (B[t + 1] * beta[t + 1])[None, :]
        xi += w[t + 1] * m / (m.sum() + EPS)
    return gamma, xi


def m_step_emis(X, gamma, w):
    n, d = X.shape; means = np.zeros((2, d)); covs = np.zeros((2, d, d))
    for k in range(2):
        r = gamma[:, k] * w; R = r.sum() + EPS
        mu = (r[:, None] * X).sum(0) / R
        Xc = X - mu; C = (r[:, None] * Xc).T @ Xc / R
        hard = gamma[:, k] > 0.5
        delta = float(LedoitWolf().fit(X[hard]).shrinkage_) if hard.sum() >= d + 2 else 0.5
        mt = np.trace(C) / d
        covs[k] = (1 - delta) * C + delta * mt * np.eye(d) + 1e-6 * np.eye(d)
        means[k] = mu
    return means, covs


COLD_VAR_FLOOR = 1e-6          # 콜드 스타트에서 '앵커 외' 피처에 부여하는 초기 분산 바닥값
COLD_ANCHOR = 0               # 적합 분산을 그대로 쓰는 피처 인덱스 = EMIS_COLS[0] = breadth


def cold_emission(X):
    """콜드 스타트 emission 초기값 — **의도적인 강한 정규화**.

    ■ 왜 '정상적인' 초기화를 쓰지 않는가
      판정 시작 시점의 학습창은 13개월뿐이다(패널 2017-01~, 판정 2018-01~. daily_price가
      2017년부터라 창을 더 늘릴 수 없는 데이터 제약). 13샘플·2상태·3피처에서 EM을
      '각 피처의 적합 분산'으로 출발시키면 두 상태의 우도가 비슷해 식별이 안 되고,
      한 상태가 전체 가중치를 흡수하며 붕괴한다(공분산 NaN → cholesky 실패, 또는 P_bear 0 고착).

    ■ 그래서 쓰는 정책
      앵커 피처(EMIS_COLS[0] = breadth)에만 GaussianHMM이 적합한 분산을 주고,
      나머지 피처의 초기 분산은 바닥값(COLD_VAR_FLOOR)으로 고정한다.
      나머지 축 방향으로 밀도가 매우 뾰족해져 두 상태가 강제로 분리되고,
      이후 웜스타트(m_step_emis + Ledoit-Wolf)가 공분산을 정상 범위로 회복시킨다.
      즉 '초기 상태 분리를 강제하는 정규화'이며, 현재 production 성과의 전제다.

    ■ 바꾸려면 학습창 확대가 선행되어야 한다
      2026-08-10 검증(analysis/hsmm_slowbear.py):
        · 모든 피처에 적합 분산을 주도록 '정상화' → Bear 0개월, 수치가드 60회 발동
        · 거기에 4번째 피처(ma200_slope_60) 추가 → 전 구간 P_bear=0, 가드 140회
        · 장기 패널(2004~, analysis/hsmm_longrun.py)에서도 slow bear 개선 없음
      따라서 이 초기화는 유지한다. 변경은 장기 패널로 창을 채운 뒤에만 검토한다.

    ※ 이 함수는 walk_forward의 최초 적합(params is None)에서만 호출되고,
      이후에는 웜스타트로 이어진다. 그래서 전체 궤적이 이 초기값에 의존한다.
    """
    d = X.shape[1]
    hm = GaussianHMM(2, "diag", n_iter=50, random_state=SEED)
    hm.fit(X)
    covs = np.zeros((2, d, d))
    for k in range(2):
        var = np.zeros(d)                                        # 앵커 외 피처는 0 →
        var[COLD_ANCHOR] = float(np.asarray(hm.covars_[k])[COLD_ANCHOR, COLD_ANCHOR])
        covs[k] = np.diag(var) + COLD_VAR_FLOOR * np.eye(d)      # → 바닥값만 남는다
    return dict(means=hm.means_.copy(), covs=covs,
                Amat=np.array([[0.9, 0.1], [0.2, 0.8]]), pi=np.array([0.85, 0.15]))


def fit_emission(X, w, init, n_iter):
    """가중 Baum-Welch(시간감쇠·Ledoit-Wolf). A는 세그먼트 추출용으로만 학습."""
    means, covs, Amat, pi = init['means'], init['covs'], init['Amat'], init['pi']
    gamma = None
    for _ in range(n_iter):
        logB = emis_logB(X, means, covs)
        gamma, xi = forward_backward(logB, Amat, pi, w)
        pi = gamma[0] / (gamma[0].sum() + EPS)
        Amat = xi / (xi.sum(axis=1, keepdims=True) + EPS)
        means, covs = m_step_emis(X, gamma, w)
    return dict(means=means, covs=covs, Amat=Amat, pi=pi, gamma=gamma)


def state_durations(path):
    durs = {0: [], 1: []}; s, L = int(path[0]), 1
    for k in range(1, len(path)):
        if int(path[k]) == s: L += 1
        else: durs[s].append(L); s, L = int(path[k]), 1
    durs[s].append(L); return durs


def dur_pmf(durs, dmax=DMAX):
    """세그먼트 길이 → 지속기간 분포. 음이항(NB) 적합, 부족하면 기하 fallback. d=1..dmax."""
    d = np.arange(1, dmax + 1)
    if len(durs) >= 3:
        m, v = float(np.mean(durs)), float(np.var(durs, ddof=1))
        if v > m > 0:
            p = m / v; r = m * m / (v - m)
            if r > 0:
                pmf = nbinom.pmf(d - 1, r, p); s = pmf.sum()
                if s > 0 and np.isfinite(s): return pmf / s
    m = max(float(np.mean(durs)) if len(durs) else 6.0, 1.5); q = 1.0 / m
    pmf = (1 - q) ** (d - 1) * q; return pmf / pmf.sum()


def to_hazard(pmf):
    """지속기간 pmf → exit hazard h(d)=P(끝|d까지 생존). 마지막은 강제 종료(1)."""
    surv = pmf[::-1].cumsum()[::-1]                # P(D>=d)
    haz = pmf / (surv + EPS); haz[-1] = 1.0
    return np.clip(haz, 1e-6, 1 - 1e-6)


def hsmm_filter(logB, stress, haz, bear, pi):
    """확장상태(state×경과기간) forward 필터 → filtered P(state_t). exit hazard를 스트레스로 변조."""
    T = logB.shape[0]; D = haz.shape[1]; bull = 1 - bear
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    haz_l = np.log(haz / (1 - haz))               # logit(base hazard)
    filt = np.zeros((T, 2))
    a = np.zeros((2, D)); a[:, 0] = pi * B[0]; a /= a.sum() + EPS; filt[0] = a.sum(1)
    for t in range(1, T):
        z = stress[t]
        he = np.empty_like(haz)
        he[bull] = 1 / (1 + np.exp(-(haz_l[bull] + KAPPA * z)))   # 스트레스↑ → Bull 이탈↑
        he[bear] = 1 / (1 + np.exp(-(haz_l[bear] - KAPPA * z)))   # 스트레스↑ → Bear 유지↑
        he = np.clip(he, 1e-6, 1 - 1e-6); he[:, -1] = 1.0
        cont = a * (1 - he); endm = (a * he).sum(1)
        nxt = np.zeros((2, D))
        nxt[:, 1:] = cont[:, :-1]                  # 경과기간 d→d+1
        nxt[bull, 0] = endm[bear]; nxt[bear, 0] = endm[bull]   # 종료→상대상태 d=1
        nxt = nxt * B[t][:, None]; nxt /= nxt.sum() + EPS
        a = nxt; filt[t] = a.sum(1)
    return filt


def plain_hmm_filter(logB, stress, A0, pi, bear):
    """박사님 스펙: 2-state 비동차 forward 필터 → filtered 연속확률 P(state_t|X_1:t).
       지속기간 제약 없음. transition 스트레스가 기저 전이행렬 A0를 시점별 변조."""
    T = logB.shape[0]; bull = 1 - bear
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
    def sig(z): return 1.0 / (1.0 + np.exp(-z))
    a = pi * B[0]; a /= a.sum() + EPS
    filt = np.zeros((T, 2)); filt[0] = a
    for t in range(1, T):
        z = stress[t]
        p_bb = sig(logit(A0[bull, bear]) + KAPPA * z)   # 스트레스↑ → Bull→Bear↑
        p_rb = sig(logit(A0[bear, bull]) - KAPPA * z)   # 스트레스↑ → Bear→Bull↓(회복 지연)
        At = np.empty((2, 2))
        At[bull, bear] = p_bb; At[bull, bull] = 1 - p_bb
        At[bear, bull] = p_rb; At[bear, bear] = 1 - p_rb
        a = (a @ At) * B[t]; a /= a.sum() + EPS
        filt[t] = a
    return filt


# ─────────────────────────── walk-forward (웜스타트) ───────────────────────────
def walk_forward(df, yms, n):
    start = yms.index(DECIDE_START) if DECIDE_START in yms else 12
    TRz = roll_z(df, TRAN_COLS).values
    EM = df[EMIS_COLS].values
    stress_raw = TRz[:, 0] - TRz[:, 1]                           # 환율↑ + 외국인유출(-flow) = 스트레스↑
    pbear = np.full(n, np.nan); params, sc, last_refit = None, None, -10 ** 9
    _mode = f"HSMM(Dmax {DMAX})" if USE_DURATION else "plain 2-state HMM(박사님 스펙, filtered)"
    print(f"walk-forward: 판단 {yms[start]}~{yms[-1]} ({n - start}개월), {_mode} 웜스타트(창 {WINDOW_M}월)...", flush=True)
    for t in range(start, n):
        lo = max(0, t + 1 - WINDOW_M); Xr = EM[lo:t + 1]; nw = len(Xr)
        w = 0.5 ** ((nw - 1 - np.arange(nw)) / HL)               # 시간감쇠(최근=1)
        if params is None or (t - last_refit) >= REFIT_EVERY:
            sc = StandardScaler().fit(Xr)
            params = fit_emission(sc.transform(Xr), w, cold_emission(sc.transform(Xr)) if params is None else params,
                                  40 if params is None else 10)
            last_refit = t
        Xz = sc.transform(Xr)
        means, covs = params['means'], params['covs']
        logB = emis_logB(Xz, means, covs)
        bear = int(np.argmax(bear_score(means)))
        sw = stress_raw[lo:t + 1]; sw = (sw - sw.mean()) / (sw.std() + EPS)
        if USE_DURATION:                                   # HSMM(명시적 지속기간) — 스펙 밖 옵션
            gamma, _ = forward_backward(logB, params['Amat'], params['pi'], w)
            durs = state_durations(gamma.argmax(1))
            haz = np.vstack([to_hazard(dur_pmf(durs[0])), to_hazard(dur_pmf(durs[1]))])
            filt = hsmm_filter(logB, sw, haz, bear, params['pi'])
        else:                                              # 박사님 스펙: plain 2-state HMM filtered
            filt = plain_hmm_filter(logB, sw, params['Amat'], params['pi'], bear)
        pbear[t] = filt[-1, bear]
        if t % 12 == 0 or t == n - 1:
            print(f"  {yms[t]}  P(bear)={pbear[t]:.2f}", flush=True)
    return pbear, start


# ─────────────────────────── 평가/리포트 ───────────────────────────
def perf(rets):
    r = np.asarray([x for x in rets if not np.isnan(x)])
    if len(r) == 0: return {}
    eq = np.cumprod(1 + r); yrs = len(r) / 12
    cagr = eq[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12); shp = (r.mean() * 12) / vol if vol else float('nan')
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(cagr=cagr, vol=vol, sharpe=shp, mdd=mdd, calmar=(cagr / abs(mdd) if mdd else float('nan')))


def line(name, m):
    if not m: return f"  {name:16} (n/a)"
    return f"  {name:16} {m['cagr']*100:>6.1f}% {m['sharpe']:>7.2f} {m['mdd']*100:>7.1f}% {m['calmar']:>7.2f} {m['vol']*100:>6.1f}%"


def mdd_window(rets, labels):
    pairs = [(labels[i], rets[i]) for i in range(len(rets)) if not np.isnan(rets[i])]
    if not pairs: return None
    labs = [p[0] for p in pairs]; eq = np.cumprod([1 + p[1] for p in pairs])
    dd = eq / np.maximum.accumulate(eq) - 1
    tr = int(np.argmin(dd)); pk = int(np.argmax(eq[:tr + 1]))
    return labs[pk], labs[tr], dd[tr] * 100


def main():
    df, yms, n, ret, rvol, dvol, dd6 = build_features()
    pbear_raw, start = walk_forward(df, yms, n)
    idx = list(range(start, n))

    # ── P_bear EMA 스무딩 (§5) ──
    pbear = pbear_raw.copy()
    for t in idx[1:]:
        pbear[t] = PBEAR_EMA * pbear_raw[t] + (1 - PBEAR_EMA) * pbear[t - 1]

    # ── 연속 익스포저 (§5): 소프트 비대칭 vol-타겟 (하방변동성 기준) ──
    #    변동성 축소분(목표 대비 초과분)을 Pbear에 '비례' 적용 → 불장(Pbear↓)엔 안 깎고,
    #    하락장(Pbear↑)에만 방어. 하방변동성만 써서 상승 변동성엔 안 깎임(멜트업 보호).
    cur_vol = np.maximum(dvol, VOL_FLOOR)
    # 목표변동성 = 고정상수가 아니라 '백테스트 시작~당월'의 하방변동성 확장평균(대표님 지시 2026-07-30).
    # 당월까지의 값만 쓰므로 룩어헤드 없음. 판단 이전(t<start) 구간은 쓰이지 않으나 상수로 채워 둔다.
    tgt_vol = np.full(n, TARGET_VOL, dtype=float)
    tgt_vol[start:] = np.cumsum(dvol[start:]) / np.arange(1, n - start + 1)
    cut = 1.0 - np.minimum(1.0, tgt_vol / cur_vol)      # 변동성 축소분(=목표 대비 초과분)
    vol_factor = 1.0 - pbear * cut                      # Pbear 비례 적용(소프트 비대칭)
    raw_exp = np.clip((1 - pbear) * vol_factor, EXP_FLOOR, 1.0)
    # ── Z단계 리밸밴드 (§6): 변화가 밴드 넘을 때만 0.05 스텝으로 조정 ──
    exposure = raw_exp.copy(); held = None
    for t in idx:
        if held is None or abs(raw_exp[t] - held) >= REBAL_BAND:
            held = round(raw_exp[t] / 0.05) * 0.05
        exposure[t] = min(max(held, EXP_FLOOR), 1.0)
    # ── 이산 레짐(히스테리시스, 리포트/이벤트용) ──
    reg = ["Bull"] * n; p = "Bull"
    for t in idx:
        p = ("Bear" if pbear[t] >= T_OUT else "Bull") if p == "Bear" else ("Bear" if pbear[t] >= T_IN else "Bull")
        reg[t] = p
    exp_bin = np.array([EXP_FLOOR if reg[t] == "Bear" else 1.0 for t in range(n)])

    # ── 자체 백테스트: KOSPI에 익스포저 적용 (cash=0수익) ──
    r_bh = np.array([ret[t] if (t in idx and not np.isnan(ret[t])) else np.nan for t in range(n)])
    r_ov = np.array([exposure[t] * ret[t] if (t in idx and not np.isnan(ret[t])) else np.nan for t in range(n)])
    r_bn = np.array([exp_bin[t] * ret[t] if (t in idx and not np.isnan(ret[t])) else np.nan for t in range(n)])

    print(f"\n{'='*74}\n  자체 리포트: KOSPI 대비 HSMM 레짐 오버레이 ({yms[start]}~{yms[-2]}, {len(idx)-1}개월)\n{'='*74}")
    print(f"  {'전략':16} {'CAGR':>7} {'Sharpe':>7} {'MDD':>8} {'Calmar':>7} {'Vol':>7}")
    print(line("KOSPI 매수보유", perf(r_bh)))
    print(line("연속(vol타겟)", perf(r_ov)))
    print(line("이산(Bull/Bear)", perf(r_bn)))
    print(f"\n  * 연속 = 소프트 비대칭 vol-타겟(목표=BT시작~당월 하방변동성 확장평균 {tgt_vol[start]:.1%}→{tgt_vol[-1]:.1%}"
          f"/하방변동성, 축소를 P_bear 비례 적용), 하한 {EXP_FLOOR:.0%}, 리밸밴드 {REBAL_BAND}")
    print(f"    이산 = P_bear 히스테리시스({T_IN}/{T_OUT}); Bear시 익스포저 {EXP_FLOOR:.0%}")

    labs = [yms[t] for t in range(n)]
    for nm, rr in [("KOSPI", r_bh), ("연속", r_ov), ("이산", r_bn)]:
        mw = mdd_window(rr, labs)
        if mw: print(f"  MDD구간[{nm}] {mw[0]} → {mw[1]}  ({mw[2]:.1f}%)")

    pd.DataFrame({"ym": [yms[t] for t in idx], "ret": [ret[t] for t in idx],
        "pbear_raw": [pbear_raw[t] for t in idx], "pbear": [pbear[t] for t in idx],
        "rvol": [rvol[t] for t in idx], "dvol": [dvol[t] for t in idx],
        "tgt_vol": [tgt_vol[t] for t in idx],
        "exposure": [exposure[t] for t in idx], "exp_bin": [exp_bin[t] for t in idx],
        "regime": [reg[t] for t in idx]}).to_csv(A / "hsmm_final_path.csv", index=False, encoding="utf-8-sig")

    bearm = sum(1 for t in idx if reg[t] == "Bear")
    print(f"\n  평균 익스포저 {np.nanmean(exposure[start:]):.2f} (연속) / Bear월수 {bearm}/{len(idx)} (이산)")

    # ── 위기 detection (이산 레짐) ──
    evs = [i for i in idx if dd6[i] <= -15 and (i == start or not (dd6[i - 1] <= -15))]
    onset = [t for t in idx if reg[t] == "Bear" and reg[t - 1] != "Bear"]
    NAMES = {"2018-10": "2018 하락", "2020-02": "2020 코로나", "2021-11": "2021~22 하락",
             "2022-01": "2022 하락", "2024-08": "2024 급락", "2025-09": "2025 하락"}
    print(f"\n=== 위기 detection ({yms[start]}~; 값=선행개월 +빠름/-늦음, X=놓침) ===")
    for ev in evs:
        cand = [s for s in onset if abs(s - ev) <= WIN]
        tag = f"{ev - min(cand, key=lambda s: abs(s-ev)):+d}개월" if cand else "X(놓침)"
        print(f"  {NAMES.get(yms[ev], yms[ev]):14} 낙폭 {dd6[ev]:>5.0f}%   →  {tag}")

    # ── 최근 익스포저 경로 ──
    print(f"\n=== 최근 12개월 익스포저 경로 ===")
    print(f"  {'월':9}{'P_bear':>8}{'현재변동성':>10}{'연속익스포저':>12}{'이산':>7}")
    for t in idx[-12:]:
        print(f"  {yms[t]:9}{pbear[t]:>8.2f}{cur_vol[t]*100:>9.0f}%{exposure[t]:>12.2f}{reg[t]:>7}")

    _md = f"HSMM(명시적 지속기간 NB, Dmax{DMAX})" if USE_DURATION else "plain 2-state HMM(박사님 스펙, 지속기간 제약 없음)"
    print(f"\n모델: {_md} — filtered 연속확률 + 스트레스 변조 전이.")
    print(f"  Emission {EMIS_COLS} / Transition(스트레스) {TRAN_COLS}.")
    print(f"과적합방지: 2-state·LedoitWolf·시간감쇠(HL={HL:.0f}월)·5년롤링창·웜스타트.")
    print(f"창 {yms[0]}~: 외국인 순매수 DB 2000~ 있으나 breadth/신저가용 종목시세(daily_price) 2017~ 병목 → 판단 2018부터.")


if __name__ == "__main__":
    main()
