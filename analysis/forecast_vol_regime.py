"""
analysis/forecast_vol_regime.py

시계열 변동성 예측(GARCH/HAR-RV)으로 레짐 신호를 만들고, 기존 AI v2 대비
Bear 포착력을 평가하는 *분석 스크립트* (운영 코드 미수정).

아이디어
--------
월별 레짐 라벨은 98개월뿐이라 직접 시계열 예측이 어렵다. 대신 *일별* KOSPI
로그수익률(수천 포인트)에서 예측 가능한 양 = **다음 달 변동성**을 forecast하고,
음의 drift와 결합해 Bear 확률로 매핑한다. Bear ≈ 고변동성 + 음의 drift.

  일별 r_t  ──┬─ HAR-RV (OLS)      ─┐
              └─ GARCH(1,1) (MLE)  ─┴─ 다음달 σ̂  ──┐
              └─ trailing drift               ─────┴─ (walk-forward LR) → Bear prob

모든 추정은 expanding window + 엄격한 cutoff(as_of - 1d)로 lookahead-free.
하드코딩 threshold 없음: σ̂→Bear 매핑은 walk-forward 로지스틱으로 *학습*.

평가
----
1. 예측기 자체 품질: σ̂_forecast vs 실제 다음달 실현변동성 상관 (forecaster가
   애초에 작동하는지).
2. 레짐 신호력: σ̂(+drift)가 real_bear(다음달 ret<-3%)를 분리하는 AUC
   (threshold-free, 학습 불필요).
3. walk-forward LR로 실제 Bear Recall/Precision 산출 → AI v2(70%) / 원본(26.9%)과 비교.

입력:  Railway PG (alpha_lab.daily_price 069500),
       analysis/regime_agent_multimodel_results_gemini.json (라벨 + AI v2 judgment)
출력:  analysis/vol_regime_features.csv

사용:  .venv/bin/python analysis/forecast_vol_regime.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import psycopg2
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from dotenv import load_dotenv
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

TRADING_DAYS_PER_MONTH = 22
RESULTS_JSON = Path(__file__).parent / "regime_agent_multimodel_results_gemini.json"
OUT_CSV = Path(__file__).parent / "vol_regime_features.csv"


# ──────────────────────────── 데이터 ────────────────────────────
def load_kospi_returns(conn):
    """KOSPI ETF 069500 일별 종가 → 로그수익률 시계열 (index=date)."""
    df = pd.read_sql(
        "SELECT trade_date::date AS dt, close::float AS close "
        "FROM alpha_lab.daily_price WHERE stock_code='069500' ORDER BY trade_date",
        conn,
    )
    df = df.dropna().drop_duplicates(subset="dt").set_index("dt")
    df["ret"] = np.log(df["close"]).diff()
    return df.dropna()


# ──────────────────────────── HAR-RV ────────────────────────────
def har_forecast_monthly_vol(rets):
    """expanding-window HAR-RV로 다음 22거래일 평균 일별분산 예측 → 월별 vol(%).

    rets: cutoff까지의 일별 로그수익률 (np.array, decimal).
    일별 RV proxy = r^2 (intraday 없음). HAR: target ~ RV1 + RV5 + RV22.
    """
    n = len(rets)
    if n < 80:
        return np.nan
    rv = rets ** 2
    rv_s = pd.Series(rv)
    RV1 = rv_s.values
    RV5 = rv_s.rolling(5).mean().values
    RV22 = rv_s.rolling(22).mean().values
    # fwd[t] = mean(rv[t+1 .. t+22]) — backward rolling을 shift(-22)로 forward화
    fwd = rv_s.rolling(TRADING_DAYS_PER_MONTH).mean().shift(-TRADING_DAYS_PER_MONTH)

    # 학습 표본: feature와 target이 모두 정의된 구간 (마지막 22일은 target 미정)
    idx = np.arange(22, n - TRADING_DAYS_PER_MONTH)
    if len(idx) < 40:
        return np.nan
    X = np.column_stack([np.ones(len(idx)), RV1[idx], RV5[idx], RV22[idx]])
    y = fwd.values[idx]
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < 40:
        return np.nan
    coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)

    # 예측: cutoff 시점(마지막 관측)의 feature로 다음달 평균 RV forecast
    x_last = np.array([1.0, RV1[-1], RV5[-1], RV22[-1]])
    if not np.all(np.isfinite(x_last)):
        return np.nan
    avg_var = max(coef @ x_last, 1e-8)
    monthly_vol = np.sqrt(TRADING_DAYS_PER_MONTH * avg_var) * 100
    return monthly_vol


# ──────────────────────────── GARCH(1,1) ────────────────────────────
def _garch_negloglik(params, r):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(r)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma2 = np.clip(sigma2, 1e-12, None)
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + r ** 2 / sigma2)
    return -ll


def garch_forecast_monthly_vol(rets):
    """expanding-window GARCH(1,1) MLE → 다음 22거래일 누적변동성(%)."""
    n = len(rets)
    if n < 80:
        return np.nan
    r = (rets - rets.mean()) * 100.0  # 스케일업 (optimizer 안정)
    var0 = np.var(r)
    x0 = np.array([var0 * 0.05, 0.08, 0.90])
    bnds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999)]
    try:
        res = minimize(_garch_negloglik, x0, args=(r,), method="L-BFGS-B",
                       bounds=bnds, options={"maxiter": 200})
        omega, alpha, beta = res.x
    except Exception:
        return np.nan
    # 마지막 조건부 분산
    sigma2 = var0
    for t in range(1, n):
        sigma2 = omega + alpha * r[t - 1] ** 2 + beta * sigma2
    # h-step forecast: sigma2_{t+h} → 무조건분산으로 수렴
    uncond = omega / max(1 - alpha - beta, 1e-6)
    fc_var = 0.0
    s = omega + alpha * r[-1] ** 2 + beta * sigma2  # sigma2_{t+1}
    for h in range(1, TRADING_DAYS_PER_MONTH + 1):
        if h > 1:
            s = uncond + (alpha + beta) * (s - uncond)  # E[sigma2_{t+h}] = omega + (a+b)*prev
        fc_var += s
    # r은 *100 스케일이므로 vol(%)은 sqrt(fc_var) (이미 % 단위)
    return np.sqrt(max(fc_var, 1e-8))


def realized_next_month_vol(rets_full, as_of):
    """평가용: as_of 이후 ~ 다음달의 실제 실현변동성(%). (라벨 검증용, feature 아님)"""
    start = as_of
    end = as_of + relativedelta(months=1)
    sub = rets_full[(rets_full.index >= start) & (rets_full.index < end)]
    if len(sub) < 5:
        return np.nan
    return sub["ret"].std() * np.sqrt(len(sub)) * 100


def trailing_drift(rets, days=63):
    """직전 3개월(63거래일) 평균 일별수익률 → 월 환산(%)."""
    if len(rets) < days:
        return np.nan
    return rets[-days:].mean() * TRADING_DAYS_PER_MONTH * 100


# ──────────────────────────── 메인 ────────────────────────────
def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    print("Loading KOSPI 069500 daily returns from Railway PG...")
    px = load_kospi_returns(conn)
    conn.close()
    print(f"  {len(px)} daily returns, {px.index.min()} → {px.index.max()}")

    with open(RESULTS_JSON) as f:
        records = json.load(f)
    records.sort(key=lambda r: r["as_of"])

    rows = []
    for r in records:
        as_of = date.fromisoformat(r["as_of"][:10])
        cutoff = as_of - timedelta(days=1)
        hist = px[px.index <= cutoff]["ret"].values
        if len(hist) < 80:
            continue

        har = har_forecast_monthly_vol(hist)
        garch = garch_forecast_monthly_vol(hist)
        drift = trailing_drift(hist)
        realized = realized_next_month_vol(px, as_of)

        actual = r.get("kospi_next_month_return")
        if actual is None:
            continue

        rows.append({
            "as_of": r["as_of"][:10],
            "har_vol_fcst": har,
            "garch_vol_fcst": garch,
            "trailing_drift": drift,
            "realized_next_vol": realized,           # 평가용
            "kospi_next_ret": actual,                # 라벨용
            "real_bear": int(actual < -3),
            "ai_says_bear": int(r.get("judgment") == "약세"),
        })

    df = pd.DataFrame(rows).dropna(subset=["har_vol_fcst", "garch_vol_fcst", "trailing_drift"])
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*70}\n  Saved {len(df)} rows → {OUT_CSV}\n{'='*70}")

    # ── 1. forecaster 품질: σ̂ vs 실제 실현변동성 ──
    print("\n[1] Forecaster 품질 (σ̂ vs 실제 다음달 실현변동성, Pearson r)")
    val = df.dropna(subset=["realized_next_vol"])
    for col in ["har_vol_fcst", "garch_vol_fcst"]:
        r_corr = np.corrcoef(val[col], val["realized_next_vol"])[0, 1]
        print(f"    {col:<18} r = {r_corr:+.3f}   (n={len(val)})")

    # ── 2. 레짐 신호력: real_bear 분리 AUC (threshold-free) ──
    print("\n[2] real_bear 분리 AUC (학습 불필요, 단변량)")
    print(f"    real_bear: {df['real_bear'].sum()} / {len(df)}개월")
    for col in ["har_vol_fcst", "garch_vol_fcst"]:
        auc = roc_auc_score(df["real_bear"], df[col])
        print(f"    {col:<18} AUC = {auc:.3f}")
    # drift는 낮을수록 bear이므로 부호 반전해서 AUC
    auc_drift = roc_auc_score(df["real_bear"], -df["trailing_drift"])
    print(f"    {'(-)trailing_drift':<18} AUC = {auc_drift:.3f}")

    # ── 3. walk-forward LR (σ̂ + drift) → Bear Recall/Precision ──
    print("\n[3] Walk-forward LR (har_vol + garch_vol + drift) → 실제 Bear 판정")
    feats = ["har_vol_fcst", "garch_vol_fcst", "trailing_drift"]
    X = df[feats].values
    y = df["real_bear"].values
    preds = np.full(len(df), np.nan)
    probs = np.full(len(df), np.nan)
    MIN_TRAIN = 36
    for i in range(MIN_TRAIN, len(df)):
        Xtr, ytr = X[:i], y[:i]
        if ytr.sum() < 3 or (len(ytr) - ytr.sum()) < 3:
            continue
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        clf.fit(sc.transform(Xtr), ytr)
        p = clf.predict_proba(sc.transform(X[i:i + 1]))[0, 1]
        probs[i] = p
        preds[i] = int(p >= 0.5)
    df["lr_bear_prob"] = probs
    df["lr_pred_bear"] = preds

    ev = df.dropna(subset=["lr_pred_bear"])
    tp = int(((ev["lr_pred_bear"] == 1) & (ev["real_bear"] == 1)).sum())
    fp = int(((ev["lr_pred_bear"] == 1) & (ev["real_bear"] == 0)).sum())
    fn = int(((ev["lr_pred_bear"] == 0) & (ev["real_bear"] == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    wf_auc = (roc_auc_score(ev["real_bear"], ev["lr_bear_prob"])
              if ev["real_bear"].nunique() > 1 else float("nan"))
    print(f"    평가구간: {len(ev)}개월 (walk-forward, MIN_TRAIN={MIN_TRAIN})")
    print(f"    Bear Recall    = {recall:.1%}  (TP={tp}, FN={fn})")
    print(f"    Bear Precision = {prec:.1%}  (FP={fp})")
    print(f"    OOS AUC        = {wf_auc:.3f}")

    # 같은 구간에서 AI v2 비교
    ai = ev
    ai_tp = int(((ai["ai_says_bear"] == 1) & (ai["real_bear"] == 1)).sum())
    ai_fn = int(((ai["ai_says_bear"] == 0) & (ai["real_bear"] == 1)).sum())
    ai_fp = int(((ai["ai_says_bear"] == 1) & (ai["real_bear"] == 0)).sum())
    ai_recall = ai_tp / (ai_tp + ai_fn) if (ai_tp + ai_fn) else 0.0
    ai_prec = ai_tp / (ai_tp + ai_fp) if (ai_tp + ai_fp) else 0.0
    print(f"\n    [동일 구간 AI v2] Recall = {ai_recall:.1%}  Precision = {ai_prec:.1%}")
    print(f"{'='*70}")
    print("판정 기준: [2] AUC ≥ 0.65 또는 [3] OOS AUC ≥ 0.65 면 신호 有 → AI v2와 결합/대체 검토.")


if __name__ == "__main__":
    main()
