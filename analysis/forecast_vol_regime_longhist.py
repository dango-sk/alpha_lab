"""
analysis/forecast_vol_regime_longhist.py

GARCH/HAR 변동성 예측을 *장기* KOSPI 지수(^KS11, 1996-12~, 330개월·Bear 8사이클)에
재평가. 기존 forecast_vol_regime.py는 069500 ETF·98개월(2017~)이라 Bear 에피소드가
~3개뿐이었음 → HMM과 동일하게 장기 데이터로 공정 비교.

검증된 forecast 함수(har/garch/drift)는 forecast_vol_regime.py에서 재사용.
데이터 소스만 ^KS11(macro_indicators)로 교체하고 월 리스트를 가격에서 직접 생성.

입력:  Railway PG (alpha_lab.macro_indicators indicator='kospi' freq='D')
       analysis/regime_agent_multimodel_results_gemini.json (AI v2 비교용)
출력:  analysis/vol_regime_longhist_features.csv

사용:  .venv/bin/python analysis/forecast_vol_regime_longhist.py
       (GARCH MLE를 330회 expanding 적합 → 수 분 소요)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# 검증된 forecast 함수 재사용 (decimal 로그수익률 입력 가정)
from forecast_vol_regime import (
    har_forecast_monthly_vol,
    garch_forecast_monthly_vol,
    trailing_drift,
)

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

RESULTS_JSON = Path(__file__).parent / "regime_agent_multimodel_results_gemini.json"
OUT_CSV = Path(__file__).parent / "vol_regime_longhist_features.csv"
BEAR_THRESHOLD = -3.0
MIN_TRAIN_DAYS = 252      # 첫 예측 전 최소 일수(~1년) — forecaster 적합 가능하게
LR_MIN_TRAIN = 60         # walk-forward LR 최소 학습 월수


def load_kospi_index(conn):
    """^KS11 일별 종가 → decimal 로그수익률 (forecast 함수가 decimal 가정)."""
    df = pd.read_sql(
        "SELECT period::date AS dt, value::float AS close "
        "FROM alpha_lab.macro_indicators WHERE indicator='kospi' AND freq='D' "
        "ORDER BY period",
        conn,
    )
    if df.empty:
        raise SystemExit("❌ kospi 데이터 없음. 먼저 scripts/backfill_global_indices.py 실행")
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.dropna().drop_duplicates(subset="dt").set_index("dt").sort_index()
    df["ret"] = np.log(df["close"]).diff()   # decimal
    return df.dropna()


def month_end_index(df):
    ym = pd.PeriodIndex(df.index, freq="M")
    last = {}
    for dt, p in zip(df.index, ym):
        last[p] = dt
    return [last[p] for p in sorted(last)]


def load_ai_v2_bears():
    if not RESULTS_JSON.exists():
        return {}
    with open(RESULTS_JSON) as f:
        recs = json.load(f)
    return {r["as_of"][:7]: int(r.get("judgment") == "약세") for r in recs}


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    print("Loading ^KS11 daily from macro_indicators...")
    df = load_kospi_index(conn)
    conn.close()
    print(f"  {len(df)} daily returns, {df.index.min().date()} → {df.index.max().date()}")

    rets = df["ret"].values
    dates = df.index
    close = df["close"]
    m_ends = month_end_index(df)

    rows = []
    for i in range(len(m_ends) - 1):
        e = m_ends[i]            # cutoff (이 시점까지만 사용)
        nxt = m_ends[i + 1]      # 예측 대상 월의 월말
        mask = dates <= e
        if mask.sum() < MIN_TRAIN_DAYS:
            continue
        hist = rets[mask]        # decimal 로그수익률

        har = har_forecast_monthly_vol(hist)
        garch = garch_forecast_monthly_vol(hist)
        drift = trailing_drift(hist)

        # 예측 대상 월 실현변동성(평가용)
        nxt_mask = (dates > e) & (dates <= nxt)
        realized = df["ret"][nxt_mask].std() * np.sqrt(max(nxt_mask.sum(), 1)) * 100 if nxt_mask.sum() >= 5 else np.nan

        month_ret = (close.loc[nxt] / close.loc[e] - 1) * 100
        pred_ym = pd.Period(nxt, freq="M").strftime("%Y-%m")
        rows.append({
            "pred_month": pred_ym,
            "cutoff": e.isoformat(),
            "har_vol_fcst": har,
            "garch_vol_fcst": garch,
            "trailing_drift": drift,
            "realized_next_vol": realized,
            "month_ret": month_ret,
            "real_bear": int(month_ret < BEAR_THRESHOLD),
        })

    out = pd.DataFrame(rows).dropna(subset=["har_vol_fcst", "garch_vol_fcst", "trailing_drift"])
    out.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*70}\n  Saved {len(out)} rows → {OUT_CSV}\n{'='*70}")
    print(f"  real_bear: {out['real_bear'].sum()} / {len(out)}개월  "
          f"({out['pred_month'].iloc[0]} ~ {out['pred_month'].iloc[-1]})")

    # [1] forecaster 품질
    print("\n[1] Forecaster 품질 (σ̂ vs 실제 실현변동성, Pearson r)")
    val = out.dropna(subset=["realized_next_vol"])
    for col in ["har_vol_fcst", "garch_vol_fcst"]:
        r = np.corrcoef(val[col], val["realized_next_vol"])[0, 1]
        print(f"    {col:<18} r = {r:+.3f}  (n={len(val)})")

    # [2] real_bear 분리 AUC (단변량, 학습불필요)
    print("\n[2] real_bear 분리 AUC")
    for col in ["har_vol_fcst", "garch_vol_fcst"]:
        print(f"    {col:<18} AUC = {roc_auc_score(out['real_bear'], out[col]):.3f}")
    print(f"    {'(-)trailing_drift':<18} AUC = {roc_auc_score(out['real_bear'], -out['trailing_drift']):.3f}")

    # [3] walk-forward LR (har+garch+drift)
    print("\n[3] Walk-forward LR (har_vol + garch_vol + drift)")
    feats = ["har_vol_fcst", "garch_vol_fcst", "trailing_drift"]
    X = out[feats].values
    y = out["real_bear"].values
    probs = np.full(len(out), np.nan)
    for i in range(LR_MIN_TRAIN, len(out)):
        ytr = y[:i]
        if ytr.sum() < 5 or (len(ytr) - ytr.sum()) < 5:
            continue
        sc = StandardScaler().fit(X[:i])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        clf.fit(sc.transform(X[:i]), ytr)
        probs[i] = clf.predict_proba(sc.transform(X[i:i + 1]))[0, 1]
    out["lr_bear_prob"] = probs
    ev = out.dropna(subset=["lr_bear_prob"])
    pred = (ev["lr_bear_prob"] >= 0.5).astype(int)
    tp = int(((pred == 1) & (ev["real_bear"] == 1)).sum())
    fp = int(((pred == 1) & (ev["real_bear"] == 0)).sum())
    fn = int(((pred == 0) & (ev["real_bear"] == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    wf_auc = roc_auc_score(ev["real_bear"], ev["lr_bear_prob"]) if ev["real_bear"].nunique() > 1 else float("nan")
    print(f"    평가구간 {len(ev)}개월 | Recall={recall:.1%} (TP={tp},FN={fn})  "
          f"Precision={prec:.1%} (FP={fp})  OOS AUC={wf_auc:.3f}")

    # AI v2 겹침구간 비교
    ai = load_ai_v2_bears()
    ov = ev[ev["pred_month"].isin(ai.keys())].copy()
    if len(ov):
        ov["ai_bear"] = ov["pred_month"].map(ai)
        a_tp = int(((ov["ai_bear"] == 1) & (ov["real_bear"] == 1)).sum())
        a_fn = int(((ov["ai_bear"] == 0) & (ov["real_bear"] == 1)).sum())
        a_fp = int(((ov["ai_bear"] == 1) & (ov["real_bear"] == 0)).sum())
        a_rec = a_tp / max(a_tp + a_fn, 1)
        a_prec = a_tp / max(a_tp + a_fp, 1)
        print(f"\n    [AI v2 겹침 {len(ov)}개월] AI v2 Recall={a_rec:.1%} Precision={a_prec:.1%}")
    print(f"{'='*70}")
    print("판정: [2] AUC ≥ 0.65 또는 [3] OOS AUC ≥ 0.65 → 신호 有.")


if __name__ == "__main__":
    main()
