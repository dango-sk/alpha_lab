"""
analysis/hmm_regime.py

장기 KOSPI 지수(1996-12~)에 Gaussian HMM을 walk-forward로 적용해 Bear 레짐을
탐지하고, 기존 AI v2 대비 Bear 포착력을 평가하는 *분석 스크립트* (운영 미수정).

왜 HMM인가
----------
앞선 시도(ML 필터 AUC 0.467, 변동성 예측 AUC 0.56)가 막힌 이유 두 가지:
  A. 표본/Bear 에피소드 부족 (8년에 ~3개)  → KOSPI 지수 1996~로 확장(~8개 에피소드)
  B. 변동성=크기지 방향 아님              → HMM은 일별 *수익률* 자체를 모델링,
                                            각 state가 평균(방향)+분산(크기)을 동시 학습

HMM 핵심
--------
숨은 state 2~3개를 가정. 학습하는 것:
  1) 각 state의 수익률 평균·분산  (평온: 평균≈+/저변동, 위기: 평균≈−/고변동)
  2) state 간 전이확률           (state가 잘 안 바뀜 → whipsaw↓, hysteresis 내장)
  3) 관측으로 본 현재 state 확률
Bear state = 학습된 평균수익률이 가장 낮은 state. P(현재=Bear state)를 신호로 사용.

lookahead 방지
--------------
매 월말 e에서 *그 시점까지의* 일별 수익률로만 HMM을 새로 적합(expanding window).
e의 P(Bear)로 *다음 달* 수익률(label)을 예측. label은 적합에 절대 안 들어감.

입력:  Railway PG (alpha_lab.macro_indicators indicator='kospi' freq='D')
       └ 먼저 scripts/backfill_global_indices.py 실행해 ^KS11 적재 필요
       analysis/regime_agent_multimodel_results_gemini.json (AI v2 비교용, 2018-04~)
출력:  analysis/hmm_regime_features.csv

사용:  .venv/bin/python analysis/hmm_regime.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

RESULTS_JSON = Path(__file__).parent / "regime_agent_multimodel_results_gemini.json"
OUT_CSV = Path(__file__).parent / "hmm_regime_features.csv"

N_STATES_LIST = [2, 3]      # 비교: 2-state vs 3-state
MIN_TRAIN_DAYS = 500        # 첫 적합 전 최소 일수(~2년) — bear state 추정 가능하게
BEAR_THRESHOLD = -3.0       # real_bear: 다음달 수익률 < -3% (앞 실험과 동일)
SEED = 42


def load_kospi_index(conn):
    """KOSPI 종합지수 일별 종가 → 로그수익률(%). macro_indicators에서 로드."""
    df = pd.read_sql(
        "SELECT period::date AS dt, value::float AS close "
        "FROM alpha_lab.macro_indicators WHERE indicator='kospi' AND freq='D' "
        "ORDER BY period",
        conn,
    )
    if df.empty:
        raise SystemExit(
            "❌ macro_indicators에 indicator='kospi' 데이터가 없습니다.\n"
            "   먼저 실행: .venv/bin/python scripts/backfill_global_indices.py"
        )
    df["dt"] = pd.to_datetime(df["dt"])   # date 객체 → DatetimeIndex
    df = df.dropna().drop_duplicates(subset="dt").set_index("dt").sort_index()
    df["ret"] = np.log(df["close"]).diff() * 100.0   # % 스케일 (HMM 수치안정)
    return df.dropna()


def month_end_index(df):
    """각 캘린더 월의 마지막 거래일 날짜 리스트 (오름차순)."""
    ym = pd.PeriodIndex(df.index, freq="M")
    last = {}
    for dt, p in zip(df.index, ym):
        last[p] = dt   # 같은 월이면 뒤 날짜로 덮어써져 결국 월말 거래일
    return [last[p] for p in sorted(last)]


def fit_hmm_pbear(X, n_states):
    """X(=일별수익률, shape (n,1))로 HMM 적합 → (현재 P(Bear), state 요약)."""
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag",
        n_iter=30, tol=1e-3, random_state=SEED,
    )
    model.fit(X)
    means = model.means_[:, 0]
    bear_state = int(np.argmin(means))   # 평균수익률 최저 = Bear
    post = model.predict_proba(X)
    p_bear_now = float(post[-1, bear_state])
    summary = {
        "means": np.round(means, 3).tolist(),
        "vols": np.round(np.sqrt(model.covars_[:, 0, 0]), 3).tolist(),
        "bear_state": bear_state,
    }
    return p_bear_now, summary


def run_walkforward(df, n_states):
    """월말마다 expanding-window HMM 적합 → 각 월말 P(Bear). 다음달을 예측."""
    rets = df["ret"].values
    dates = df.index
    close = df["close"]
    m_ends = month_end_index(df)

    rows = []
    last_summary = None
    for i in range(len(m_ends) - 1):
        e = m_ends[i]            # 적합 cutoff (이 시점까지만 사용)
        nxt = m_ends[i + 1]      # 예측 대상 월의 월말
        # cutoff까지 일별 수익률
        mask = dates <= e
        if mask.sum() < MIN_TRAIN_DAYS:
            continue
        X = rets[mask].reshape(-1, 1)
        try:
            p_bear, summary = fit_hmm_pbear(X, n_states)
        except Exception:
            continue
        last_summary = summary
        # 예측 대상 월 수익률(label): close[nxt]/close[e]-1
        month_ret = (close.loc[nxt] / close.loc[e] - 1) * 100
        pred_ym = pd.Period(nxt, freq="M").strftime("%Y-%m")
        rows.append({
            "pred_month": pred_ym,
            "cutoff": e.isoformat(),
            "p_bear": p_bear,
            "month_ret": month_ret,
            "real_bear": int(month_ret < BEAR_THRESHOLD),
        })
    return pd.DataFrame(rows), last_summary


def load_ai_v2_bears():
    """AI v2 judgment='약세' 월 집합 (year-month) — 동일구간 비교용."""
    if not RESULTS_JSON.exists():
        return {}
    with open(RESULTS_JSON) as f:
        recs = json.load(f)
    out = {}
    for r in recs:
        ym = r["as_of"][:7]
        out[ym] = int(r.get("judgment") == "약세")
    return out


def evaluate(df, ai_bears, n_states):
    print(f"\n{'='*70}\n  HMM {n_states}-state\n{'='*70}")
    print(f"  평가 월수: {len(df)}  (월말 walk-forward, MIN_TRAIN={MIN_TRAIN_DAYS}일)")
    print(f"  real_bear: {df['real_bear'].sum()} / {len(df)}개월  "
          f"(첫 예측월 {df['pred_month'].iloc[0]} ~ {df['pred_month'].iloc[-1]})")

    # [2] threshold-free: P(Bear) AUC
    if df["real_bear"].nunique() > 1:
        auc = roc_auc_score(df["real_bear"], df["p_bear"])
        print(f"  [2] P(Bear) AUC (학습불필요)       = {auc:.3f}")

    # [3] P(Bear) >= 0.5 → Bear 판정 (확률 0.5는 magic market threshold 아님)
    pred = (df["p_bear"] >= 0.5).astype(int)
    tp = int(((pred == 1) & (df["real_bear"] == 1)).sum())
    fp = int(((pred == 1) & (df["real_bear"] == 0)).sum())
    fn = int(((pred == 0) & (df["real_bear"] == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    print(f"  [3] Bear Recall (P>=0.5)           = {recall:.1%}  (TP={tp}, FN={fn})")
    print(f"      Bear Precision                 = {prec:.1%}  (FP={fp})")

    # AI v2 동일구간 비교 (2018-04~)
    ov = df[df["pred_month"].isin(ai_bears.keys())].copy()
    if len(ov):
        ov["ai_bear"] = ov["pred_month"].map(ai_bears)
        h_pred = (ov["p_bear"] >= 0.5).astype(int)
        h_rec = ((h_pred == 1) & (ov["real_bear"] == 1)).sum() / max((ov["real_bear"] == 1).sum(), 1)
        a_rec = ((ov["ai_bear"] == 1) & (ov["real_bear"] == 1)).sum() / max((ov["real_bear"] == 1).sum(), 1)
        a_tp = int(((ov["ai_bear"] == 1) & (ov["real_bear"] == 1)).sum())
        a_fp = int(((ov["ai_bear"] == 1) & (ov["real_bear"] == 0)).sum())
        a_prec = a_tp / max(a_tp + a_fp, 1)
        print(f"\n  [AI v2 겹침구간 {len(ov)}개월] HMM Recall={h_rec:.1%} | "
              f"AI v2 Recall={a_rec:.1%} Precision={a_prec:.1%}")
    return pred


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    print("Loading KOSPI 지수(^KS11) from macro_indicators...")
    df = load_kospi_index(conn)
    conn.close()
    print(f"  {len(df)} daily returns, {df.index.min().date()} → {df.index.max().date()}")

    ai_bears = load_ai_v2_bears()
    base = None        # 첫 n의 전체 결과(month_ret/real_bear 등 메타 포함)
    pbear_cols = []    # 각 n의 p_bear 컬럼
    for n in N_STATES_LIST:
        res, summary = run_walkforward(df, n)
        if summary:
            print(f"\n[{n}-state] 마지막 적합 state 성격: "
                  f"평균={summary['means']} 변동성={summary['vols']} "
                  f"→ Bear=state#{summary['bear_state']}")
        evaluate(res, ai_bears, n)
        res = res.set_index("pred_month")
        if base is None:
            base = res.drop(columns=["p_bear"])
        pbear_cols.append(res[["p_bear"]].rename(columns={"p_bear": f"p_bear_{n}s"}))

    # CSV 저장: 메타 + 각 n-state의 p_bear (재적합 없이 루프 결과 재사용)
    for col in pbear_cols:
        base = base.join(col, how="left")
    base.to_csv(OUT_CSV)
    print(f"\n{'='*70}\n  Saved → {OUT_CSV}")
    print("판정: [2] AUC ≥ 0.65 또는 [3] Recall이 AI v2(겹침구간) 상회 → 신호 有 → 결합/대체 검토.")


if __name__ == "__main__":
    main()
