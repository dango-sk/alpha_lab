"""
analysis/step2_forecast_dd.py

Step 2: 지도학습(RF/GBM)으로 *미래 큰 낙폭*을 예측.
KOSPI/SP500/SOX(1996~)의 모멘텀·변동성·낙폭 feature → 다음 구간 낙폭 라벨 예측.

라벨 3종 (forward-looking, KOSPI 일별에서 생성):
  bear_1m_ret_3 : close[e+1m]/close[e]-1 < -3%        (기존 1개월 라벨, 비교용)
  bear_3m_dd_10 : [e, e+3m] 고점대비 최대낙폭 <= -10%  (trailing-stop 정신)
  bear_6m_dd_15 : [e, e+6m] 고점대비 최대낙폭 <= -15%  (정확히 -15% stop 트리거)

핵심 주의 — 라벨 겹침:
  Nm 라벨은 인접 달끼리 미래구간을 공유 → 자기상관 → AUC 부풀려짐.
  walk-forward에서 test월 직전 H개월을 학습에서 제외(purge/embargo)해 누출 차단.
  독립 표본 ≈ 개월수 / H 임을 함께 리포트.

feature는 전부 cutoff(e=월말) 이전 데이터만 사용 (lookahead-free).

입력:  Railway PG (kospi/sp500/sox, freq='D')
       analysis/hmm_regime_features.csv (Step1 HMM p_bear, 있으면 비교)
       analysis/regime_agent_multimodel_results_gemini.json (AI v2 비교)
출력:  analysis/step2_dd_features.csv

사용:  .venv/bin/python analysis/step2_forecast_dd.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

HMM_CSV = Path(__file__).parent / "hmm_regime_features.csv"
AI_JSON = Path(__file__).parent / "regime_agent_multimodel_results_gemini.json"
OUT_CSV = Path(__file__).parent / "step2_dd_features.csv"

MIN_TRAIN = 60   # 첫 예측 전 최소 학습 월수
# 라벨: (이름, kind, horizon_months, threshold_pct)
LABELS = [
    ("bear_1m_ret_3", "ret", 1, -3.0),
    ("bear_3m_dd_10", "dd", 3, -10.0),
    ("bear_6m_dd_15", "dd", 6, -15.0),
]


def load_series(conn, indicator):
    df = pd.read_sql(
        "SELECT period::date AS dt, value::float AS close FROM alpha_lab.macro_indicators "
        "WHERE indicator=%s AND freq='D' ORDER BY period",
        conn, params=(indicator,),
    )
    df["dt"] = pd.to_datetime(df["dt"])
    return df.dropna().drop_duplicates(subset="dt").set_index("dt")["close"].sort_index()


def month_ends(idx):
    ym = pd.PeriodIndex(idx, freq="M")
    last = {}
    for dt, p in zip(idx, ym):
        last[p] = dt
    return [last[p] for p in sorted(last)]


def make_label(close, e, kind, months, thr):
    """e 시점 기준 forward 라벨. ret: 월간수익률<thr / dd: 구간 최대낙폭<=thr."""
    end = e + relativedelta(months=months)
    fut = close[(close.index > e) & (close.index <= end)]
    if len(fut) < 3:
        return np.nan
    if kind == "ret":
        return int((fut.iloc[-1] / close.loc[e] - 1) * 100 < thr)
    # dd: 진입가(e)부터의 running peak 대비 최저 낙폭
    path = pd.concat([pd.Series([close.loc[e]], index=[e]), fut])
    dd = (path / path.cummax() - 1).min() * 100
    return int(dd <= thr)


def feats(close, e):
    """cutoff e까지의 데이터로 모멘텀·변동성·낙폭·MA갭 (lookahead-free)."""
    s = close[close.index <= e]
    if len(s) < 252:
        return None
    ret = s.pct_change()
    p = s.iloc[-1]
    f = {
        "mom_1m": p / s.iloc[-22] - 1,
        "mom_3m": p / s.iloc[-64] - 1,
        "mom_6m": p / s.iloc[-127] - 1,
        "vol_1m": ret.iloc[-21:].std() * np.sqrt(21),
        "vol_3m": ret.iloc[-63:].std() * np.sqrt(63),
        "dd_cur": p / s.iloc[-252:].max() - 1,
        "ma200_gap": p / s.iloc[-200:].mean() - 1,
    }
    return {k: float(v) for k, v in f.items()}


def walkforward(df, feat_cols, label, horizon):
    """expanding + embargo(H개월 purge) walk-forward. RF/HistGBM OOS 확률."""
    X = df[feat_cols].values
    y = df[label].values
    rf_p = np.full(len(df), np.nan)
    gb_p = np.full(len(df), np.nan)
    for i in range(MIN_TRAIN, len(df)):
        cut = i - horizon          # test월 직전 H개월 제외(라벨 겹침 차단)
        if cut < MIN_TRAIN // 2:
            continue
        Xtr, ytr = X[:cut], y[:cut]
        if np.isnan(ytr).any():
            m = ~np.isnan(ytr)
            Xtr, ytr = Xtr[m], ytr[m]
        if ytr.sum() < 5 or (len(ytr) - ytr.sum()) < 5:
            continue
        rf = RandomForestClassifier(n_estimators=300, max_depth=4,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr)
        rf_p[i] = rf.predict_proba(X[i:i + 1])[0, 1]
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                            max_iter=200, random_state=42)
        gb.fit(Xtr, ytr)
        gb_p[i] = gb.predict_proba(X[i:i + 1])[0, 1]
    return rf_p, gb_p


def report(df, label, horizon, feat_cols, rf_p, gb_p, hmm_col, ai_map):
    print(f"\n{'='*72}\n  라벨: {label}  (horizon={horizon}m)")
    valid = df[df[label].notna()]
    pos = int(valid[label].sum())
    print(f"  양성 {pos}/{len(valid)} ({pos/len(valid):.0%})  |  "
          f"독립표본 ≈ {len(valid)//horizon}개 (겹침 보정)")

    for name, p in [("RandomForest", rf_p), ("HistGBM", gb_p)]:
        ev = df.assign(_p=p).dropna(subset=["_p", label])
        if ev[label].nunique() < 2:
            continue
        auc = roc_auc_score(ev[label], ev["_p"])
        pred = (ev["_p"] >= 0.5).astype(int)
        tp = int(((pred == 1) & (ev[label] == 1)).sum())
        fp = int(((pred == 1) & (ev[label] == 0)).sum())
        fn = int(((pred == 0) & (ev[label] == 1)).sum())
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        print(f"    {name:13} OOS AUC={auc:.3f}  Recall={rec:.0%}  Precision={prec:.0%}  (n={len(ev)})")

    # 단일 raw feature 기준선 (mom_3m) + HMM p_bear (Step1) 비교
    base = df.dropna(subset=[label])
    for col in ["kospi_mom_3m", "kospi_dd_cur"]:
        if col in base and base[label].nunique() > 1:
            auc = roc_auc_score(base[label], -base[col])  # 음일수록 bear
            print(f"    [raw {col:14}] AUC={auc:.3f}")
    if hmm_col and hmm_col in df:
        b = df.dropna(subset=[label, hmm_col])
        if b[label].nunique() > 1:
            print(f"    [Step1 HMM p_bear  ] AUC={roc_auc_score(b[label], b[hmm_col]):.3f}")
    # AI v2 (겹침구간)
    if ai_map:
        ov = base[base["pred_month"].isin(ai_map.keys())].copy()
        if len(ov) and ov[label].nunique() > 1:
            ov["ai"] = ov["pred_month"].map(ai_map)
            tp = int(((ov["ai"] == 1) & (ov[label] == 1)).sum())
            fn = int(((ov["ai"] == 0) & (ov[label] == 1)).sum())
            fp = int(((ov["ai"] == 1) & (ov[label] == 0)).sum())
            print(f"    [AI v2 겹침 {len(ov)}m   ] Recall={tp/max(tp+fn,1):.0%} "
                  f"Precision={tp/max(tp+fp,1):.0%}")


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    print("Loading kospi/sp500/sox daily...")
    kospi = load_series(conn, "kospi")
    spx = load_series(conn, "sp500")
    sox = load_series(conn, "sox")
    conn.close()
    print(f"  kospi {kospi.index.min().date()}~{kospi.index.max().date()} ({len(kospi)})")

    series = {"kospi": kospi, "spx": spx, "sox": sox}
    m_ends = month_ends(kospi.index)

    rows = []
    for i in range(len(m_ends) - 1):
        e = m_ends[i]
        row = {"pred_month": pd.Period(m_ends[i + 1], freq="M").strftime("%Y-%m"),
               "cutoff": e.isoformat()}
        ok = True
        for pre, s in series.items():
            f = feats(s, e)
            if f is None:
                ok = False
                break
            for k, v in f.items():
                row[f"{pre}_{k}"] = v
        if not ok:
            continue
        for name, kind, months, thr in LABELS:
            row[name] = make_label(kospi, e, kind, months, thr)
        rows.append(row)

    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if any(c.startswith(p + "_") for p in series)]
    df.to_csv(OUT_CSV, index=False)
    print(f"  panel: {len(df)}개월, feature {len(feat_cols)}개 → {OUT_CSV}")

    # Step1 HMM p_bear 병합 (있으면)
    hmm_col = None
    if HMM_CSV.exists():
        h = pd.read_csv(HMM_CSV)
        if "p_bear_2s" in h:
            df = df.merge(h[["pred_month", "p_bear_2s"]], on="pred_month", how="left")
            hmm_col = "p_bear_2s"

    ai_map = {}
    if AI_JSON.exists():
        ai_map = {r["as_of"][:7]: int(r.get("judgment") == "약세")
                  for r in json.load(open(AI_JSON))}

    for name, kind, months, thr in LABELS:
        rf_p, gb_p = walkforward(df, feat_cols, name, months)
        report(df, name, months, feat_cols, rf_p, gb_p, hmm_col, ai_map)

    print(f"\n{'='*72}")
    print("판정: dd 라벨(특히 6m)에서 RF/GBM AUC가 1m(0.6)을 확실히 넘으면 → Step2 성공.")
    print("      (독립표본 적으니 AUC만 보지 말고 raw feature/HMM 기준선 대비 우위로 판단)")


if __name__ == "__main__":
    main()
