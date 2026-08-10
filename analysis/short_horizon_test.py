"""
analysis/short_horizon_test.py

"기간이 짧을수록 예측력이 떨어지나?"를 직접 확인.
주간 단위로 여러 forward horizon(1주~6달)의 *방향(상승/하락)* 예측 AUC를 비교.

horizon: 5,10,21,42,63,126 거래일 (≈ 1주,2주,1달,2달,3달,6달)
정답지: forward H일 수익률 > 0 → 상승(1)
신호:   ① raw 모멘텀(ma_gap_50) 단변량 AUC (학습 불필요)
        ② walk-forward 로지스틱(모멘텀 feature) AUC
모두 lookahead-free (feature는 cutoff i 이하, label은 i 이후). 라벨 겹침 embargo 처리.

입력:  Railway PG (kospi freq='D')
사용:  .venv/bin/python analysis/short_horizon_test.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

HORIZONS = [(5, "1주"), (10, "2주"), (21, "1달"), (42, "2달"), (63, "3달"), (126, "6달")]
CADENCE = 5          # 주간 단위 결정점
MIN_HIST = 252       # feature 계산 최소 일수
MIN_TRAIN = 150      # walk-forward 최소 학습 표본


def load_close(conn):
    df = pd.read_sql(
        "SELECT period::date AS dt, value::float AS close FROM alpha_lab.macro_indicators "
        "WHERE indicator='kospi' AND freq='D' ORDER BY period", conn)
    df["dt"] = pd.to_datetime(df["dt"])
    return df.dropna().drop_duplicates(subset="dt").set_index("dt")["close"].sort_index()


def feat_row(c, dret, i):
    """index i(=cutoff)까지의 모멘텀 feature."""
    return [
        c[i] / c[i - 5] - 1,                       # ret_1주
        c[i] / c[i - 21] - 1,                      # ret_1달
        c[i] / c[i - 63] - 1,                      # ret_3달
        c[i] / c[i - 20:i + 1].mean() - 1,         # ma_gap_20
        c[i] / c[i - 50:i + 1].mean() - 1,         # ma_gap_50
        dret[i - 21:i + 1].std(),                  # vol_1달
    ]


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    close = load_close(conn)
    conn.close()
    c = close.values
    dret = np.concatenate([[0.0], np.diff(c) / c[:-1]])
    n = len(c)
    print(f"KOSPI {close.index[0].date()} ~ {close.index[-1].date()} ({n}일)\n")

    print(f"  {'horizon':10} {'표본':>6} {'상승비율':>8} {'raw모멘텀 AUC':>13} {'로지스틱 AUC':>13}")
    print("  " + "-" * 56)
    for H, name in HORIZONS:
        idxs = [i for i in range(MIN_HIST, n - H, CADENCE)]
        X, y, magap = [], [], []
        for i in idxs:
            X.append(feat_row(c, dret, i))
            y.append(int(c[i + H] / c[i] - 1 > 0))
            magap.append(c[i] / c[i - 50:i + 1].mean() - 1)   # 단변량 신호
        X, y, magap = np.array(X), np.array(y), np.array(magap)

        raw_auc = roc_auc_score(y, magap) if len(set(y)) > 1 else float("nan")

        # walk-forward 로지스틱 (embargo = H/CADENCE step)
        emb = max(1, H // CADENCE)
        probs = np.full(len(y), np.nan)
        for k in range(MIN_TRAIN, len(y)):
            cut = k - emb
            if cut < MIN_TRAIN // 2:
                continue
            ytr = y[:cut]
            if ytr.sum() < 10 or (len(ytr) - ytr.sum()) < 10:
                continue
            sc = StandardScaler().fit(X[:cut])
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500)
            lr.fit(sc.transform(X[:cut]), ytr)
            probs[k] = lr.predict_proba(sc.transform(X[k:k + 1]))[0, 1]
        m = ~np.isnan(probs)
        lr_auc = roc_auc_score(y[m], probs[m]) if m.sum() > 20 and len(set(y[m])) > 1 else float("nan")

        print(f"  {name:10} {len(y):>6} {y.mean()*100:>7.0f}% {raw_auc:>13.3f} {lr_auc:>13.3f}")

    print("\n  (AUC 0.5=동전던지기. 기간 짧을수록 0.5에 가까워지면 '짧으면 예측 불가' 확인)")
    print("  주의: H>5는 주간 결정점끼리 forward 구간 겹침 → AUC 다소 낙관적.")


if __name__ == "__main__":
    main()
