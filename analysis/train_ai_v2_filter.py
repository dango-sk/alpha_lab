"""
analysis/train_ai_v2_filter.py

Tier 0 (2 feature: kospi_ret_6m + foreign_5d_norm) Logistic Regression L2 — AI v2 약세 호출의 TP/FP 분류.

평가:
- TimeSeriesSplit 3-fold (작은 표본에 fold당 sample 확보)
- Bootstrap 200회로 AUC 신뢰구간
- Feature importance
- Naive baseline (AI 그대로) vs ML 필터 confusion matrix 비교

사용:
  .venv/bin/python analysis/train_ai_v2_filter.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, brier_score_loss,
    confusion_matrix
)
from sklearn.utils import resample


FEATURES = ['kospi_ret_6m', 'foreign_5d_norm']


def evaluate_baseline(df_sub):
    """AI 그대로 신뢰 baseline — 모든 AI Bear 호출을 Bear로 인정."""
    y = df_sub['real_bear'].values
    pred = np.ones(len(y))  # 항상 1 (Bear 인정)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall': tp / (tp + fn) if (tp + fn) else 0,
            'precision': tp / (tp + fp) if (tp + fp) else 0}


def main():
    print("=" * 70)
    print("  AI v2 Filter Training — Logistic Regression L2 (Tier 0, 2 features)")
    print("=" * 70)

    df = pd.read_csv(Path(__file__).parent / 'ml_features.csv', parse_dates=['as_of'])
    df = df.sort_values('as_of').reset_index(drop=True)

    # AI Bear 호출 케이스만 필터링 (filter task)
    sub = df[df['ai_says_bear'] == 1].copy().dropna(subset=FEATURES + ['real_bear']).reset_index(drop=True)
    print(f"\n표본: {len(sub)}개월 (AI Bear 호출)")
    print(f"  TP {sub['real_bear'].sum()} / FP {len(sub) - sub['real_bear'].sum()}")
    print(f"  기간: {sub['as_of'].min().date()} ~ {sub['as_of'].max().date()}")

    X = sub[FEATURES].values
    y = sub['real_bear'].values

    # ── 1. Baseline (AI 그대로 신뢰) ──
    base = evaluate_baseline(sub)
    print(f"\n=== Baseline (AI 그대로 — 모든 Bear 인정) ===")
    print(f"  Recall    : {base['recall']*100:.1f}%  (TP {base['tp']} / TP+FN {base['tp']+base['fn']})")
    print(f"  Precision : {base['precision']*100:.1f}%  (TP {base['tp']} / TP+FP {base['tp']+base['fp']})")

    # ── 2. TimeSeriesSplit 3-fold CV ──
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=3)
    fold_aucs, fold_briers = [], []
    print(f"\n=== TimeSeriesSplit 3-fold ===")
    for fold_idx, (tr, te) in enumerate(tscv.split(Xs)):
        if y[te].sum() == 0 or y[te].sum() == len(te):
            print(f"  Fold {fold_idx}: skipped (test set 단일 클래스)")
            continue
        clf = LogisticRegression(penalty='l2', C=0.5, max_iter=1000, class_weight='balanced')
        clf.fit(Xs[tr], y[tr])
        proba = clf.predict_proba(Xs[te])[:, 1]
        auc = roc_auc_score(y[te], proba)
        brier = brier_score_loss(y[te], proba)
        fold_aucs.append(auc)
        fold_briers.append(brier)
        print(f"  Fold {fold_idx}: AUC={auc:.3f}  Brier={brier:.3f}  "
              f"train={len(tr)} ({y[tr].sum()} bears)  test={len(te)} ({y[te].sum()} bears)")

    if fold_aucs:
        print(f"\n  CV AUC mean   : {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
        print(f"  CV Brier mean : {np.mean(fold_briers):.3f}")

    # ── 3. Bootstrap CI (in-sample 신뢰구간) ──
    n_boot = 200
    boot_aucs = []
    print(f"\n=== Bootstrap {n_boot}회 (in-sample AUC 신뢰구간) ===")
    for i in range(n_boot):
        idx = resample(np.arange(len(Xs)), n_samples=len(Xs), random_state=i)
        # 표본에 한 클래스만 있으면 skip
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        try:
            clf_b = LogisticRegression(penalty='l2', C=0.5, max_iter=1000, class_weight='balanced')
            clf_b.fit(Xs[idx], y[idx])
            proba_b = clf_b.predict_proba(Xs)[:, 1]
            boot_aucs.append(roc_auc_score(y, proba_b))
        except Exception:
            continue

    if boot_aucs:
        lo, mid, hi = np.percentile(boot_aucs, [2.5, 50, 97.5])
        print(f"  AUC median: {mid:.3f}  95% CI: [{lo:.3f}, {hi:.3f}]")
        print(f"  AUC > 0.5 (random보다 나음) bootstrap 비율: {np.mean(np.array(boot_aucs) > 0.5)*100:.0f}%")

    # ── 4. Full-data 학습 & feature importance ──
    print(f"\n=== Feature 계수 (L2, full data, standardized) ===")
    clf_full = LogisticRegression(penalty='l2', C=0.5, max_iter=1000, class_weight='balanced')
    clf_full.fit(Xs, y)
    for name, coef in zip(FEATURES, clf_full.coef_[0]):
        sign = "약세 ↑" if coef > 0 else "약세 ↓"
        print(f"  {name:<25} {coef:+.3f}   ({sign})")
    print(f"  intercept{' '*16}{clf_full.intercept_[0]:+.3f}")

    # ── 5. ML 필터 적용 vs Baseline 비교 (in-sample, 참고용) ──
    proba_full = clf_full.predict_proba(Xs)[:, 1]
    for thr in [0.3, 0.4, 0.5, 0.6]:
        pred_thr = (proba_full >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred_thr, labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if (tp + fn) else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        if rec + prec > 0:
            f1 = 2 * rec * prec / (rec + prec)
        else:
            f1 = 0
        print(f"\n=== Threshold {thr} (in-sample) ===")
        print(f"  Recall    : {rec*100:.1f}% ({tp}/{tp+fn})")
        print(f"  Precision : {prec*100:.1f}% ({tp}/{tp+fp})")
        print(f"  F1        : {f1:.3f}")
        print(f"  Bear calls 줄어든 수: {len(sub)} → {pred_thr.sum()}")

    # ── 6. Save with proba for further analysis ──
    sub['ml_proba'] = proba_full
    out = Path(__file__).parent / 'ml_features_with_proba.csv'
    sub.to_csv(out, index=False)
    print(f"\nSaved → {out}")

    # ── 7. Verdict ──
    print(f"\n{'=' * 70}")
    print(f"  Verdict")
    print(f"{'=' * 70}")
    cv_auc = np.mean(fold_aucs) if fold_aucs else 0
    boot_med = np.median(boot_aucs) if boot_aucs else 0
    if cv_auc >= 0.70 and boot_med >= 0.65:
        print("  ✓ 의미 있는 신호 — production 통합 고려 가능")
    elif cv_auc >= 0.55 and boot_med >= 0.55:
        print("  △ 약한 신호 — Tier 2 확장 또는 다른 feature 탐색")
    else:
        print("  ✗ 학습 가능한 신호 없음 — AI v2 + ML 방향 폐기 권장")


if __name__ == '__main__':
    main()
