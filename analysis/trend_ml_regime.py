"""
analysis/trend_ml_regime.py

학습형 추세 레짐: 모멘텀 feature → 로지스틱/GBM → P(상승추세) → in/out.
rule-based(MA) 아닌 *예측 모델*로 강세/약세를 분류하고, 그 신호로 타이밍.

목표: 강세/약세를 정확히 잡기 (분류 정확도) + 실제 수익 (PnL).
정답지(label): forward H개월 수익률 > 0 → 강세(1), 아니면 약세(0).
  H=1, H=3 둘 다 평가 (1개월 방향은 어렵고 3개월이 더 예측가능 — 데이터가 말함).

feature(전부 cutoff e 이전, lookahead-free):
  ret_1/3/6/12m, ma_gap_50/100/150/200, ma_slope_50, vol_1/3m, dist_high, up_days_1m
walk-forward: expanding + embargo(H개월 purge, 라벨 겹침 차단).
PnL: P(상승)>=임계 → 보유, 아니면 현금. 임계는 전반부<2012로 튜닝. 거래비용 반영.
비교: BM(무손절), MA150 추세추종(rule-based 승자), 학습모델.

입력:  Railway PG (kospi freq='D')
출력:  analysis/trend_ml_features.csv

사용:  .venv/bin/python analysis/trend_ml_regime.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent / ".env")

OUT_CSV = Path(__file__).parent / "trend_ml_features.csv"
MIN_TRAIN = 60
SPLIT = pd.Timestamp("2012-01-01")
COST = 0.001          # 편도 거래비용 0.1%
ANN = 252


def load_kospi(conn):
    df = pd.read_sql(
        "SELECT period::date AS dt, value::float AS close FROM alpha_lab.macro_indicators "
        "WHERE indicator='kospi' AND freq='D' ORDER BY period", conn)
    df["dt"] = pd.to_datetime(df["dt"])
    return df.dropna().drop_duplicates(subset="dt").set_index("dt")["close"].sort_index()


def month_ends(idx):
    ym = pd.PeriodIndex(idx, freq="M")
    last = {}
    for dt, p in zip(idx, ym):
        last[p] = dt
    return [last[p] for p in sorted(last)]


def feats(s):
    """cutoff까지 시리즈 s로 모멘텀·추세 feature."""
    if len(s) < 252:
        return None
    p = s.iloc[-1]
    ret = s.pct_change()
    ma = lambda n: s.iloc[-n:].mean()
    return {
        "ret_1m": p / s.iloc[-22] - 1,
        "ret_3m": p / s.iloc[-64] - 1,
        "ret_6m": p / s.iloc[-127] - 1,
        "ret_12m": p / s.iloc[-253] - 1 if len(s) >= 253 else p / s.iloc[0] - 1,
        "ma_gap_50": p / ma(50) - 1,
        "ma_gap_100": p / ma(100) - 1,
        "ma_gap_150": p / ma(150) - 1,
        "ma_gap_200": p / ma(200) - 1,
        "ma_slope_50": ma(50) / s.iloc[-71:-21].mean() - 1 if len(s) >= 71 else 0.0,
        "vol_1m": ret.iloc[-21:].std() * np.sqrt(21),
        "vol_3m": ret.iloc[-63:].std() * np.sqrt(63),
        "dist_high": p / s.iloc[-252:].max() - 1,
        "up_days_1m": float((ret.iloc[-21:] > 0).mean()),
    }


def fwd_bull(close, e, months):
    end = e + relativedelta(months=months)
    fut = close[(close.index > e) & (close.index <= end)]
    if len(fut) < 3:
        return np.nan
    return int(fut.iloc[-1] / close.loc[e] - 1 > 0)


def walkforward(df, feat_cols, label, horizon):
    """expanding + embargo(H) walk-forward. 로지스틱 & GBM의 P(강세)."""
    X = df[feat_cols].values
    y = df[label].values
    lr_p = np.full(len(df), np.nan)
    gb_p = np.full(len(df), np.nan)
    for i in range(MIN_TRAIN, len(df)):
        cut = i - horizon
        if cut < MIN_TRAIN // 2:
            continue
        Xtr, ytr = X[:cut], y[:cut]
        m = ~np.isnan(ytr)
        Xtr, ytr = Xtr[m], ytr[m]
        if ytr.sum() < 5 or (len(ytr) - ytr.sum()) < 5:
            continue
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        lr.fit(sc.transform(Xtr), ytr)
        lr_p[i] = lr.predict_proba(sc.transform(X[i:i + 1]))[0, 1]
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                            max_iter=200, random_state=42)
        gb.fit(Xtr, ytr)
        gb_p[i] = gb.predict_proba(X[i:i + 1])[0, 1]
    return lr_p, gb_p


def clf_report(df, label, prob_col, name):
    ev = df.dropna(subset=[prob_col, label])
    if ev[label].nunique() < 2:
        print(f"    {name}: (단일 클래스)")
        return
    auc = roc_auc_score(ev[label], ev[prob_col])
    pred = (ev[prob_col] >= 0.5).astype(int)
    acc = (pred == ev[label]).mean()
    bull = ev[label] == 1
    bear = ev[label] == 0
    bull_rec = (pred[bull] == 1).mean()      # 강세 맞춘 비율
    bear_rec = (pred[bear] == 0).mean()      # 약세 맞춘 비율
    print(f"    {name:11} AUC={auc:.3f}  정확도={acc:.0%}  강세recall={bull_rec:.0%}  약세recall={bear_rec:.0%}")


def metrics(eq):
    total = eq.iloc[-1] / eq.iloc[0] - 1
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    d = eq.pct_change().dropna()
    sharpe = d.mean() / d.std() * np.sqrt(ANN) if d.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    return total, cagr, sharpe, mdd


def sim_signal(close, invested_by_month, cost=COST):
    """월별 invested 플래그(True/False)로 일별 시뮬."""
    ret = close.pct_change().fillna(0.0)
    eq = pd.Series(index=close.index, dtype=float)
    v, prev, sw = 1.0, True, 0
    for i, (dt, px) in enumerate(close.items()):
        inv = invested_by_month.get(dt.strftime("%Y-%m"), True)
        if i > 0 and prev:
            v *= (1 + ret.iloc[i])
        eq.iloc[i] = v
        if inv != prev:
            sw += 1
            v *= (1 - cost)
        prev = inv
    return eq, sw


def sim_ma(close, window, cost=COST):
    ma = close.rolling(window).mean()
    ret = close.pct_change().fillna(0.0)
    eq = pd.Series(index=close.index, dtype=float)
    v, prev, sw = 1.0, True, 0
    for i, (dt, px) in enumerate(close.items()):
        inm = bool(px > ma.iloc[i]) if pd.notna(ma.iloc[i]) else True
        if i > 0 and prev:
            v *= (1 + ret.iloc[i])
        eq.iloc[i] = v
        if inm != prev:
            sw += 1
            v *= (1 - cost)
        prev = inm
    return eq, sw


def sim_exposure(close, exp, cost=COST):
    """일별 노출(0~1)로 시뮬. exp[i]=다음날 보유비중. 노출 변할 때 |Δ|*cost 비용."""
    ret = close.pct_change().fillna(0.0).values
    eq = np.empty(len(close))
    v, prev, ch = 1.0, exp[0], 0
    for i in range(len(close)):
        if i > 0:
            v *= (1 + prev * ret[i])
        eq[i] = v
        if exp[i] != prev:
            ch += 1
            v *= (1 - cost * abs(exp[i] - prev))
        prev = exp[i]
    return pd.Series(eq, index=close.index), ch


def combo_compare(close, df, prob_col, ma_window=150):
    """BM · MA단독 · 학습모델단독 · 결합①(AND) · 결합②(비중) 비교."""
    ma = close.rolling(ma_window).mean()
    # 모델 임계: 전반부<2012로 Sharpe 튜닝
    tr_close = close[close.index < SPLIT]
    tr_months = set(df[df["cutoff"] < SPLIT]["pred_month"])
    best_t, best_v = 0.5, -1e9
    for t in np.arange(0.35, 0.66, 0.05):
        inv = {r["pred_month"]: (r[prob_col] >= t) for _, r in df.iterrows()
               if pd.notna(r[prob_col]) and r["pred_month"] in tr_months}
        if len(inv) < 12:
            continue
        eq, _ = sim_signal(tr_close, inv)
        v = metrics(eq)[2]
        if v > best_v:
            best_v, best_t = v, round(t, 2)

    mbull = {r["pred_month"]: (r[prob_col] >= best_t) for _, r in df.iterrows()
             if pd.notna(r[prob_col])}
    idx, pxv, mav = close.index, close.values, ma.values
    n = len(close)
    maex = np.where(np.isnan(mav), 1.0, (pxv > mav).astype(float))
    mdl = np.array([1.0 if mbull.get(d.strftime("%Y-%m"), True) else 0.0 for d in idx])
    exp_map = {
        "BM(보유)": np.ones(n),
        f"MA{ma_window}단독": maex,
        "학습모델단독": mdl,
        "결합①AND": ((mdl > 0) & (maex > 0)).astype(float),
        "결합②비중": 0.5 * mdl + 0.5 * maex,
    }
    start = pd.Timestamp(min(mbull) + "-01")
    print(f"\n{'#'*72}\n  결합 비교 (3m 학습모델 P>={best_t} + MA{ma_window}), 편도비용 {COST*100:.1f}%")
    for ws, wl in [(start, "전체(예측구간)"), (pd.Timestamp("2018-04-01"), "OOS 2018~")]:
        mask = (idx >= ws).to_numpy() if hasattr(idx >= ws, "to_numpy") else (idx >= ws)
        w = close[mask]
        print(f"\n  {wl}  ({w.index[0].date()} ~ {w.index[-1].date()})")
        print(f"    {'전략':16} {'누적':>9} {'CAGR':>7} {'Sharpe':>7} {'MDD':>8} {'전환':>5}")
        bm_sh = metrics(sim_exposure(w, exp_map["BM(보유)"][mask])[0])[2]
        for name, exp in exp_map.items():
            eq, ch = sim_exposure(w, exp[mask])
            t, c, s, m = metrics(eq)
            star = " ★" if name != "BM(보유)" and s > bm_sh else ""
            print(f"    {name:16} {t*100:>8.0f}% {c*100:>6.1f}% {s:>7.2f} {m*100:>7.1f}% {ch:>5}{star}")
    print("  ★=BM Sharpe 초과. 결합①/②가 Sharpe↑ & MDD↓ 둘 다면 결합 성공.")


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    close = load_kospi(conn)
    conn.close()
    print(f"KOSPI {close.index[0].date()} ~ {close.index[-1].date()} ({len(close)})")

    m_ends = month_ends(close.index)
    rows = []
    for i in range(len(m_ends) - 1):
        e = m_ends[i]
        f = feats(close[close.index <= e])
        if f is None:
            continue
        f["pred_month"] = pd.Period(m_ends[i + 1], freq="M").strftime("%Y-%m")
        f["cutoff"] = e
        f["bull_1m"] = fwd_bull(close, e, 1)
        f["bull_3m"] = fwd_bull(close, e, 3)
        rows.append(f)
    df = pd.DataFrame(rows)
    feat_cols = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "ma_gap_50", "ma_gap_100",
                 "ma_gap_150", "ma_gap_200", "ma_slope_50", "vol_1m", "vol_3m",
                 "dist_high", "up_days_1m"]
    df.to_csv(OUT_CSV, index=False)
    print(f"panel {len(df)}개월, feature {len(feat_cols)}개")

    # 분류 성능 (강세/약세 잘 잡나)
    best = {}
    for H, label in [(1, "bull_1m"), (3, "bull_3m")]:
        print(f"\n{'='*72}\n  정답지: {label} (forward {H}m 수익률>0 → 강세)  "
              f"강세 {int(df[label].sum())}/{df[label].notna().sum()}")
        lr_p, gb_p = walkforward(df, feat_cols, label, H)
        df[f"lr_{label}"] = lr_p
        df[f"gb_{label}"] = gb_p
        clf_report(df, label, f"lr_{label}", "로지스틱")
        clf_report(df, label, f"gb_{label}", "GBM")
        best[label] = f"gb_{label}"

    # PnL: 학습모델 P(강세)로 in/out (임계 전반부 튜닝) vs BM vs MA150
    df["cutoff"] = pd.to_datetime(df["cutoff"])
    for H, label in [(1, "bull_1m"), (3, "bull_3m")]:
        prob_col = best[label]
        # 임계 튜닝 (전반부<2012, train Sharpe 최대)
        tr_months = set(df[df["cutoff"] < SPLIT]["pred_month"])
        tr_close = close[close.index < SPLIT]
        best_t, best_v = 0.5, -1e9
        for t in np.arange(0.35, 0.66, 0.05):
            inv = {r["pred_month"]: (r[prob_col] >= t) for _, r in df.iterrows()
                   if pd.notna(r[prob_col]) and r["pred_month"] in tr_months}
            if len(inv) < 12:
                continue
            eq, _ = sim_signal(tr_close, inv)
            v = metrics(eq)[2]
            if v > best_v:
                best_v, best_t = v, round(t, 2)

        inv_all = {r["pred_month"]: (r[prob_col] >= best_t) for _, r in df.iterrows()
                   if pd.notna(r[prob_col])}
        start = pd.Timestamp(min(inv_all) + "-01")
        # OOS 구간 (2018~)
        for win_start, wl in [(start, "전체(예측가능구간)"), (pd.Timestamp("2018-04-01"), "OOS 2018~")]:
            win = close[close.index >= win_start]
            print(f"\n  [{label} PnL, 임계 P>={best_t}] {wl}  ({win.index[0].date()}~{win.index[-1].date()})")
            print(f"    {'전략':18} {'누적':>9} {'CAGR':>7} {'Sharpe':>7} {'MDD':>8} {'전환':>5}")
            bm, _ = sim_signal(win, {})
            ml, sw = sim_signal(win, inv_all)
            ma, sw2 = sim_ma(win, 150)
            bm_sh = metrics(bm)[2]
            for nm, eq, s in [("BM", bm, 0), ("학습모델", ml, sw), ("MA150", ma, sw2)]:
                t, c, sh, m = metrics(eq)
                star = " ★" if nm != "BM" and sh > bm_sh else ""
                print(f"    {nm:18} {t*100:>8.0f}% {c*100:>6.1f}% {sh:>7.2f} {m*100:>7.1f}% {s:>5}{star}")

    # ── 결합 비교 (3m 학습모델 + MA150) ──
    combo_compare(close, df, "gb_bull_3m", ma_window=150)

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")
    print("판정: 학습모델 AUC가 0.5 넘고(강세/약세 분류력), PnL에서 ★(BM 초과)면 성공.")


if __name__ == "__main__":
    main()
