"""
analysis/build_ml_features.py

AI v2 ML 필터용 Tier 1 (4 feature) feature 추출.
lookahead-free, strict cutoff (as_of - 1d).

입력:
  - analysis/regime_agent_multimodel_results_gemini.json (AI v2 98개월 결과)
  - Railway PG (KOSPI 069500 종가, VIX, 외국인 수급)

출력:
  - analysis/ml_features.csv

사용:
  .venv/bin/python analysis/build_ml_features.py
"""

import os
import json
import numpy as np
import pandas as pd
import psycopg2
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')


def load_kospi_prices(conn):
    """KOSPI ETF 069500 일별 종가."""
    return pd.read_sql(
        "SELECT trade_date::date AS dt, close::float AS close "
        "FROM alpha_lab.daily_price WHERE stock_code='069500' ORDER BY trade_date",
        conn,
    )


def load_macro_daily(conn, indicator):
    """일별 매크로 지표 (VIX, 외국인 수급 등)."""
    df = pd.read_sql(
        "SELECT period::date AS dt, value::float AS value "
        "FROM alpha_lab.macro_indicators WHERE indicator=%s AND freq='D' ORDER BY period",
        conn, params=(indicator,)
    )
    return df


def kospi_ret_months(prices_df, as_of, n_months):
    """as_of 기준 (as_of - n_months) ~ (as_of - 1d)의 KOSPI 누적 수익률 (%)."""
    cutoff = as_of - timedelta(days=1)
    start = as_of - relativedelta(months=n_months)
    sub = prices_df[(prices_df['dt'] >= start) & (prices_df['dt'] <= cutoff)]
    if len(sub) < 2:
        return np.nan
    return (sub['close'].iloc[-1] / sub['close'].iloc[0] - 1) * 100


def vix_zscore(vix_df, as_of, lookback_months=3):
    """VIX 가장 최근 값의 직전 N개월 표본 대비 z-score."""
    cutoff = as_of - timedelta(days=1)
    start = as_of - relativedelta(months=lookback_months)
    sub = vix_df[(vix_df['dt'] >= start) & (vix_df['dt'] <= cutoff)]
    if len(sub) < 10:
        return np.nan
    vals = sub['value'].values
    if vals.std() == 0:
        return 0.0
    return (vals[-1] - vals.mean()) / vals.std()


def foreign_5d_zscore(foreign_df, as_of, history_days=180):
    """외국인 KOSPI 5일 누적 순매수, 직전 history_days 기간 5일 누적 분포 대비 z-score."""
    cutoff = as_of - timedelta(days=1)
    start = as_of - timedelta(days=history_days)
    sub = foreign_df[(foreign_df['dt'] >= start) & (foreign_df['dt'] <= cutoff)]
    if len(sub) < 30:
        return np.nan
    vals = sub['value'].values
    last5 = vals[-5:].sum() if len(vals) >= 5 else vals.sum()
    rolling = pd.Series(vals).rolling(5).sum().dropna().values
    if rolling.std() == 0:
        return 0.0
    return (last5 - rolling.mean()) / rolling.std()


def main():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])

    print("Loading AI v2 results...")
    with open(Path(__file__).parent / 'regime_agent_multimodel_results_gemini.json') as f:
        records = json.load(f)
    records.sort(key=lambda r: r['as_of'])

    print("Loading DB data (KOSPI 069500, VIX, 외국인)...")
    kospi = load_kospi_prices(conn)
    vix = load_macro_daily(conn, 'vix')
    foreign = load_macro_daily(conn, 'investor_foreign_kospi')
    print(f"  KOSPI rows: {len(kospi)}, VIX rows: {len(vix)}, 외국인 rows: {len(foreign)}")
    conn.close()

    rows = []
    for r in records:
        as_of = date.fromisoformat(r['as_of'][:10])
        probs = r.get('probabilities') or {}

        # ── Tier 1 features ──
        # 1) AI 약세 확신도 (P_crash + P_bear)
        f1 = float(probs.get('크래시', 0) + probs.get('약세', 0))
        # 2) KOSPI 직전 6개월 수익률
        f2 = kospi_ret_months(kospi, as_of, 6)
        # 3) VIX z-score (3개월)
        f3 = vix_zscore(vix, as_of, 3)
        # 4) 외국인 5일 누적 z-score
        f4 = foreign_5d_zscore(foreign, as_of)

        # ── Label & meta ──
        actual = r.get('kospi_next_month_return')
        if actual is None:
            continue
        label_real_bear = int(actual < -3)
        ai_says_bear = int(r.get('judgment') == '약세')

        rows.append({
            'as_of': r['as_of'][:10],
            'ai_says_bear': ai_says_bear,
            'ai_v2_p_bear_total': f1,
            'kospi_ret_6m': f2,
            'vix_zscore_3m': f3,
            'foreign_5d_norm': f4,
            'kospi_next_ret': actual,
            'real_bear': label_real_bear,
        })

    df = pd.DataFrame(rows)
    out_path = Path(__file__).parent / 'ml_features.csv'
    df.to_csv(out_path, index=False)

    # ── Summary ──
    print()
    print("=" * 70)
    print(f"  Saved {len(df)} rows → {out_path}")
    print("=" * 70)
    print(f"AI says Bear (54개월 기대): {df['ai_says_bear'].sum()}")
    print(f"Actually Bear (실제 <-3%): {df['real_bear'].sum()}")
    print()
    print("NaN counts per column:")
    print(df.isna().sum())
    print()

    # Filter task subset
    sub = df[df['ai_says_bear'] == 1]
    n_tp = sub['real_bear'].sum()
    n_fp = len(sub) - n_tp
    print(f"=== Filter task subset (AI Bear calls): {len(sub)}개월 ===")
    print(f"  TP (real bear): {n_tp}")
    print(f"  FP (not real bear): {n_fp}")
    print(f"  Class balance: {n_tp/len(sub)*100:.0f}% positive")
    print()

    # Feature distribution by TP/FP
    print("=== Feature 평균 (TP vs FP, AI Bear calls만) ===")
    for col in ['ai_v2_p_bear_total', 'kospi_ret_6m', 'vix_zscore_3m', 'foreign_5d_norm']:
        tp_mean = sub.loc[sub['real_bear'] == 1, col].mean()
        fp_mean = sub.loc[sub['real_bear'] == 0, col].mean()
        sep = abs(tp_mean - fp_mean)
        print(f"  {col:<25} TP={tp_mean:+.3f}   FP={fp_mean:+.3f}   |Δ|={sep:.3f}")


if __name__ == '__main__':
    main()
