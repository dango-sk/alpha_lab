"""
Step 6: 시그널 생성 (밸류 + 모멘텀)

뉴스/산업 센티먼트 제거. 밸류 점수(Step 3) + 모멘텀 보너스만 계산.
A+M(VM) 전략에서 tech_score로 사용.
"""
import sqlite3
import sys
import bisect
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from config.settings import DB_PATH, BACKTEST_CONFIG


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def calc_momentum(conn, stock_code, as_of_date, lookback_days=60):
    """최근 N일 수익률 (모멘텀)"""
    start_date = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    start_price = conn.execute("""
        SELECT close FROM daily_price
        WHERE stock_code = ? AND trade_date >= ?
        ORDER BY trade_date ASC LIMIT 1
    """, (stock_code, start_date)).fetchone()

    end_price = conn.execute("""
        SELECT close FROM daily_price
        WHERE stock_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT 1
    """, (stock_code, as_of_date)).fetchone()

    if start_price and end_price and start_price[0] > 0:
        return (end_price[0] / start_price[0]) - 1.0
    return None


def _percentile_normalize(scores):
    """점수 배열을 백분위(0~100)로 정규화"""
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    n = len(arr)
    if n == 1:
        return [50.0]
    series = pd.Series(arr)
    ranks = series.rank(method="average")
    percentiles = ((ranks - 1) / (n - 1) * 100).round(1)
    return percentiles.tolist()


def generate_signals_for_date(calc_date):
    """특정 날짜의 시그널 생성 (밸류 재계산 + 모멘텀)"""

    from step3_calc_value_factors import calc_valuation_for_date
    calc_valuation_for_date(target_date=calc_date)

    conn = get_db()

    stocks = conn.execute("""
        SELECT vf.stock_code, vf.value_score, vf.quality_pass,
               sm.stock_name, sm.sector
        FROM valuation_factors vf
        JOIN stock_master sm ON vf.stock_code = sm.stock_code
        WHERE vf.quality_pass = 1 AND vf.calc_date = ?
        ORDER BY vf.value_score DESC
        LIMIT 300
    """, (calc_date,)).fetchall()

    # ─── 모멘텀 계산 ───
    momentum_map = {}
    for code, _, _, _, _ in stocks:
        mom = calc_momentum(conn, code, calc_date, lookback_days=60)
        if mom is not None:
            momentum_map[code] = mom

    # 모멘텀 하위 20% 커트라인 (밸류트랩 방지)
    if momentum_map:
        sorted_mom = sorted(momentum_map.values())
        cutoff_idx = max(1, len(sorted_mom) // 5)
        mom_cutoff = sorted_mom[cutoff_idx]
    else:
        mom_cutoff = -999

    # ─── raw 점수 계산 ───
    raw_signals = []
    for code, value_score, _, name, sector in stocks:
        mom = momentum_map.get(code)
        if mom is not None and mom <= mom_cutoff:
            continue

        # 모멘텀 가산/감산 (+-5점)
        MOM_MAX_BONUS = 5
        if mom is not None and momentum_map:
            sorted_moms = sorted(momentum_map.values())
            rank = bisect.bisect_left(sorted_moms, mom)
            percentile = rank / max(len(sorted_moms) - 1, 1)
            mom_bonus = (percentile - 0.5) * 2 * MOM_MAX_BONUS
            mom_bonus = max(-MOM_MAX_BONUS, min(MOM_MAX_BONUS, round(mom_bonus, 1)))
        else:
            mom_bonus = 0

        raw_total = value_score + mom_bonus

        raw_signals.append({
            "stock_code": code,
            "calc_date": calc_date,
            "value_score": value_score,
            "mom_bonus": mom_bonus,
            "raw_total": raw_total,
        })

    if not raw_signals:
        conn.close()
        return []

    # ─── 백분위 정규화 ───
    raw_totals = [s["raw_total"] for s in raw_signals]
    normalized_scores = _percentile_normalize(raw_totals)

    # DB 저장
    for sig, norm_score in zip(raw_signals, normalized_scores):
        conn.execute("""
            INSERT OR REPLACE INTO signals
            (stock_code, calc_date, value_score, news_score, tech_score,
             industry_bonus, total_score, signal_type, signal_label, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
        """, (
            sig["stock_code"], sig["calc_date"], sig["value_score"],
            50.0,  # news_score: 중립 (사용 안함)
            sig["mom_bonus"],
            0,  # industry_bonus: 미사용
            norm_score,
            "hold",  # signal_type: 미사용
            "hold",
        ))

    conn.commit()
    conn.close()

    print(f"    {calc_date}: {len(raw_signals)}종목 시그널 생성")
    return raw_signals


def generate_all_signals():
    """백테스트 기간 전체의 월별 시그널 생성"""
    print("\n" + "=" * 60)
    print("Step 6: 시그널 생성 (밸류 + 모멘텀)")
    print(f"   기간: {BACKTEST_CONFIG['start']} ~ {BACKTEST_CONFIG['end']}")
    print("=" * 60)

    conn = get_db()
    trade_dates = conn.execute("""
        SELECT DISTINCT trade_date FROM daily_price
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, (BACKTEST_CONFIG["start"], BACKTEST_CONFIG["end"])).fetchall()

    monthly_dates = []
    current_month = ""
    for (td,) in trade_dates:
        month = td[:7]
        if month != current_month:
            monthly_dates.append(td)
            current_month = month
    conn.close()

    print(f"  리밸런싱 날짜: {len(monthly_dates)}개월")

    for date in monthly_dates:
        generate_signals_for_date(date)


if __name__ == "__main__":
    print("Step 6: 시그널 생성 (밸류 + 모멘텀)")
    print(f"   DB: {DB_PATH}")
    generate_all_signals()
