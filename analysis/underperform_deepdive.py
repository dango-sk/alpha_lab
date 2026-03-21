"""
KOSPI 대비 언더퍼폼 월 Deep Dive 분석

1. 섹터 쏠림: 포트폴리오 vs KOSPI 200 섹터 비중
2. 종목 집중도: 상위 5개 종목 비중 합계
3. 개별 종목 기여도: 가장 손해 본 종목 Top 5
4. 팩터 노출 편향: 저PER/저PBR 등 밸류 트랩 여부
"""
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config.settings import BACKTEST_CONFIG
from lib.db import get_conn
from scripts.step7_backtest import (
    get_monthly_rebalance_dates, calc_etf_return, _apply_mcap_cap,
)
from lib.factor_engine import (
    score_stocks_from_strategy, code_to_module, DEFAULT_STRATEGY_CODE,
    clear_factor_cache,
)


def get_strategy_stocks(conn, calc_date, top_n=30):
    strategy_module = code_to_module(DEFAULT_STRATEGY_CODE)
    candidates = score_stocks_from_strategy(conn, calc_date, strategy_module)
    filtered = []
    for code, score in candidates:
        if len(filtered) >= top_n:
            break
        price_exists = conn.execute("""
            SELECT COUNT(*) FROM daily_price
            WHERE stock_code = ? AND trade_date >= date(?, '-5 days')
              AND trade_date <= ?
        """, (code, calc_date, calc_date)).fetchone()[0]
        if price_exists == 0:
            continue
        vol_data = conn.execute("""
            SELECT AVG(close * volume) FROM daily_price
            WHERE stock_code = ? AND trade_date <= ?
              AND trade_date >= date(?, '-30 days')
        """, (code, calc_date, calc_date)).fetchone()
        if vol_data and vol_data[0] and vol_data[0] >= 100_000_000:
            filtered.append((code, score))
    return filtered


def get_stock_return(conn, code, start_date, end_date):
    sp = conn.execute("""
        SELECT close FROM daily_price
        WHERE stock_code = ? AND trade_date >= ?
        ORDER BY trade_date ASC LIMIT 1
    """, (code, start_date)).fetchone()
    ep = conn.execute("""
        SELECT close FROM daily_price
        WHERE stock_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT 1
    """, (code, end_date)).fetchone()
    if sp and ep and sp[0] > 0:
        return (ep[0] - sp[0]) / sp[0]
    return 0.0


def get_stock_name(conn, code):
    r = conn.execute("SELECT stock_name FROM stock_master WHERE stock_code = ?", (code,)).fetchone()
    return r[0] if r else code


def get_sector(conn, code):
    r = conn.execute("SELECT sector FROM stock_master WHERE stock_code = ?", (code,)).fetchone()
    if r and r[0]:
        return r[0].replace("코스피 ", "").replace("코스닥 ", "")
    return "미분류"


def get_mcap_weights(conn, stocks, start_date):
    """시총 비례 비중 + cap 적용"""
    raw_mcaps = []
    for code, _ in stocks:
        row = conn.execute("""
            SELECT market_cap FROM daily_price
            WHERE stock_code = ? AND trade_date >= ?
            ORDER BY trade_date ASC LIMIT 1
        """, (code, start_date)).fetchone()
        raw_mcaps.append(row[0] if row and row[0] else 0)

    cap = BACKTEST_CONFIG.get("weight_cap_pct", 10) / 100
    weights = _apply_mcap_cap(raw_mcaps, cap=cap)
    return weights


def get_valuation_snapshot(conn, code, calc_date):
    """해당 시점의 밸류에이션 지표"""
    r = conn.execute("""
        SELECT per, pbr, ev_ebitda FROM valuation_factors
        WHERE stock_code = ? AND calc_date = ?
    """, (code, calc_date)).fetchone()
    if r:
        return {"per": r[0], "pbr": r[1], "ev_ebitda": r[2]}
    return {"per": None, "pbr": None, "ev_ebitda": None}


def main():
    conn = get_conn()
    rb_dates = get_monthly_rebalance_dates(conn)
    total_months = len(rb_dates) - 1

    print("=" * 70)
    print("KOSPI 대비 언더퍼폼 월 Deep Dive")
    print("=" * 70)

    # ── 1단계: 언더퍼폼 월 식별 ──
    print("\n▶ 월별 성과 계산 중...")
    BACKTEST_CONFIG["weight_cap_pct"] = 10

    month_data = []
    for i in range(total_months):
        start, end = rb_dates[i], rb_dates[i + 1]
        stocks = get_strategy_stocks(conn, start, 30)
        if not stocks:
            continue

        weights = get_mcap_weights(conn, stocks, start)
        kospi_ret = calc_etf_return(conn, "KS200", start, end)
        if kospi_ret is None:
            kospi_ret = 0.0

        # 종목별 수익률 + 비중 기여도
        stock_details = []
        port_ret = 0.0
        for j, (code, score) in enumerate(stocks):
            ret = get_stock_return(conn, code, start, end)
            w = weights[j]
            contrib = ret * w
            port_ret += contrib
            stock_details.append({
                "code": code,
                "name": get_stock_name(conn, code),
                "sector": get_sector(conn, code),
                "weight": w,
                "return": ret,
                "contribution": contrib,
                "score": score,
                "valuation": get_valuation_snapshot(conn, code, start),
            })

        excess = port_ret - kospi_ret
        month_data.append({
            "date": start,
            "port_ret": port_ret,
            "kospi_ret": kospi_ret,
            "excess": excess,
            "stocks": stock_details,
        })

        if (i + 1) % 12 == 0:
            print(f"  ... {i+1}/{total_months}개월")

        clear_factor_cache()

    # 언더퍼폼 월 (excess < 0)
    under_months = [m for m in month_data if m["excess"] < 0]
    under_months.sort(key=lambda x: x["excess"])

    print(f"\n전체 {len(month_data)}개월 중 언더퍼폼 {len(under_months)}개월 ({len(under_months)/len(month_data):.0%})")
    print(f"평균 언더퍼폼: {np.mean([m['excess'] for m in under_months])*100:.2f}%p")

    # ── 2단계: 최악의 10개월 상세 분석 ──
    worst_n = min(10, len(under_months))
    print(f"\n{'='*70}")
    print(f"최악의 {worst_n}개월 상세 분석")
    print(f"{'='*70}")

    all_worst_sectors = []
    all_worst_stocks = []
    all_worst_valuations = {"per": [], "pbr": [], "ev_ebitda": []}

    for rank, m in enumerate(under_months[:worst_n]):
        print(f"\n{'─'*70}")
        print(f"#{rank+1} | {m['date']} | 전략 {m['port_ret']*100:+.1f}% vs KOSPI {m['kospi_ret']*100:+.1f}% | 초과 {m['excess']*100:+.1f}%p")
        print(f"{'─'*70}")

        stocks = m["stocks"]

        # (1) 섹터 분포
        sector_weight = {}
        for s in stocks:
            sec = s["sector"]
            sector_weight[sec] = sector_weight.get(sec, 0) + s["weight"]
        sorted_sectors = sorted(sector_weight.items(), key=lambda x: -x[1])

        print(f"\n  [섹터 비중]")
        for sec, w in sorted_sectors[:5]:
            print(f"    {sec}: {w*100:.1f}%")
            all_worst_sectors.append(sec)

        top_sector_weight = sorted_sectors[0][1] if sorted_sectors else 0
        top3_sector_weight = sum(w for _, w in sorted_sectors[:3])
        print(f"    → Top1 섹터 비중: {top_sector_weight*100:.1f}%, Top3 합계: {top3_sector_weight*100:.1f}%")

        # (2) 종목 집중도
        top5_weight = sum(s["weight"] for s in sorted(stocks, key=lambda x: -x["weight"])[:5])
        print(f"\n  [종목 집중도] 상위 5종목 비중: {top5_weight*100:.1f}%")

        # (3) 손실 기여 Top 5
        worst_contribs = sorted(stocks, key=lambda x: x["contribution"])[:5]
        print(f"\n  [손실 기여 Top 5]")
        for s in worst_contribs:
            print(f"    {s['name']:<12} ({s['sector']:<8}) "
                  f"비중 {s['weight']*100:.1f}% × 수익률 {s['return']*100:+.1f}% "
                  f"= 기여 {s['contribution']*100:+.2f}%p")
            all_worst_stocks.append(s["name"])

        # (4) 밸류에이션 분포
        pers = [s["valuation"]["per"] for s in stocks if s["valuation"]["per"] and s["valuation"]["per"] > 0]
        pbrs = [s["valuation"]["pbr"] for s in stocks if s["valuation"]["pbr"] and s["valuation"]["pbr"] > 0]
        evs = [s["valuation"]["ev_ebitda"] for s in stocks if s["valuation"]["ev_ebitda"] and s["valuation"]["ev_ebitda"] > 0]

        print(f"\n  [밸류에이션 분포]")
        if pers:
            print(f"    PER: 중앙값 {np.median(pers):.1f}, 평균 {np.mean(pers):.1f}")
            all_worst_valuations["per"].extend(pers)
        if pbrs:
            print(f"    PBR: 중앙값 {np.median(pbrs):.2f}, 평균 {np.mean(pbrs):.2f}")
            all_worst_valuations["pbr"].extend(pbrs)
        if evs:
            print(f"    EV/EBITDA: 중앙값 {np.median(evs):.1f}, 평균 {np.mean(evs):.1f}")
            all_worst_valuations["ev_ebitda"].extend(evs)

    # ── 3단계: 패턴 종합 ──
    print(f"\n{'='*70}")
    print("패턴 종합 분석")
    print(f"{'='*70}")

    # 섹터 빈도
    from collections import Counter
    sec_counts = Counter(all_worst_sectors)
    print(f"\n[언더퍼폼 월에 자주 등장하는 섹터 (Top 5)]")
    for sec, cnt in sec_counts.most_common(5):
        print(f"  {sec}: {cnt}회")

    # 종목 빈도
    stock_counts = Counter(all_worst_stocks)
    print(f"\n[손실 기여 상위에 반복 등장하는 종목]")
    for name, cnt in stock_counts.most_common(10):
        if cnt >= 2:
            print(f"  {name}: {cnt}회")

    # 밸류에이션 비교: 언더퍼폼 월 vs 전체
    print(f"\n[밸류에이션 비교: 언더퍼폼 월 포트폴리오]")
    if all_worst_valuations["per"]:
        print(f"  PER 중앙값: {np.median(all_worst_valuations['per']):.1f}")
    if all_worst_valuations["pbr"]:
        print(f"  PBR 중앙값: {np.median(all_worst_valuations['pbr']):.2f}")
    if all_worst_valuations["ev_ebitda"]:
        print(f"  EV/EBITDA 중앙값: {np.median(all_worst_valuations['ev_ebitda']):.1f}")

    # 아웃퍼폼 월과 비교
    over_months = [m for m in month_data if m["excess"] >= 0]
    if over_months:
        print(f"\n[비교: 아웃퍼폼 월 ({len(over_months)}개월)]")
        over_sectors = []
        over_valuations = {"per": [], "pbr": [], "ev_ebitda": []}
        for m in over_months:
            for s in m["stocks"]:
                sec = s["sector"]
                over_sectors.append(sec)
                v = s["valuation"]
                if v["per"] and v["per"] > 0: over_valuations["per"].append(v["per"])
                if v["pbr"] and v["pbr"] > 0: over_valuations["pbr"].append(v["pbr"])
                if v["ev_ebitda"] and v["ev_ebitda"] > 0: over_valuations["ev_ebitda"].append(v["ev_ebitda"])

        over_sec_counts = Counter(over_sectors)
        print(f"  자주 등장하는 섹터: {', '.join(s for s, _ in over_sec_counts.most_common(5))}")
        if over_valuations["per"]:
            print(f"  PER 중앙값: {np.median(over_valuations['per']):.1f} (언더퍼폼: {np.median(all_worst_valuations['per']):.1f})")
        if over_valuations["pbr"]:
            print(f"  PBR 중앙값: {np.median(over_valuations['pbr']):.2f} (언더퍼폼: {np.median(all_worst_valuations['pbr']):.2f})")

    conn.close()
    print("\n완료!")


if __name__ == "__main__":
    main()
