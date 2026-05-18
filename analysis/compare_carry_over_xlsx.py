"""
A/C 모드만 실행해서 손절 이벤트를 xlsx로 저장 (분석용 monkey-patch, production 무수정).
시트:
  - A_off     : carry off 모드 손절 이벤트
  - C_natural : carry over 자연 설계 모드 손절 이벤트
  - 2026_진단 : 2026년 월별 레짐/편입수/최대 drawdown (왜 손절이 안 발동했는지)
컬럼: 날짜, 종목코드, 종목명, 최초편입가, 고점, 현재가(손절가), 고점대비%
실행: python analysis/compare_carry_over_xlsx.py
"""
import os, sys, json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd


def _enrich_events(conn, stocks, start_date, events, carry_over, basis):
    """손절 이벤트에 최초편입가/고점/손절가/손실율 부착.
    events: [(code, trigger_dt, loss_pct, weight), ...]
    """
    if not events:
        return []
    codes = [c for c, _, _, _ in events]
    placeholders = ",".join(["?"] * len(codes))

    # 시작일 entry price
    start_rows = conn.execute(f"""
        SELECT dp.stock_code, dp.adj_close
        FROM daily_price dp
        INNER JOIN (
            SELECT stock_code, MIN(trade_date) as d
            FROM daily_price WHERE stock_code IN ({placeholders}) AND trade_date >= ?
            GROUP BY stock_code
        ) t ON dp.stock_code = t.stock_code AND dp.trade_date = t.d
    """, (*codes, start_date)).fetchall()
    entry_map = {r[0]: r[1] for r in start_rows}

    # 일별 가격
    daily_rows = conn.execute(f"""
        SELECT stock_code, trade_date, adj_close
        FROM daily_price
        WHERE stock_code IN ({placeholders}) AND trade_date >= ? AND adj_close > 0
        ORDER BY trade_date
    """, (*codes, start_date)).fetchall()
    by_code = {}
    for c, d, p in daily_rows:
        by_code.setdefault(c, []).append((d, p))

    enriched = []
    for code, trigger_dt, _loss_pct, weight in events:
        entry_price = entry_map.get(code, 0) or 0
        prev = (carry_over or {}).get(code)
        first_entry = prev["first_entry_price"] if prev else entry_price
        peak = prev["peak_price"] if prev else entry_price
        trigger_price = entry_price
        # 시작일 직후부터 trigger_dt까지 peak 갱신 + trigger_price 확정
        for d, p in by_code.get(code, []):
            if d <= start_date:
                continue
            if p > peak:
                peak = p
            if d == trigger_dt:
                trigger_price = p
                break
        drawdown = (trigger_price - peak) / peak if peak else 0
        ret_from_first = (trigger_price - first_entry) / first_entry if first_entry else 0
        enriched.append({
            "날짜": start_date,
            "손절일": trigger_dt,
            "종목코드": code,
            "최초편입가": round(first_entry, 0),
            "고점": round(peak, 0),
            "손절가": round(trigger_price, 0),
            "고점대비_%": round(drawdown * 100, 2),
            "최초편입가대비_%": round(ret_from_first * 100, 2),
            "기준": basis,
            "비중_%": round(weight * 100, 2),
        })
    return enriched


def run():
    from lib.data import run_regime_combo_backtest
    from lib.db import get_conn
    import step7_backtest as bt

    bull_key = "수정전략_코스피_cap30%_top30_tx30bp_월간"
    bear_key = "cap30%_손절율15%(고점)"

    _orig_calc_plain = bt.calc_portfolio_return
    _orig_calc_sl = bt.calc_portfolio_return_with_stoploss

    rows = {"off": [], "nat": []}
    # 2026 진단용 (모든 리밸런싱 기간의 편입 종목 + 기간 내 max drawdown)
    diag_2026 = []

    def _collect_2026(conn, stocks, start_date, end_date, regime_label):
        if not start_date.startswith("2026"):
            return
        if not stocks:
            diag_2026.append({"날짜": start_date, "레짐": regime_label, "편입수": 0,
                              "최대고점대비_%": None, "최악종목": None})
            return
        codes = [c for c, _ in stocks]
        ph = ",".join(["?"] * len(codes))
        start_rows = conn.execute(f"""
            SELECT dp.stock_code, dp.adj_close
            FROM daily_price dp
            INNER JOIN (
                SELECT stock_code, MIN(trade_date) as d
                FROM daily_price WHERE stock_code IN ({ph}) AND trade_date >= ?
                GROUP BY stock_code
            ) t ON dp.stock_code = t.stock_code AND dp.trade_date = t.d
        """, (*codes, start_date)).fetchall()
        entry = {r[0]: r[1] for r in start_rows}
        daily = conn.execute(f"""
            SELECT stock_code, trade_date, adj_close FROM daily_price
            WHERE stock_code IN ({ph}) AND trade_date >= ? AND trade_date <= ? AND adj_close > 0
            ORDER BY trade_date
        """, (*codes, start_date, end_date)).fetchall()
        worst_dd = 0.0
        worst_code = None
        per_code = {}
        for c, d, p in daily:
            per_code.setdefault(c, []).append(p)
        for c, prices in per_code.items():
            if not prices:
                continue
            peak = entry.get(c, prices[0])
            for p in prices:
                if p > peak:
                    peak = p
                dd = (p - peak) / peak if peak else 0
                if dd < worst_dd:
                    worst_dd = dd
                    worst_code = c
        diag_2026.append({
            "날짜": start_date, "레짐": regime_label, "편입수": len(stocks),
            "최대고점대비_%": round(worst_dd * 100, 2),
            "최악종목": worst_code,
        })

    # ── A) carry off ──
    def _calc_sl_off(conn, stocks, start_date, end_date, **kw):
        kw["carry_over"] = {}
        ret, evs, _ = _orig_calc_sl(conn, stocks, start_date, end_date, **kw)
        _collect_2026(conn, stocks, start_date, end_date, "Bear")
        rows["off"].extend(_enrich_events(conn, stocks, start_date, evs, {},
                                          kw.get("stop_loss_basis", "peak")))
        return ret, evs, {}

    def _calc_plain_off(conn, stocks, start_date, end_date):
        _collect_2026(conn, stocks, start_date, end_date, "Bull")
        return _orig_calc_plain(conn, stocks, start_date, end_date)

    bt.calc_portfolio_return_with_stoploss = _calc_sl_off
    bt.calc_portfolio_return = _calc_plain_off
    print("\n[1/2] A) carry off 실행")
    try:
        run_regime_combo_backtest(
            bull_key=bull_key, bear_key=bear_key,
            universe="KOSPI", rebal_type="monthly", regime_mode="ai",
        )
    finally:
        bt.calc_portfolio_return = _orig_calc_plain
        bt.calc_portfolio_return_with_stoploss = _orig_calc_sl

    # ── C) carry over 자연 설계 ──
    _shared_carry = [{}]
    diag_2026.clear()  # A 모드에서 채운 거 제거 (C 한 번만 기록)

    def _calc_plain_nat(conn, stocks, start_date, end_date):
        _collect_2026(conn, stocks, start_date, end_date, "Bull")
        ret, _evs, new_carry = _orig_calc_sl(
            conn, stocks, start_date, end_date,
            stop_loss_pct=9999, stop_loss_mode="sell", stop_loss_basis="peak",
            carry_over=_shared_carry[0],
        )
        _shared_carry[0] = new_carry
        return ret

    def _calc_sl_nat(conn, stocks, start_date, end_date, **kw):
        prev_carry = dict(_shared_carry[0])  # enrich용 스냅샷
        kw["carry_over"] = _shared_carry[0]
        ret, evs, new_carry = _orig_calc_sl(conn, stocks, start_date, end_date, **kw)
        _shared_carry[0] = new_carry
        _collect_2026(conn, stocks, start_date, end_date, "Bear")
        rows["nat"].extend(_enrich_events(conn, stocks, start_date, evs, prev_carry,
                                          kw.get("stop_loss_basis", "peak")))
        return ret, evs, new_carry

    _shared_carry[0] = {}
    bt.calc_portfolio_return = _calc_plain_nat
    bt.calc_portfolio_return_with_stoploss = _calc_sl_nat
    print("\n[2/2] C) carry over 자연 설계 실행")
    try:
        run_regime_combo_backtest(
            bull_key=bull_key, bear_key=bear_key,
            universe="KOSPI", rebal_type="monthly", regime_mode="ai",
        )
    finally:
        bt.calc_portfolio_return = _orig_calc_plain
        bt.calc_portfolio_return_with_stoploss = _orig_calc_sl

    # 종목명 매핑
    code_to_name_path = Path(__file__).parent.parent / "data" / "code_to_name.json"
    code_to_name = {}
    if code_to_name_path.exists():
        with open(code_to_name_path, encoding="utf-8") as f:
            code_to_name = json.load(f)

    def add_name(records):
        for r in records:
            r["종목명"] = code_to_name.get(r["종목코드"], r["종목코드"])
        return records

    rows_off = add_name(rows["off"])
    rows_nat = add_name(rows["nat"])
    for r in diag_2026:
        r["최악종목명"] = code_to_name.get(r.get("최악종목"), r.get("최악종목"))

    cols = ["날짜", "손절일", "종목코드", "종목명", "최초편입가", "고점", "손절가",
            "고점대비_%", "최초편입가대비_%", "기준", "비중_%"]
    df_off = pd.DataFrame(rows_off, columns=cols) if rows_off else pd.DataFrame(columns=cols)
    df_nat = pd.DataFrame(rows_nat, columns=cols) if rows_nat else pd.DataFrame(columns=cols)
    df_2026 = pd.DataFrame(diag_2026,
                           columns=["날짜", "레짐", "편입수", "최대고점대비_%", "최악종목", "최악종목명"])

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"stoploss_events_{stamp}.xlsx"

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_off.to_excel(writer, sheet_name="A_off", index=False)
        df_nat.to_excel(writer, sheet_name="C_natural", index=False)
        df_2026.to_excel(writer, sheet_name="2026_진단", index=False)
        for sheet_name, df in [("A_off", df_off), ("C_natural", df_nat), ("2026_진단", df_2026)]:
            ws = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns):
                width = max(len(str(col)), df[col].astype(str).map(len).max() if len(df) else 0) + 2
                ws.set_column(i, i, min(width, 30))

    print(f"\n💾 저장: {out_path.relative_to(Path(__file__).parent.parent)}")
    print(f"  A_off:     {len(df_off)}건")
    print(f"  C_natural: {len(df_nat)}건")
    print(f"  2026 진단: {len(df_2026)}개월")


if __name__ == "__main__":
    run()
