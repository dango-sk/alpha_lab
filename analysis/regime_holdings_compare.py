"""
V20(강세장) → A0(약세장) 레짐 전환 시 포트폴리오 종목 비교

레짐 전환이 실제로 일어나는 시점을 찾아서,
전환 전(V20) / 전환 후(A0) 포트폴리오 30종목을 나란히 보여줌.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config.settings import BACKTEST_CONFIG
from lib.db import get_conn
from lib.factor_engine import (
    code_to_module, score_stocks_from_strategy,
    prefetch_all_data, clear_prefetch_cache,
)
from step7_backtest import get_db, get_rebalance_dates, get_universe_stocks


def get_strategy_code_from_db(name):
    import psycopg2, os
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SELECT strategy_code FROM alpha_lab.backtest_cache WHERE name = %s LIMIT 1", (name,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def get_stock_names(conn, codes):
    """종목코드 → 종목명 매핑."""
    if not codes:
        return {}
    placeholders = ",".join(["%s"] * len(codes))
    rows = conn.execute(
        f"SELECT DISTINCT stock_code, stock_name FROM fnspace_master WHERE stock_code IN ({placeholders})",
        tuple(codes)
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def main():
    ma_window = 50

    # ── 전략 코드 로드 ──
    from lib.factor_engine import DEFAULT_STRATEGY_CODE
    v20_code = get_strategy_code_from_db("v20 R30 G30 D20 W30")
    a0_code = DEFAULT_STRATEGY_CODE

    if not v20_code:
        print("V20 전략을 DB에서 찾을 수 없습니다.")
        return

    v20_module = code_to_module(v20_code)
    a0_module = code_to_module(a0_code)

    # ── 리밸런싱 날짜 ──
    conn = get_db()
    rb_dates = get_rebalance_dates(conn, "monthly")
    conn.close()

    # ── 레짐 신호 계산 ──
    _conn_regime = get_conn()
    regimes = []
    for d in rb_dates:
        rows = _conn_regime.execute(
            "SELECT close FROM daily_price WHERE stock_code = '069500' "
            f"AND trade_date <= %s ORDER BY trade_date DESC LIMIT {ma_window + 1}",
            (d,)
        ).fetchall()
        prices = [r[0] for r in rows if r[0]]
        if len(prices) < ma_window + 1:
            regimes.append("Bull")
        else:
            current = prices[0]
            ma = float(np.mean(prices[1:ma_window + 1]))
            regimes.append("Bull" if current >= ma else "Bear")

    # ── 레짐 전환 시점 찾기 ──
    transitions = []
    for i in range(1, len(regimes)):
        if regimes[i - 1] != regimes[i]:
            transitions.append({
                "idx": i,
                "date": rb_dates[i],
                "prev_date": rb_dates[i - 1],
                "from": regimes[i - 1],
                "to": regimes[i],
            })

    print(f"기간: {rb_dates[0]} ~ {rb_dates[-1]} ({len(rb_dates)}개 리밸런싱)")
    print(f"레짐 전환: {len(transitions)}회\n")

    # 전환 목록 출력
    for t in transitions:
        arrow = "🐂→🐻" if t["from"] == "Bull" else "🐻→🐂"
        print(f"  {t['prev_date']} → {t['date']}  {arrow} ({t['from']}→{t['to']})")

    # ── 데이터 프리페치 ──
    print("\n팩터 데이터 로딩 중...")
    pf_conn = get_db()
    prefetch_all_data(pf_conn)
    pf_conn.close()

    # ── Bull→Bear 전환 시점에서 V20 vs A0 포트폴리오 비교 ──
    bull_to_bear = [t for t in transitions if t["from"] == "Bull"]
    bear_to_bull = [t for t in transitions if t["from"] == "Bear"]

    conn = get_db()

    for label, trans_list, before_strategy, before_name, after_strategy, after_name in [
        ("Bull→Bear (V20→A0)", bull_to_bear, v20_module, "V20", a0_module, "A0"),
        ("Bear→Bull (A0→V20)", bear_to_bull, a0_module, "A0", v20_module, "V20"),
    ]:
        print(f"\n{'='*90}")
        print(f"▶ {label} 전환 시점 포트폴리오 비교")
        print(f"{'='*90}")

        for t in trans_list:
            calc_date = t["date"]
            prev_date = t["prev_date"]

            print(f"\n{'─'*90}")
            print(f"전환일: {prev_date} ({before_name}) → {calc_date} ({after_name})")
            print(f"{'─'*90}")

            universe_set = get_universe_stocks(conn, calc_date)

            # 전환 전 전략 (이전 리밸런싱 시점 기준)
            before_candidates = score_stocks_from_strategy(conn, calc_date, before_strategy)
            before_top30 = [(c, s) for c, s in before_candidates if c in universe_set][:30]

            # 전환 후 전략
            after_candidates = score_stocks_from_strategy(conn, calc_date, after_strategy)
            after_top30 = [(c, s) for c, s in after_candidates if c in universe_set][:30]

            before_codes = set(c for c, _ in before_top30)
            after_codes = set(c for c, _ in after_top30)

            # 종목명 가져오기
            all_codes = before_codes | after_codes
            names = get_stock_names(conn, list(all_codes))

            overlap = before_codes & after_codes
            only_before = before_codes - after_codes
            only_after = after_codes - before_codes

            print(f"\n  겹치는 종목: {len(overlap)}개 / 30개 ({len(overlap)/30:.0%})")
            print(f"  {before_name}에만 있는 종목: {len(only_before)}개")
            print(f"  {after_name}에만 있는 종목: {len(only_after)}개")
            print(f"  → 예상 턴오버: ~{len(only_before)/30:.0%}")

            # 나란히 출력
            print(f"\n  {'#':>3}  {'[' + before_name + '] 종목':<30} {'점수':>6}  →  {'[' + after_name + '] 종목':<30} {'점수':>6}  상태")
            print(f"  {'─'*85}")

            for i in range(30):
                b_code, b_score = before_top30[i] if i < len(before_top30) else ("", 0)
                a_code, a_score = after_top30[i] if i < len(after_top30) else ("", 0)

                b_name = names.get(b_code, b_code)[:14]
                a_name = names.get(a_code, a_code)[:14]

                # 상태 표시
                if a_code in before_codes:
                    status = "유지"
                else:
                    status = "신규"

                b_in_after = "유지" if b_code in after_codes else "제거"

                print(f"  {i+1:>3}  {b_name:<14} ({b_code}) {b_score:>6.2f}  →  "
                      f"{a_name:<14} ({a_code}) {a_score:>6.2f}  {status}")

            # 제거/신규 종목 요약
            print(f"\n  [제거 종목] ({len(only_before)}개):")
            removed = [(c, s) for c, s in before_top30 if c in only_before]
            for c, s in removed:
                print(f"    - {names.get(c, c):<14} ({c}) 점수 {s:.2f}")

            print(f"\n  [신규 편입] ({len(only_after)}개):")
            added = [(c, s) for c, s in after_top30 if c in only_after]
            for c, s in added:
                print(f"    + {names.get(c, c):<14} ({c}) 점수 {s:.2f}")

    conn.close()
    _conn_regime.close()
    clear_prefetch_cache()
    print("\n완료!")


if __name__ == "__main__":
    main()
