
"""
레짐 예측 빈도 실험 - 2026년 1~5월을 월 1회/2회/4회로 예측해 적중도 비교.

- 원본 regime_agent.py를 수정하지 않고 그대로 import해서 재사용한다.
- run_month(as_of)는 'as_of 전일까지' 데이터만 쓰므로 임의 시점 재현이 lookahead-safe.
- 시점: 매월 1·8·15·22일 (4회 기준). 1회=[1], 2회=[1,15], 4회=[1,8,15,22] 로 subset.
- 공정 비교: 과거 교훈 few-shot은 모든 시점에 '2026년 이전' 결과만 동일하게 주입.

[채점 방식 - reign(다음 예측까지) 기준]
각 예측은 '다음 예측 시점까지' 시장을 지배한다고 보고, 그 reign 기간의 KOSPI 수익률로 채점.
빈도마다 reign 길이가 다름: 1회≈한달, 2회≈보름, 4회≈일주일.
  - 부호 적중   : 예상수익률 부호 == reign 실제수익률 부호
  - 레짐 적중   : 예측 Bull/Bear == 실제 Bull/Bear
                  (둘 다 히스테리시스 적용: 직전 Bull→ val<=-2면 Bear / 직전 Bear→ val>=+1이면 Bull)

실행:
  python analysis/regime_freq_test.py            # 예측 생성(완료분 스킵)
  python analysis/regime_freq_test.py --score    # 빈도 비교표 출력
"""
import json
import argparse
import bisect
from pathlib import Path
from datetime import date

DAYS = [1, 8, 15, 22]                 # 4회 기준
FREQ = {"1회": [1], "2회": [1, 15], "4회": [1, 8, 15, 22]}
BULL_TH, BEAR_TH = 1.0, -2.0          # data.py와 동일: er>=+1 Bull복귀 / er<=-2 Bear전환

# 파일 경로 (1~5월 / 전체기간 분리)
_R5 = Path(__file__).parent / "regime_freq_test_results.json"
_S5 = Path(__file__).parent / "regime_freq_test_score.json"
_RFULL = Path(__file__).parent / "regime_freq_test_full_results.json"
_SFULL = Path(__file__).parent / "regime_freq_test_full_score.json"

# 모드별 설정 (main의 _setup에서 채움)
YM = []                # [(year, month), ...]
TERMINAL = "2026-06-01"  # 마지막 reign 종료 경계(다음달 시작)
OUT = _R5
SCORE_OUT = _S5


def _year_months(start, end):
    """(y,m) start~end 포함 리스트."""
    y, m = start
    out = []
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _setup(full):
    global YM, TERMINAL, OUT, SCORE_OUT
    if full:
        YM = _year_months((2018, 4), (2026, 5))   # 백테스트 전체기간 98개월
        TERMINAL, OUT, SCORE_OUT = "2026-06-01", _RFULL, _SFULL
    else:
        YM = _year_months((2026, 1), (2026, 5))    # 기존 1~5월
        TERMINAL, OUT, SCORE_OUT = "2026-06-01", _R5, _S5


# ── 1단계: 예측 생성 ──────────────────────────────────────────
def _load_lesson_base():
    """과거 교훈 후보 풀 = regime_agent_results.json 전체(2018~2026).
    build_lessons가 각 시점마다 'as_of 이전'만 골라 쓰므로(lookahead 방지),
    전체를 넘기면 시점별로 직전까지의 교훈이 누적 적용된다 (기존 백테스트 방식)."""
    p = Path(__file__).parent / "regime_agent_results.json"
    if not p.exists():
        return []
    return json.load(open(p, encoding="utf-8"))


def generate():
    import regime_agent as ra  # 여기서만 import (DB연결 + 데이터 프리로드 발생)
    # 빈도실험: 뉴스 조회를 '직전 월 통째' → 'as_of 전일 기준 최근 N일'로 교체.
    # run_news_agent가 모듈 전역 get_news_summary를 참조하므로 여기서 바꿔치기하면 됨.
    ra.get_news_summary = ra.get_news_summary2  # 원본 파일 로직은 그대로, 실행시에만 교체
    base = _load_lesson_base()
    print(f"[freq-test] 과거 교훈 후보 풀: {len(base)}건 (전체기간 누적, 같은 달 제외)")

    done = {}
    if OUT.exists():
        for r in json.load(open(OUT, encoding="utf-8")):
            done[r["as_of"]] = r
    results = list(done.values())

    targets = [date(y, m, d) for (y, m) in YM for d in DAYS]
    for as_of in targets:
        key = str(as_of)
        if key in done and "error" not in done[key]:
            print(f"[freq-test] skip {key} (이미 완료)")
            continue
        try:
            # 같은 달(YYYY-MM) 판정은 교훈에서 제외 (그 달 결과는 아직 미실현 → lookahead 누수 방지)
            cur_month = key[:7]
            base_excl = [x for x in base if x.get("as_of", "")[:7] != cur_month]
            r = ra.run_month(as_of, past_results=base_excl)
            slim = {k: r.get(k) for k in (
                "as_of", "expected_return", "judgment", "confidence",
                "summary", "kospi_next_month_return", "direction_correct")}
            results = [x for x in results if x["as_of"] != key]
            results.append(slim)
        except Exception as e:
            print(f"[freq-test] ERR {key}: {e}")
            try:
                ra.get_conn().rollback()
            except Exception:
                pass
            results = [x for x in results if x["as_of"] != key]
            results.append({"as_of": key, "error": str(e)})

        results.sort(key=lambda x: x["as_of"])
        json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        er = next((x.get("expected_return") for x in results if x["as_of"] == key), None)
        print(f"[freq-test] saved {key}  er={er}  ({len(results)}/{len(targets)})")

    print("[freq-test] DONE")


# ── 2단계: 채점 (reign 기준, 부호 + 레짐) ─────────────────────
def _load_prices():
    """KOSPI ETF(069500) 종가 시계열 로드 (채점용, 가벼운 단독 연결)."""
    import os
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SELECT trade_date::date, close FROM alpha_lab.daily_price "
                "WHERE stock_code='069500' ORDER BY trade_date")
    rows = cur.fetchall()
    conn.close()
    dts = [str(d) for d, _ in rows]
    px = {str(d): float(c) for d, c in rows}
    return dts, px


def _regime_seq(vals, init="Bull"):
    """히스테리시스(직전 레짐 유지) Bull/Bear 시퀀스. vals = er 또는 실제수익률 리스트."""
    out, prev = [], init
    for v in vals:
        if prev == "Bull":
            cur = "Bear" if v <= BEAR_TH else "Bull"
        else:
            cur = "Bull" if v >= BULL_TH else "Bear"
        out.append(cur)
        prev = cur
    return out


def score():
    preds = {p["as_of"]: p for p in json.load(open(OUT, encoding="utf-8"))
             if "error" not in p}
    dts, px = _load_prices()

    def p_onafter(d):
        i = bisect.bisect_left(dts, d)
        return px[dts[i]] if i < len(dts) else None

    def p_before(d):
        # d(다음 경계) 직전 거래일 종가 = 해당 구간의 마지막 거래일
        i = bisect.bisect_left(dts, d) - 1
        return px[dts[i]] if i >= 0 else None

    print(f"채점 = reign(다음 예측까지) | 레짐 임계값: Bull>={BULL_TH:+.0f}% / Bear<={BEAR_TH:+.0f}% (히스테리시스)\n")
    print(f"{'빈도':6} {'예측수':>5} {'부호 적중':>10} {'레짐 적중':>10}")

    out = {
        "method": "reign(다음 예측까지)",
        "bull_th": BULL_TH, "bear_th": BEAR_TH,
        "summary": {}, "detail": {},
    }
    detail_store = {}
    for fname, days in FREQ.items():
        sched = [f"{y}-{m:02d}-{d:02d}" for (y, m) in YM for d in days] + [TERMINAL]
        items = []  # (as_of, end, er, reign_ret)
        for i in range(len(sched) - 1):
            a, b = sched[i], sched[i + 1]
            if a not in preds:
                continue
            p0, p1 = p_onafter(a), p_before(b)   # 시작=예측일 이후 첫거래일, 종료=다음경계 직전 거래일
            if p0 is None or p1 is None:
                continue
            r = (p1 - p0) / p0 * 100
            items.append((a, b, preds[a]["expected_return"], r))

        ers = [it[2] for it in items]
        rets = [it[3] for it in items]
        pred_reg = _regime_seq(ers)     # 예측 레짐
        act_reg = _regime_seq(rets)     # 실제 레짐 (동일 히스테리시스)

        n = len(items)
        sign_h = sum(1 for _, _, er, r in items
                     if (er > 0 and r > 0) or (er < 0 and r < 0) or er == 0)
        reg_h = sum(1 for i in range(n) if pred_reg[i] == act_reg[i])
        print(f"{fname:6} {n:>5} {sign_h/n*100:>8.0f}% {reg_h/n*100:>9.0f}%")
        detail_store[fname] = list(zip(items, pred_reg, act_reg))

        out["summary"][fname] = {
            "n": n,
            "sign_hit": sign_h, "sign_acc": round(sign_h / n * 100, 1),
            "regime_hit": reg_h, "regime_acc": round(reg_h / n * 100, 1),
        }
        out["detail"][fname] = [
            {"as_of": a, "end": b, "er": er, "actual_ret": round(r, 2),
             "pred_regime": pr, "actual_regime": ac,
             "sign_ok": (er > 0 and r > 0) or (er < 0 and r < 0) or er == 0,
             "regime_ok": pr == ac}
            for (a, b, er, r), pr, ac in detail_store[fname]
        ]

    # 4회 세부
    print("\n[4회 세부]  예측일→종료      er    실제%    예측/실제레짐   부호  레짐")
    for (a, b, er, r), pr, ac in detail_store["4회"]:
        sg = (er > 0 and r > 0) or (er < 0 and r < 0) or er == 0
        rg = pr == ac
        print(f"  {a}→{b}  {er:+5.1f}  {r:+7.2f}%   {pr}/{ac:<4}    "
              f"{'O' if sg else 'X'}    {'O' if rg else 'X'}")

    json.dump(out, open(SCORE_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[freq-test] 채점 결과 저장: {SCORE_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="저장된 예측을 채점만 한다")
    ap.add_argument("--full", action="store_true", help="백테스트 전체기간(2018-04~2026-05) 대상")
    args = ap.parse_args()
    _setup(args.full)
    print(f"[freq-test] 모드: {'전체기간' if args.full else '2026 1~5월'} | 대상 {len(YM)}개월 × {len(DAYS)}시점 | OUT={OUT.name}")
    if args.score:
        score()
    else:
        generate()


if __name__ == "__main__":
    main()
