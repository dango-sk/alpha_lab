"""
뉴스 에이전트로 30종목 점수/순위 매기기

실행: python analysis/news_rank.py
      python analysis/news_rank.py --date 2026-05-04
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from lib.db import get_conn, read_sql
from lib.ai_stock_filter2 import run_news_ranking_agent, search_stock_news


def _fetch_macro_context(conn, calc_date: str) -> str:
    """alpha_lab.news_nate에서 기준일 직전 1개월 매크로 뉴스 제목을 가져온다.
    시장 전체 배경(금리/환율/유가/지정학)으로 Claude에 공통 주입."""
    from datetime import datetime, timedelta
    calc_dt = datetime.strptime(calc_date, "%Y-%m-%d")
    start = (calc_dt - timedelta(days=30)).strftime("%Y-%m-%d")

    df = read_sql("""
        SELECT published_date, title
        FROM news_nate
        WHERE published_date >= ? AND published_date < ?
        ORDER BY published_date DESC
        LIMIT 60
    """, conn, params=(start, calc_date))

    if df.empty:
        return ""
    lines = [f"- [{r['published_date']}] {r['title']}" for _, r in df.iterrows()]
    return "\n".join(lines)


def _build_stock_info(conn, stocks: list[tuple[str, float]]) -> dict:
    result = {}
    for code, factor_score in stocks:
        code_a = f"A{code}" if not code.startswith("A") else code
        df = read_sql("SELECT stock_name FROM fnspace_master WHERE stock_code = ? LIMIT 1", conn, params=(code_a,))
        name = df.iloc[0]["stock_name"] if not df.empty else code
        result[code] = {"name": name, "factor_score": factor_score}
    return result


def _save_news_to_db(conn, calc_date: str, code: str, name: str,
                     factor_score: float, items: list):
    """Gemini로 새로 가져온 뉴스를 alpha_lab.news_google에 적재 (재사용 위해)."""
    if not items:
        return
    cur = conn.cursor()
    for it in items:
        fact = it.get("fact", "")
        if not fact:
            continue
        cur.execute("""
            INSERT INTO news_google (
                calc_date, stock_code, stock_name, factor_score,
                news_date, source_type, source_name, category,
                fact, figures, price_reaction
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (calc_date, stock_code, fact) DO NOTHING
        """, (calc_date, code, name, factor_score,
              it.get("date"), it.get("source_type"), it.get("source_name"),
              it.get("category"), fact, it.get("figures"), it.get("price_reaction")))
    conn.commit()


def _fetch_news_hybrid(conn, tech_data: dict, calc_date: str) -> dict:
    """종목별 뉴스 조회. DB(news_google)에 있으면 DB 사용,
    없으면 Gemini(search_stock_news)로 가져와 DB에 저장 후 사용."""
    news = {}
    total = len(tech_data)
    for i, (code, info) in enumerate(tech_data.items(), 1):
        df = read_sql("""
            SELECT news_date, source_type, source_name, category,
                   fact, figures, price_reaction
            FROM news_google
            WHERE calc_date = ? AND stock_code = ?
            ORDER BY news_date DESC
        """, conn, params=(calc_date, code))

        db_items = [{
            "date": r["news_date"], "source_type": r["source_type"],
            "source_name": r["source_name"], "category": r["category"],
            "fact": r["fact"], "figures": r["figures"],
            "price_reaction": r["price_reaction"],
        } for _, r in df.iterrows()]

        # DB 뉴스가 3건 이상이면 그대로 사용
        if len(db_items) >= 3:
            news[code] = json.dumps(
                {"stock_code": code, "stock_name": info["name"], "news_items": db_items},
                ensure_ascii=False, indent=2)
            print(f"[{i}/{total}] {info['name']} ({code}) DB 뉴스 {len(db_items)}건", flush=True)
        else:
            # DB 3건 미만 → Gemini로 보강 후 DB 저장, 기존 DB 뉴스와 합침
            print(f"[{i}/{total}] {info['name']} ({code}) DB {len(db_items)}건(부족) → Gemini 보강...", flush=True)
            raw = search_stock_news(info["name"], code, calc_date)
            try:
                parsed = json.loads(raw)
                gemini_items = parsed.get("news_items", [])
                _save_news_to_db(conn, calc_date, code, info["name"],
                                 info["factor_score"], gemini_items)
                # 기존 DB + Gemini 합치고 fact 기준 중복 제거
                merged, seen = [], set()
                for it in db_items + gemini_items:
                    f = it.get("fact", "")
                    if f and f not in seen:
                        seen.add(f)
                        merged.append(it)
                news[code] = json.dumps(
                    {"stock_code": code, "stock_name": info["name"], "news_items": merged},
                    ensure_ascii=False, indent=2)
                print(f"      → Gemini {len(gemini_items)}건 추가, 총 {len(merged)}건(DB 저장 완료)", flush=True)
            except (json.JSONDecodeError, TypeError):
                # Gemini 파싱 실패 → 기존 DB 뉴스라도 사용
                news[code] = json.dumps(
                    {"stock_code": code, "stock_name": info["name"], "news_items": db_items},
                    ensure_ascii=False, indent=2)
                print(f"      → Gemini 파싱 실패, DB {len(db_items)}건만 사용", flush=True)
    return news

STOCKS = {
    "2026-03-03": [
        ("161390", 92.5), ("078930", 90.0), ("081660", 90.0), ("073240", 90.0),
        ("000240", 87.5), ("007070", 87.5), ("294870", 87.5), ("069960", 86.2),
        ("192080", 86.2), ("111770", 85.0), ("000270", 83.8), ("000120", 83.8),
        ("001120", 83.8), ("009540", 83.7), ("000880", 82.5), ("011210", 81.2),
        ("006040", 81.2), ("028670", 80.0), ("009970", 78.8), ("298020", 78.8),
        ("097950", 78.7), ("375500", 78.7), ("032640", 77.5), ("005850", 77.5),
        ("402340", 76.2), ("001040", 76.2), ("030000", 75.0), ("282330", 75.0),
        ("007310", 75.0), ("000660", 73.8),
    ],
    "2026-04-01": [
        ("161390", 92.5), ("111770", 92.5), ("000240", 91.2), ("069960", 91.2),
        ("073240", 90.0), ("009540", 88.8), ("000880", 88.8), ("005850", 87.5),
        ("001800", 87.5), ("192080", 87.5), ("081660", 86.2), ("294870", 86.2),
        ("000270", 85.0), ("011210", 82.5), ("012630", 82.5), ("034220", 81.2),
        ("001120", 81.2), ("000660", 80.0), ("402340", 80.0), ("057050", 80.0),
        ("267250", 78.8), ("000120", 78.7), ("078930", 77.5), ("003550", 76.2),
        ("028670", 76.2), ("030000", 76.2), ("034230", 76.2), ("100840", 75.0),
        ("086280", 73.8), ("011070", 73.8),
    ],
    "2026-05-04": [
        ("161390", 95.0), ("111770", 93.8), ("078930", 92.5), ("001120", 90.0),
        ("005850", 87.5), ("294870", 87.5), ("069960", 86.2), ("006650", 86.2),
        ("000270", 85.0), ("030000", 85.0), ("012630", 85.0), ("073240", 85.0),
        ("009540", 83.8), ("007340", 83.7), ("267250", 82.5), ("032640", 82.5),
        ("028670", 82.5), ("192080", 82.5), ("081660", 81.2), ("383220", 80.0),
        ("000240", 80.0), ("034230", 80.0), ("086280", 78.8), ("282330", 78.8),
        ("001800", 78.8), ("000880", 77.5), ("000120", 77.5), ("005930", 76.2),
        ("402340", 76.2), ("021240", 76.2),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="특정 날짜만 실행 (예: 2026-05-04), 없으면 전체")
    args = parser.parse_args()

    dates = [args.date] if args.date else list(STOCKS.keys())
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    conn = get_conn()
    try:
        for date in dates:
            stocks = STOCKS.get(date)
            if not stocks:
                print(f"날짜 없음: {date}")
                continue

            print(f"\n[{date}] 뉴스 에이전트 실행 중... ({len(stocks)}종목)")

            tech_data = _build_stock_info(conn, stocks)
            news_data = _fetch_news_hybrid(conn, tech_data, date)
            macro_context = _fetch_macro_context(conn, date)
            print(f"  매크로 배경 뉴스 {macro_context.count(chr(10))+1 if macro_context else 0}건 포함")
            result = run_news_ranking_agent(tech_data, news_data, date, macro_context)

            rankings = result.get("rankings", [])
            print(f"\n{'='*70}")
            print(f"  {'순위':>4}  {'종목명':<18}  {'종목코드':>8}  {'뉴스점수':>8}  {'팩터점수':>8}  이유")
            print(f"  {'-'*65}")
            for item in rankings:
                rank   = item.get("rank", "")
                name   = item.get("stock_name", "")
                code   = item.get("stock_code", "")
                score  = item.get("score", "")
                fscore = item.get("factor_score")
                reason = item.get("reason", "")[:40]
                print(f"  {rank:>4}  {name:<18}  {code:>8}  {score:>8}  {fscore if fscore else '-':>8}  {reason}")
            print(f"  {'='*65}")

            out = out_dir / f"{date}_news_rank.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n저장: {out}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
