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
from lib.ai_stock_filter import search_stock_news, run_news_ranking_agent


def _build_stock_info(conn, stocks: list[tuple[str, float]]) -> dict:
    result = {}
    for code, factor_score in stocks:
        code_a = f"A{code}" if not code.startswith("A") else code
        df = read_sql("SELECT stock_name FROM fnspace_master WHERE stock_code = ? LIMIT 1", conn, params=(code_a,))
        name = df.iloc[0]["stock_name"] if not df.empty else code
        result[code] = {"name": name, "factor_score": factor_score}
    return result


def _fetch_news(tech_data: dict, calc_date: str) -> dict:
    news = {}
    total = len(tech_data)
    for i, (code, info) in enumerate(tech_data.items(), 1):
        print(f"[{i}/{total}] {info['name']} ({code}) 뉴스 검색 중...", flush=True)
        news[code] = search_stock_news(info["name"], code, calc_date)
        print(news[code], flush=True)
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
            news_data = _fetch_news(tech_data, date)
            result = run_news_ranking_agent(tech_data, news_data)

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

            out = out_dir / f"news_rank_{date}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n저장: {out}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
