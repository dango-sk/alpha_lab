"""
네이트 뉴스 기반 매크로 뉴스 수집 → alpha_lab.news_nate
한국경제(hk) 날짜별 전체 기사에서 키워드 필터링 후 본문 수집

실행: python scripts/collect_news_nate.py --start 2018.05.01 --end 2021.11.30
"""

import os
import re
import time
import random
import argparse
import psycopg2
import requests
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / '.env')
DATABASE_URL = os.environ['DATABASE_URL']

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# 언론사 코드
MEDIA = ['']  # 빈 문자열 = 전체 언론사

# 매크로 키워드
KEYWORDS = [
    '증시', '코스피', 'KOSPI', '금리', '환율', 'ETF', '외국인', '재정', '통화정책',
    '중국 경제', '미국 경제', '연준', 'Fed', '유가', '반도체', '수출', '경기',
    '인플레이션', '자금', '매매 동향', '패시브', '기준금리',
    '연금', '배당', '상법', 'MSCI', '신흥국',
]


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alpha_lab.news_nate (
                id             SERIAL PRIMARY KEY,
                title          TEXT NOT NULL,
                source         TEXT,
                published_date TEXT,
                url            TEXT UNIQUE NOT NULL,
                summary        TEXT,
                published_time TEXT,
                category       TEXT DEFAULT 'macro',
                query          TEXT,
                crawled_at     TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_news_nate_date ON alpha_lab.news_nate (published_date)")
        conn.commit()
    print("✅ alpha_lab.news_nate 테이블 확인/생성 완료")


def fetch_body(url, max_chars=500):
    """기사 본문 수집 (마침표 기준 500자)"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        body = soup.select_one("#realArtcContents")
        if not body:
            return ""
        text = body.get_text(strip=True)
        if len(text) <= max_chars:
            return text
        # 마침표 기준으로 자르기
        cut = text[:max_chars + 100]
        last_period = cut.rfind(".", 0, max_chars + 50)
        if last_period > max_chars - 100:
            return cut[:last_period + 1]
        return text[:max_chars]
    except Exception:
        return ""


def get_articles_for_date(date_str, media_code):
    """네이트 뉴스에서 특정 날짜의 기사 목록 수집"""
    url = f"https://news.nate.com/mediaList?cp={media_code}&cate=&date={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")

        articles = []
        seen_urls = set()
        seen_titles = []
        for a in soup.select("a[href*='view']"):
            href = a.get("href", "")
            text = a.get_text(strip=True)
            if len(text) < 15 or "news.nate.com" not in href:
                continue
            full_url = "https:" + href if href.startswith("//") else href
            if full_url in seen_urls:
                continue
            # 키워드 필터링
            if any(k in text for k in KEYWORDS):
                title = text[:120]
                # 제목 유사도 체크 (0.7 이상이면 중복)
                if any(SequenceMatcher(None, title, t).ratio() > 0.7 for t in seen_titles):
                    continue
                seen_urls.add(full_url)
                seen_titles.append(title)
                articles.append({"title": title, "url": full_url})
        return articles
    except Exception as e:
        print(f"    오류: {e}")
        return []


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def crawl_date_range(start_date, end_date):
    """날짜 범위의 뉴스 수집 → DB 저장"""
    # 평일만 리스트로 생성
    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)

    total_inserted = 0
    conn = get_conn()

    for day in tqdm(days, desc="뉴스 수집", unit="일"):
        date_str = day.strftime("%Y%m%d")
        date_iso = day.strftime("%Y-%m-%d")

        day_count = 0
        for media in MEDIA:
            articles = get_articles_for_date(date_str, media)
            time.sleep(random.uniform(0.1, 0.2))

            for art in articles:
                body = fetch_body(art["url"])
                time.sleep(random.uniform(0.1, 0.2))

                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO alpha_lab.news_nate (title, source, published_date, url, summary, category)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (url) DO NOTHING
                        """, (art["title"], media, date_iso, art["url"], body, "macro"))
                        if cur.rowcount > 0:
                            day_count += 1
                        conn.commit()
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = get_conn()

        total_inserted += day_count
        if day_count > 0:
            tqdm.write(f"  {date_iso}: {day_count}건")

    conn.close()
    return total_inserted


def get_last_date(conn):
    with conn.cursor() as cur:
        cur.execute(r"""
            SELECT MAX(published_date) FROM alpha_lab.news_nate
            WHERE published_date ~ '^\d{4}-\d{2}-\d{2}$'
        """)
        row = cur.fetchone()
        return row[0] if row and row[0] else None


def run_daily():
    """정기 수집: DB 마지막 수집일 다음날 ~ 어제"""
    conn = get_conn()
    create_table(conn)
    last = get_last_date(conn)
    conn.close()

    yesterday = datetime.now() - timedelta(days=1)

    if last:
        start = datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
    else:
        # 최초 실행: 30일 전부터
        start = yesterday - timedelta(days=30)

    if start > yesterday:
        print(f"이미 {last}까지 수집 완료, 스킵")
        return 0

    print(f"수집 기간: {start.strftime('%Y-%m-%d')} ~ {yesterday.strftime('%Y-%m-%d')}")
    total = crawl_date_range(start, yesterday)

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM alpha_lab.news_nate")
        print(f"완료! 이번: {total}건, 전체: {cur.fetchone()[0]}건")
    conn.close()
    return total


def main():
    parser = argparse.ArgumentParser(description="네이트 뉴스 매크로 수집")
    parser.add_argument("--start", default=None, help="시작일 (YYYY.MM.DD)")
    parser.add_argument("--end", default=None, help="종료일 (YYYY.MM.DD)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    args = parser.parse_args()

    # --start/--end 없으면 정기 수집 모드
    if not args.start and not args.end:
        run_daily()
        return

    conn = get_conn()
    create_table(conn)

    start = datetime.strptime(args.start, "%Y.%m.%d")
    end = datetime.strptime(args.end, "%Y.%m.%d")

    if args.resume:
        last = get_last_date(conn)
        if last:
            resume_from = datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
            if resume_from <= end:
                start = resume_from
                print(f"Resume: {last} 이후부터 수집")
            else:
                print(f"이미 {args.end}까지 수집 완료")
                conn.close()
                return
    conn.close()

    print(f"수집 기간: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")
    print()

    total = crawl_date_range(start, end)

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM alpha_lab.news_nate")
        print(f"\n완료! 이번: {total}건, 전체: {cur.fetchone()[0]}건")
    conn.close()


if __name__ == "__main__":
    main()
