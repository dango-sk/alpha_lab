"""디베이트 라운드 R1 한 번 실호출 검증 (실데이터).

전략:
- 기존 cache/ai_filter_logs/ai_filter_2026-04-30.json (git 1cc233a) 의
  tech_result/news_result 를 R0 로 사용.
- 캐시된 user_prompt 안의 30종목 텍스트를 그대로 재활용하기 위해
  _build_tech_summary / _build_news_summary 를 monkey-patch.
- run_debate_round(round_num=1, ...) 실호출 (Claude API call 2회).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from lib import ai_stock_filter as af  # noqa: E402


def _extract_stock_summaries(user_prompt: str) -> str:
    """user_prompt 헤더 한 줄을 제거하고 ### 블록만 반환."""
    idx = user_prompt.find("###")
    body = user_prompt[idx:] if idx >= 0 else user_prompt
    body = re.split(r"\nJSON 형식으로만 응답해주세요", body)[0]
    return body.rstrip()


def main():
    log_path = ROOT / "cache" / "ai_filter_logs" / "ai_filter_2026-04-30.json"
    if log_path.stat().st_size == 0:
        raw = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", "1cc233a:cache/ai_filter_logs/ai_filter_2026-04-30.json"]
        )
        data = json.loads(raw)
    else:
        data = json.loads(log_path.read_text())

    prev_tech = data["tech_result"]
    prev_news = data["news_result"]

    tech_text = _extract_stock_summaries(prev_tech["_raw"]["user_prompt"])
    news_text = _extract_stock_summaries(prev_news["_raw"]["user_prompt"])

    af._build_tech_summary = lambda _td: tech_text
    af._build_news_summary = lambda _td, _nd: news_text

    print(f"[R0] tech Top10: {[i['stock_code'] for i in prev_tech['top_10']]}")
    print(f"[R0] news Top10: {[i['stock_code'] for i in prev_news['top_10']]}")
    r0_rate, r0_agreed = af._compute_agreement(prev_tech, prev_news)
    print(f"[R0] agreement={r0_rate:.2f} agreed={r0_agreed}")
    print()

    def on_token(agent: str, _full_text: str) -> None:
        print(f"  · {agent} 응답 수신")

    print("[R1] run_debate_round 호출...")
    r1 = af.run_debate_round(
        round_num=1,
        prev_tech_result=prev_tech,
        prev_news_result=prev_news,
        tech_data={},
        news_data={},
        on_token=on_token,
    )

    print()
    print(f"[R1] tech Top10: {[i.get('stock_code') for i in r1['tech'].get('top_10', [])]}")
    print(f"[R1] tech rebuttal: {r1['tech'].get('rebuttal','(없음)')[:200]}")
    print(f"[R1] news Top10: {[i.get('stock_code') for i in r1['news'].get('top_10', [])]}")
    print(f"[R1] news rebuttal: {r1['news'].get('rebuttal','(없음)')[:200]}")
    print(f"[R1] agreement={r1['agreement_rate']:.2f} agreed={r1['agreed_stocks']}")
    print(f"[R1] diff_from_prev: {r1['diff_from_prev']}")

    tech_top = r1["tech"].get("top_10", [])
    news_top = r1["news"].get("top_10", [])
    issues = []
    if len(tech_top) != 10:
        issues.append(f"tech Top10 길이 != 10 ({len(tech_top)})")
    if len(news_top) != 10:
        issues.append(f"news Top10 길이 != 10 ({len(news_top)})")
    if not r1["tech"].get("rebuttal"):
        issues.append("tech rebuttal 없음")
    if not r1["news"].get("rebuttal"):
        issues.append("news rebuttal 없음")
    if "error" in r1["tech"]:
        issues.append(f"tech JSON 파싱 실패: {r1['tech']}")
    if "error" in r1["news"]:
        issues.append(f"news JSON 파싱 실패: {r1['news']}")

    print()
    if issues:
        print("⚠ 검증 이슈:")
        for x in issues:
            print(f"   - {x}")
        sys.exit(1)
    print("✓ R1 검증 통과")

    out_path = ROOT / "analysis" / "results" / "debate_R1_smoketest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(r1, ensure_ascii=False, indent=2))
    print(f"  결과 저장: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
