"""
시황 스터디 AI 모듈 — Claude web_search 기반 시황 브리핑, Q&A, 퀴즈/토론
"""
import json
import re
from datetime import date, datetime
from typing import Optional

from pathlib import Path

from config.settings import ANTHROPIC_API_KEY
from lib.ai import _get_client, is_ai_available

_CACHE_DIR = Path(__file__).parent.parent / "cache"
QUIZ_HISTORY_PATH = _CACHE_DIR / "quiz_history.json"
QA_HISTORY_PATH = _CACHE_DIR / "qa_history.json"
BRIEFING_PATH = _CACHE_DIR / "latest_briefing.json"

MODEL = "claude-sonnet-4-5-20250929"

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 10,
    "user_location": {
        "type": "approximate",
        "country": "KR",
    },
}

# ═══════════════════════════════════════════════════════
# 1. 시황 브리핑
# ═══════════════════════════════════════════════════════

_BRIEFING_SYSTEM = """당신은 자산운용사 리서치센터의 시니어 이코노미스트입니다.
매일 아침 트레이딩 데스크에 배포하는 시황 브리핑을 작성합니다.

웹 검색을 활용해 최신 시장 데이터와 뉴스를 수집하세요.

## 출력 형식 (반드시 준수)

먼저 ```json 블록으로 핵심 지표를 출력하고, 그 다음에 상세 분석을 마크다운으로 작성하세요.

```json
{
  "indicators": [
    {"category": "지수", "name": "S&P 500", "value": "5,123.45", "change": "+1.2%", "up": true},
    {"category": "지수", "name": "나스닥", "value": "16,234.56", "change": "-0.3%", "up": false},
    {"category": "지수", "name": "KOSPI", "value": "2,634.12", "change": "+0.8%", "up": true},
    {"category": "지수", "name": "KOSDAQ", "value": "845.23", "change": "+0.5%", "up": true},
    {"category": "금리", "name": "US 10Y", "value": "4.25%", "change": "+3bp", "up": true},
    {"category": "금리", "name": "US 2Y", "value": "3.95%", "change": "-2bp", "up": false},
    {"category": "금리", "name": "KR 3Y", "value": "2.85%", "change": "+1bp", "up": true},
    {"category": "환율", "name": "USD/KRW", "value": "1,345.2", "change": "+0.5%", "up": true},
    {"category": "환율", "name": "DXY", "value": "104.5", "change": "+0.2%", "up": true},
    {"category": "원자재", "name": "WTI", "value": "$78.5", "change": "-1.2%", "up": false},
    {"category": "원자재", "name": "금", "value": "$2,145", "change": "+0.8%", "up": true},
    {"category": "크립토", "name": "BTC", "value": "$95,000", "change": "+2.1%", "up": true},
    {"category": "크립토", "name": "ETH", "value": "$3,200", "change": "+1.5%", "up": true}
  ],
  "key_points": [
    "핵심 포인트 1",
    "핵심 포인트 2",
    "핵심 포인트 3"
  ],
  "sentiment": "cautious"
}
```

sentiment는 "risk_on", "cautious", "risk_off" 중 하나.
indicators에는 위 예시를 참고하되 실제 최신 수치를 넣으세요. 주요 지표를 빠짐없이 포함.

## 상세 분석 (JSON 블록 다음에 마크다운으로)

아래 섹션별로 작성. 각 섹션은 ## 헤더로 구분하세요.

## 글로벌 지수
각 지수의 움직임과 원인 분석. 어제 장 마감 기준.

## 금리/채권
미국, 한국 금리 동향. 스프레드 변화. 금리 움직임의 배경.

## 환율
원/달러, DXY, 엔/달러 등. 환율 변동 요인.

## 원자재
유가, 금, 구리 등. 수급 요인.

## 크립토
비트코인, 이더리움 등 주요 암호화폐 동향. 가격 변동 원인, 규제/ETF 이슈, 온체인 지표 등.

## 중앙은행/정책
연준, ECB, BOJ, 한은 관련 최신 동향.

## 한국 장 전망/영향
(오전 브리핑 시 필수) 간밤 글로벌 시장이 오늘 한국 장에 미칠 영향:
- 예상 갭 방향 (상승/하락/보합 출발)
- 수혜 섹터와 피해 섹터
- 외국인/기관 수급 방향 예상
- 주목할 종목군이나 테마
(오후/장마감 후에는 실제 한국 장 결과와 글로벌 요인의 연결 분석)

## 글로벌 매크로
각국 경제 사이클과 자산 간 연결고리를 다루세요:
- 미국/유럽/중국/일본 경제 흐름 (성장률, PMI, 고용 등 최신 지표)
- 각국 중앙은행 정책 방향 비교 (금리 차이, 통화정책 디커플링)
- 글로벌 채권 시장 (미국채/독일 분트/일본 JGB 금리 동향, 크레딧 스프레드)
- 대체투자 시그널 (부동산, 원자재, 금 등이 말해주는 매크로 방향)
- 자산 간 상관관계 변화 (주식-채권 상관관계, 달러와 위험자산 관계 등)

## 핵심 이슈
오늘 시장을 움직인 주요 이벤트 2-3개. 각 이슈에 대해:
- 무슨 일이 있었는가
- 시장에 어떤 영향을 미쳤는가
- 앞으로의 시사점

규칙:
- 수치(전일 대비 변동 포함)를 반드시 포함
- '왜 이렇게 움직였는가' 해석 포함
- 한국어로 작성
"""


def _parse_briefing(raw_text: str) -> dict:
    """브리핑 텍스트에서 JSON 지표 + 마크다운 섹션 파싱"""
    result = {
        "indicators": [],
        "key_points": [],
        "sentiment": "cautious",
        "sections": {},
        "full_text": raw_text,
    }

    # JSON 블록 추출
    json_match = re.search(r"```json\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            result["indicators"] = data.get("indicators", [])
            result["key_points"] = data.get("key_points", [])
            result["sentiment"] = data.get("sentiment", "cautious")
        except json.JSONDecodeError:
            pass

    # 마크다운 섹션 파싱 (## 헤더 기준)
    # JSON 블록 이후의 텍스트만
    md_text = raw_text
    if json_match:
        md_text = raw_text[json_match.end():]

    section_pattern = r"##\s+(.+?)\n(.*?)(?=\n##\s+|\Z)"
    for match in re.finditer(section_pattern, md_text, re.DOTALL):
        title = match.group(1).strip()
        content = match.group(2).strip()
        if content:
            result["sections"][title] = content

    return result


def generate_briefing() -> Optional[dict]:
    """오늘의 시황 브리핑 생성 (web_search 활용). 구조화된 dict 반환."""
    client = _get_client()
    if not client:
        return None

    today = date.today().strftime("%Y년 %m월 %d일")
    hour = datetime.now().hour

    if hour < 12:
        # 오전: 간밤 미국 장 마감 중심 + 한국 장 영향 분석
        time_context = (
            "현재 한국 시간 오전입니다. "
            "간밤 미국 장 마감 결과를 중심으로 브리핑해주세요. "
            "미국 주요 지수 종가, 주요 이벤트를 다루고, "
            "반드시 '오늘 한국 장 전망' 섹션을 포함하세요: "
            "미국 장 결과가 오늘 한국 장에 어떤 영향을 줄지 구체적으로 분석해주세요. "
            "수혜/피해 섹터, 예상 갭(상승/하락 출발), 외국인 수급 방향, 주목할 종목군을 포함."
        )
    elif hour < 17:
        # 오후 장중: 한국 장 진행 상황 + 글로벌 동향
        time_context = (
            "현재 한국 시간 오후입니다. "
            "한국 장 진행 상황과 글로벌 시장 동향을 함께 브리핑해주세요."
        )
    else:
        # 17시 이후: 한국 장 마감 중심 + 미국 프리마켓/선물
        time_context = (
            "현재 한국 시간 장 마감 이후입니다. "
            "오늘 한국 장 마감 결과를 중심으로 브리핑해주세요. "
            "KOSPI/KOSDAQ 종가, 주요 섹터 움직임, 외국인/기관 수급을 우선 다루고, "
            "미국 선물/프리마켓 동향, 오늘 밤 미국 장 주요 이벤트 및 예정된 경제지표도 함께 다뤄주세요."
        )

    user_msg = (
        f"오늘은 {today}입니다. {time_context}\n\n"
        "웹 검색 시 다음 키워드들을 활용하세요:\n"
        "- 'KOSPI 오늘', 'KOSDAQ 오늘', '코스피 장마감'\n"
        "- 'S&P 500 today', 'nasdaq today'\n"
        "- '원달러 환율 오늘', 'US 10Y treasury yield'\n"
        "- 'WTI crude oil price', '금 시세 오늘'\n"
        "- 'bitcoin price today'\n"
        "- 'investing.com', 'finance.naver.com'\n\n"
        "오늘의 글로벌 금융시장 시황 브리핑을 작성해주세요."
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=_BRIEFING_SYSTEM,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": user_msg}],
        )
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        raw = "\n".join(text_parts) if text_parts else ""
        if not raw:
            return None
        return _parse_briefing(raw)
    except Exception as e:
        return {"error": str(e), "indicators": [], "key_points": [], "sentiment": "cautious", "sections": {}, "full_text": ""}


def get_briefing_text(briefing) -> str:
    """Q&A/퀴즈용 브리핑 텍스트 추출 (dict 또는 레거시 str 모두 지원)"""
    if not briefing:
        return ""
    if isinstance(briefing, str):
        return briefing
    return briefing.get("full_text", "")


# ═══════════════════════════════════════════════════════
# 2. Q&A 모드
# ═══════════════════════════════════════════════════════

_QA_SYSTEM_TEMPLATE = """당신은 자산운용사의 시니어 이코노미스트 AI 어시스턴트입니다.
아래 오늘의 시황 브리핑을 참고하여 사용자의 질문에 답합니다.

필요하면 웹 검색을 추가로 활용하여 최신 정보를 보완하세요.

규칙:
- 한국어로 답변합니다.
- 전문 금융 용어를 사용하되 쉽게 설명합니다.
- 수치와 근거를 포함합니다.
- 답변은 핵심적이고 구조적으로.

## 오늘의 시황 브리핑
{briefing}
"""


def chat_qa(messages: list, briefing_text: str) -> str:
    """Q&A 모드 채팅"""
    client = _get_client()
    if not client:
        return "API 키가 설정되지 않았습니다."

    system = _QA_SYSTEM_TEMPLATE.format(briefing=briefing_text or "(아직 생성되지 않음)")
    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=system,
            tools=[WEB_SEARCH_TOOL],
            messages=api_messages,
        )
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts) if text_parts else "응답 생성에 실패했습니다."
    except Exception as e:
        return f"오류: {e}"


# ═══════════════════════════════════════════════════════
# 3. 퀴즈/토론 모드
# ═══════════════════════════════════════════════════════

DIFFICULTY_LEVELS = {
    "기초": {
        "label": "기초",
        "description": "시황 사실 확인, 기본 개념 설명",
        "prompt_hint": "금융 시장의 기본 개념과 오늘 시황의 사실 관계를 확인하는 수준. 용어 정의나 단순 인과관계를 묻는 질문.",
    },
    "심화": {
        "label": "심화",
        "description": "다자산 연계 분석, 매크로 논리 전개",
        "prompt_hint": "자산간 상관관계, 매크로 변수 간 인과관계, 정책 파급효과 등을 묻는 질문. 복수의 시장/변수를 연결하여 분석해야 함.",
    },
    "실전면접": {
        "label": "실전면접",
        "description": "AM/증권사 면접 수준의 전략 제안",
        "prompt_hint": "자산운용사 또는 증권사 면접에서 실제로 나오는 수준. 시황 해석 → 투자 전략 제안 → 리스크 관리까지 종합적 사고를 요구. '당신이 PM이라면 어떻게 하겠는가' 식의 질문.",
    },
}

# ── 글로벌 연결 객관식 퀴즈 ──

_GLOBAL_QUIZ_SYSTEM_TEMPLATE = """당신은 자산운용사 리서치센터의 시니어 이코노미스트입니다.
글로벌 매크로와 자산 간 연결고리를 테스트하는 객관식 문제를 출제합니다.

## 오늘의 시황
{briefing}
"""

_GLOBAL_QUIZ_GENERATE_PROMPT = """현재 시황을 바탕으로 글로벌 매크로 연결 객관식 문제를 1개 출제하세요.

자산 간, 국가 간, 정책 간 연결고리를 묻는 문제여야 합니다.

예시 유형:
- "미국 금리 인상 시 가장 큰 영향을 받는 자산은?"
- "중국 PMI 하락이 가장 직접적으로 영향을 미치는 원자재는?"
- "엔화 약세가 지속될 때 수혜를 받는 한국 섹터는?"
- "미국채 장단기 금리 역전이 시사하는 것은?"

반드시 아래 JSON 형식으로만 출력하세요:

```json
{
  "question": "문제 내용",
  "context": "이 문제와 관련된 오늘 시황 배경 1-2문장",
  "choices": ["A. 선택지1", "B. 선택지2", "C. 선택지3", "D. 선택지4"],
  "answer": "A",
  "explanation": "정답 해설 — 왜 이 답이 맞는지, 다른 선택지는 왜 틀린지 각각 설명. 실제 매크로 논리와 오늘 시황을 연결해서 설명."
}
```
"""


def generate_global_quiz(briefing_text: str) -> dict | None:
    """글로벌 연결 객관식 퀴즈 1문제 생성"""
    client = _get_client()
    if not client:
        return None

    system = _GLOBAL_QUIZ_SYSTEM_TEMPLATE.format(
        briefing=briefing_text or "(시황 브리핑 없음)",
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": _GLOBAL_QUIZ_GENERATE_PROMPT}],
        )
        raw = response.content[0].text
        json_match = re.search(r"```json\s*\n(.*?)\n```", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        return json.loads(raw)
    except Exception:
        return None


_QUIZ_SYSTEM_TEMPLATE = """당신은 자산운용사 면접관 역할의 시니어 CIO입니다.
지원자의 시황 분석 능력과 전략적 사고력을 평가합니다.

## 오늘의 시황
{briefing}

## 난이도: {difficulty}
{difficulty_hint}

## 역할
- 면접 질문을 출제할 때: 실제 AM/증권사 면접에서 나올 법한 현실적인 질문을 출제하세요.
- 답변을 채점할 때: 엄격하되 건설적으로 피드백하세요.
"""

_QUIZ_GENERATE_PROMPT = """현재 시황을 바탕으로 면접 질문 1개를 출제하세요.

형식:
**[질문]**
(질문 내용 — 서술형으로 답해야 하는 개방형 질문)

**[배경]**
(이 질문이 왜 중요한지, 어떤 역량을 평가하는지 1-2문장)

**[힌트]**
(답변 방향을 잡는 데 도움이 되는 키워드 2-3개)
"""

_QUIZ_GRADE_TEMPLATE = """지원자가 다음 질문에 답변했습니다. 채점해주세요.

**질문:** {question}

**지원자 답변:**
{answer}

다음 형식으로 채점하세요:

**[점수]** X/10

**[강점]**
- (잘한 부분)

**[보완점]**
- (부족한 부분, 구체적으로)

**[모범답안]**
(면접에서 높은 점수를 받을 수 있는 답변 예시 — 구조적으로 작성)

**[후속질문]**
(답변 내용을 기반으로 한 꼬리질문 1개 — 더 깊이 파고들기 위해)
"""


def generate_quiz(briefing_text: str, difficulty: str) -> str:
    """면접 스타일 퀴즈 출제"""
    client = _get_client()
    if not client:
        return "API 키가 설정되지 않았습니다."

    level = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["기초"])
    system = _QUIZ_SYSTEM_TEMPLATE.format(
        briefing=briefing_text or "(시황 브리핑 없음)",
        difficulty=level["label"],
        difficulty_hint=level["prompt_hint"],
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": _QUIZ_GENERATE_PROMPT}],
        )
        return response.content[0].text
    except Exception as e:
        return f"퀴즈 생성 실패: {e}"


def grade_answer(briefing_text: str, difficulty: str, question: str, answer: str) -> str:
    """서술형 답변 채점"""
    client = _get_client()
    if not client:
        return "API 키가 설정되지 않았습니다."

    level = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["기초"])
    system = _QUIZ_SYSTEM_TEMPLATE.format(
        briefing=briefing_text or "(시황 브리핑 없음)",
        difficulty=level["label"],
        difficulty_hint=level["prompt_hint"],
    )

    user_msg = _QUIZ_GRADE_TEMPLATE.format(question=question, answer=answer)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": user_msg}],
        )
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts) if text_parts else "채점 실패"
    except Exception as e:
        return f"채점 실패: {e}"


def chat_discussion(messages: list, briefing_text: str, difficulty: str) -> str:
    """토론 모드 — 퀴즈 채점 후 후속 토론 진행"""
    client = _get_client()
    if not client:
        return "API 키가 설정되지 않았습니다."

    level = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["기초"])
    system = _QUIZ_SYSTEM_TEMPLATE.format(
        briefing=briefing_text or "(시황 브리핑 없음)",
        difficulty=level["label"],
        difficulty_hint=level["prompt_hint"],
    ) + """

지원자와 토론 중입니다. 지원자의 답변에 대해:
- 동의할 점은 인정하되, 빈틈이 있으면 날카롭게 반론하세요.
- 추가 근거를 요구하거나 다른 관점을 제시하세요.
- 대화가 자연스럽게 심화되도록 이끄세요.
- 면접관의 톤을 유지하세요 (전문적이되 위압적이지 않게).
"""

    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=system,
            tools=[WEB_SEARCH_TOOL],
            messages=api_messages,
        )
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts) if text_parts else "응답 생성에 실패했습니다."
    except Exception as e:
        return f"오류: {e}"


# ═══════════════════════════════════════════════════════
# 4. 퀴즈 기록 저장/로드
# ═══════════════════════════════════════════════════════

def extract_score(grade_text: str) -> int | None:
    """채점 결과에서 점수 추출 (X/10 형식)"""
    m = re.search(r"\*?\*?\[?점수\]?\*?\*?\s*(\d+)\s*/\s*10", grade_text)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*/\s*10", grade_text)
    if m:
        return int(m.group(1))
    return None


def save_quiz_result(entry: dict):
    """퀴즈 결과를 파일에 추가 저장"""
    history = load_quiz_history()
    score = extract_score(entry.get("grade", ""))
    entry["score"] = score
    entry["date"] = date.today().isoformat()
    history.append(entry)

    QUIZ_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUIZ_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_quiz_history() -> list:
    """저장된 퀴즈 히스토리 로드"""
    if not QUIZ_HISTORY_PATH.exists():
        return []
    try:
        with open(QUIZ_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def get_wrong_questions(threshold: int = 6) -> list:
    """틀린 문제(점수 threshold 이하) 목록 반환"""
    history = load_quiz_history()
    return [
        h for h in history
        if h.get("score") is not None and h["score"] <= threshold
    ]


# ═══════════════════════════════════════════════════════
# 5. Q&A 대화 & 브리핑 영구 저장
# ═══════════════════════════════════════════════════════

def save_qa_history(messages: list):
    """Q&A 대화를 파일에 저장"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"date": date.today().isoformat(), "messages": messages}
    with open(QA_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_qa_history() -> list:
    """저장된 Q&A 대화 로드"""
    if not QA_HISTORY_PATH.exists():
        return []
    try:
        with open(QA_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("messages", [])
    except (json.JSONDecodeError, IOError):
        return []


def save_briefing(briefing):
    """브리핑 데이터를 파일에 저장"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"date": date.today().isoformat(), "briefing": briefing}
    with open(BRIEFING_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_briefing():
    """저장된 브리핑 로드 (당일 것만)"""
    if not BRIEFING_PATH.exists():
        return None
    try:
        with open(BRIEFING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == date.today().isoformat():
            return data.get("briefing")
        return None  # 어제 것이면 무시
    except (json.JSONDecodeError, IOError):
        return None


# ═══════════════════════════════════════════════════════
# 6. 종목 공부
# ═══════════════════════════════════════════════════════

STOCK_HISTORY_PATH = _CACHE_DIR / "stock_history.json"

_STOCK_ANALYSIS_SYSTEM = """당신은 자산운용사의 시니어 주식 애널리스트입니다.
개별 종목에 대해 깊이 있는 분석을 제공합니다.

웹 검색을 적극 활용하여 최신 정보를 수집하세요.

## 출력 형식 (반드시 준수)

먼저 ```json 블록으로 종목 요약을 출력하고, 그 다음에 상세 분석을 마크다운으로 작성하세요.

```json
{
  "stock_name": "종목명",
  "ticker": "티커/종목코드",
  "sector": "섹터",
  "market_cap": "시가총액",
  "current_price": "현재가",
  "change": "전일 대비 변동",
  "up": true,
  "pe_ratio": "PER",
  "pb_ratio": "PBR",
  "consensus": "투자의견 컨센서스 (매수/중립/매도)",
  "target_price": "목표주가 컨센서스"
}
```

## 상세 분석 (JSON 블록 다음에 마크다운으로)

## 사업 개요
회사가 뭘 하는 곳인지 핵심 사업 설명.

## 실적 및 밸류에이션
최근 실적 동향, PER/PBR/EV/EBITDA 등 밸류에이션 분석.

## 최근 이슈
최근 1-2주 주요 뉴스와 이벤트. 주가에 영향을 준 요인.

## 산업 동향
해당 섹터/산업의 현재 흐름과 전망.

## 투자 포인트
강점(Bull case)과 리스크(Bear case)를 균형 있게 서술.

규칙:
- 수치를 반드시 포함 (실적, 밸류에이션, 시가총액 등)
- 한국어로 작성
- 면접에서 이 종목에 대해 질문받았을 때 답할 수 있는 수준으로 작성
"""


_STOCK_PICKS_SYSTEM = """당신은 자산운용사의 시니어 포트폴리오 매니저입니다.
다음 주 월요일부터 일요일까지 매수 후 홀드했을 때 포트폴리오 전체 수익률 5% 이상을 목표로 하는 롱/숏 종목 조합을 추천합니다.

웹 검색을 적극 활용하여 최신 시장 동향, 실적 발표 일정, 매크로 이벤트, 수급 동향을 파악하세요.

## 시간 검증 (매우 중요)
- 오늘 날짜를 기준으로 이미 발생한 이벤트(실적 발표 완료, 지난 FOMC 등)와 아직 발생하지 않은 이벤트를 반드시 구분하세요.
- "실적 발표 예정"이라고 쓰려면 실제로 다음 주 이후에 발표 예정인지 웹 검색으로 반드시 확인하세요. 이미 발표된 실적을 "예정"이라고 쓰면 안 됩니다.
- 이미 발표된 실적은 "실적 발표 완료 (서프라이즈/미스)" 등으로 정확히 기술하고, 그 결과가 다음 주 주가에 미칠 영향을 분석하세요.
- 매크로 이벤트(금리 결정, 고용 지표 등)도 마찬가지로 이미 발표된 것과 예정된 것을 명확히 구분하세요.

## 출력 형식 (반드시 준수)

```json
{
  "picks": [
    {
      "name": "종목명",
      "ticker": "티커/종목코드",
      "market": "KR 또는 US",
      "direction": "LONG 또는 SHORT",
      "weight": "포트폴리오 비중 (%, 전체 합 100)",
      "current_price": "현재가",
      "target_price": "1주일 목표가",
      "expected_return": "기대 수익률 (%)",
      "category": "카테고리 (예: 실적 모멘텀, 매크로 수혜, 숏 스퀴즈, 밸류에이션 매력, 이벤트 드리븐 등)",
      "reason": "왜 이 종목을 이 방향으로 잡아야 하는지 구체적 근거 3~4문장. 카탈리스트, 수급, 기술적/펀더멘탈 근거를 명확히.",
      "risk": "주요 리스크 요인 1~2문장",
      "tags": ["관련 키워드1", "키워드2"]
    }
  ],
  "portfolio_summary": {
    "total_expected_return": "포트폴리오 전체 가중 기대수익률 (%)",
    "strategy_overview": "전체 전략 요약 2~3문장. 왜 이 조합이 5% 이상 수익을 낼 수 있는지.",
    "key_risks": "포트폴리오 전체 리스크 요인 1~2문장"
  }
}
```

## 추천 기준
- 한국 LONG 3~4개 + 한국 SHORT 1~2개 + 미국 LONG 2~3개 + 미국 SHORT 1~2개 = 총 8~10개
- 월요일 시초가 매수/매도 → 일요일(금요일 종가)까지 홀드 전제
- 포트폴리오 전체 가중 기대수익률이 5% 이상이 되도록 비중 조절
- 롱과 숏을 조합하여 시장 방향성 리스크를 일부 헤지
- 각 종목마다 다음 주에 특별히 움직일 카탈리스트(실적, 이벤트, 수급 등)가 있어야 함
- JSON 블록만 출력하세요 (추가 설명 불필요)
"""


def get_stock_picks() -> tuple:
    """다음 주 롱/숏 포트폴리오 추천. (picks, summary) 튜플 반환."""
    client = _get_client()
    if not client:
        return [], {}

    today = date.today().strftime("%Y년 %m월 %d일")

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=_STOCK_PICKS_SYSTEM,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": f"오늘은 {today}입니다. 다음 주 월요일부터 일요일까지 홀드할 롱/숏 포트폴리오 조합을 추천해주세요. 포트폴리오 전체 수익률 5% 이상을 목표로 해주세요."}],
        )
        text_parts = [b.text for b in response.content if b.type == "text"]
        raw = "\n".join(text_parts) if text_parts else ""
        json_match = re.search(r"```json\s*\n(.*?)\n```", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            return data.get("picks", []), data.get("portfolio_summary", {})
        data = json.loads(raw)
        return data.get("picks", []), data.get("portfolio_summary", {})
    except Exception:
        return [], {}


_WEEKLY_TRADES_SYSTEM = """당신은 자산운용사의 시니어 트레이딩 전략가입니다.
이번 주(월~금) 트레이딩 아이디어를 제시합니다.

웹 검색을 적극 활용하여 최신 시장 동향, 실적 발표 일정, 매크로 이벤트를 파악하세요.

## 출력 형식 (반드시 준수)

```json
{
  "week": "3/9(월) ~ 3/13(금)",
  "picks": [
    {
      "market": "KR",
      "direction": "LONG",
      "name": "종목명",
      "ticker": "티커/종목코드",
      "current_price": "현재가",
      "target": "목표가",
      "stop_loss": "손절가",
      "reason": "구체적 매수/매도 근거 3~4문장. 왜 이번 주에 이 방향인지, 카탈리스트가 뭔지, 기술적/펀더멘탈 근거를 명확히.",
      "risk": "주요 리스크 요인 1~2문장"
    },
    {
      "market": "KR",
      "direction": "SHORT",
      "name": "종목명",
      "ticker": "티커/종목코드",
      "current_price": "현재가",
      "target": "목표가",
      "stop_loss": "손절가",
      "reason": "구체적 매도 근거",
      "risk": "주요 리스크 요인"
    },
    {
      "market": "US",
      "direction": "LONG",
      "name": "종목명",
      "ticker": "티커",
      "current_price": "현재가",
      "target": "목표가",
      "stop_loss": "손절가",
      "reason": "구체적 매수 근거",
      "risk": "주요 리스크 요인"
    },
    {
      "market": "US",
      "direction": "SHORT",
      "name": "종목명",
      "ticker": "티커",
      "current_price": "현재가",
      "target": "목표가",
      "stop_loss": "손절가",
      "reason": "구체적 매도 근거",
      "risk": "주요 리스크 요인"
    }
  ],
  "weekly_events": ["이번 주 주요 매크로 이벤트 1", "이벤트 2", "이벤트 3"]
}
```

## 종목 선정 기준
- 한국(KR) 롱 1개 + 숏 1개, 미국(US) 롱 1개 + 숏 1개 = 총 4개
- 이번 주 구체적인 카탈리스트가 있는 종목 우선 (실적 발표, 규제 변화, 수급 변화 등)
- reason은 "왜 이번 주에"인지 시간적 근거를 반드시 포함
- 목표가와 손절가는 현실적인 수준으로 제시
- JSON 블록만 출력하세요
"""


def get_weekly_trades() -> dict | None:
    """이번 주 롱/숏 트레이딩 아이디어"""
    client = _get_client()
    if not client:
        return None

    today = date.today().strftime("%Y년 %m월 %d일")
    weekday = ["월", "화", "수", "목", "금", "토", "일"][date.today().weekday()]

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            system=_WEEKLY_TRADES_SYSTEM,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": f"오늘은 {today}({weekday})입니다. 이번 주 트레이딩 아이디어를 제시해주세요."}],
        )
        text_parts = [b.text for b in response.content if b.type == "text"]
        raw = "\n".join(text_parts) if text_parts else ""
        json_match = re.search(r"```json\s*\n(.*?)\n```", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        return json.loads(raw)
    except Exception:
        return None


def _parse_stock_analysis(raw_text: str) -> dict:
    """종목 분석 텍스트에서 JSON 요약 + 마크다운 섹션 파싱"""
    result = {
        "stock_info": {},
        "sections": {},
        "full_text": raw_text,
    }

    json_match = re.search(r"```json\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if json_match:
        try:
            result["stock_info"] = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    md_text = raw_text
    if json_match:
        md_text = raw_text[json_match.end():]

    section_pattern = r"##\s+(.+?)\n(.*?)(?=\n##\s+|\Z)"
    for match in re.finditer(section_pattern, md_text, re.DOTALL):
        title = match.group(1).strip()
        content = match.group(2).strip()
        if content:
            result["sections"][title] = content

    return result


def analyze_stock(stock_name: str) -> dict:
    """종목 분석 생성 (web_search 활용)"""
    client = _get_client()
    if not client:
        return {"error": "API 키가 설정되지 않았습니다."}

    today = date.today().strftime("%Y년 %m월 %d일")
    user_msg = f"오늘은 {today}입니다. '{stock_name}' 종목에 대해 상세 분석을 해주세요."

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=6000,
            system=_STOCK_ANALYSIS_SYSTEM,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": user_msg}],
        )
        text_parts = [b.text for b in response.content if b.type == "text"]
        raw = "\n".join(text_parts) if text_parts else ""
        if not raw:
            return {"error": "분석 생성 실패"}
        return _parse_stock_analysis(raw)
    except Exception as e:
        return {"error": str(e), "stock_info": {}, "sections": {}, "full_text": ""}


def chat_stock_qa(messages: list, stock_analysis_text: str) -> str:
    """종목 Q&A 채팅"""
    client = _get_client()
    if not client:
        return "API 키가 설정되지 않았습니다."

    system = f"""당신은 자산운용사의 시니어 주식 애널리스트 AI 어시스턴트입니다.
아래 종목 분석을 참고하여 사용자의 질문에 답합니다.
필요하면 웹 검색을 추가로 활용하세요.

규칙:
- 한국어로 답변합니다.
- 수치와 근거를 포함합니다.
- 면접에서 이 종목을 추천할 때 어떻게 설명할지 참고할 수 있도록 답변하세요.

## 종목 분석
{stock_analysis_text or "(분석 없음)"}
"""

    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=system,
            tools=[WEB_SEARCH_TOOL],
            messages=api_messages,
        )
        text_parts = [b.text for b in response.content if b.type == "text"]
        return "\n".join(text_parts) if text_parts else "응답 생성에 실패했습니다."
    except Exception as e:
        return f"오류: {e}"


def generate_stock_quiz(stock_analysis_text: str, stock_name: str, difficulty: str) -> str:
    """종목 관련 면접 퀴즈 출제"""
    client = _get_client()
    if not client:
        return "API 키가 설정되지 않았습니다."

    level = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["기초"])
    system = f"""당신은 자산운용사 면접관 역할의 시니어 CIO입니다.
지원자의 개별 종목 분석 능력과 투자 판단력을 평가합니다.

## 종목 분석 자료
{stock_analysis_text or "(분석 없음)"}

## 난이도: {level['label']}
{level['prompt_hint']}
"""

    prompt = f"""'{stock_name}' 종목에 대한 면접 질문 1개를 출제하세요.

형식:
**[질문]**
(질문 내용 — 종목 분석, 밸류에이션 판단, 투자 의사결정과 관련된 서술형 질문)

**[배경]**
(이 질문이 왜 중요한지 1-2문장)

**[힌트]**
(답변 방향 키워드 2-3개)

질문 예시 (난이도에 맞게 변형):
- 이 종목의 현재 밸류에이션을 어떻게 평가하는가?
- 이 종목에 투자한다면 어떤 카탈리스트를 기대하는가?
- 이 종목의 최대 리스크 요인은 무엇이며 어떻게 헤지하겠는가?
- 이 종목을 포트폴리오에 편입한다면 비중과 근거는?
"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"퀴즈 생성 실패: {e}"


def grade_stock_answer(stock_analysis_text: str, stock_name: str,
                       difficulty: str, question: str, answer: str) -> str:
    """종목 퀴즈 답변 채점"""
    client = _get_client()
    if not client:
        return "API 키가 설정되지 않았습니다."

    level = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["기초"])
    system = f"""당신은 자산운용사 면접관 역할의 시니어 CIO입니다.
'{stock_name}' 종목에 대한 지원자의 분석 능력을 평가합니다.

## 종목 분석 자료
{stock_analysis_text or "(분석 없음)"}

## 난이도: {level['label']}
{level['prompt_hint']}
"""

    user_msg = _QUIZ_GRADE_TEMPLATE.format(question=question, answer=answer)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": user_msg}],
        )
        text_parts = [b.text for b in response.content if b.type == "text"]
        return "\n".join(text_parts) if text_parts else "채점 실패"
    except Exception as e:
        return f"채점 실패: {e}"


def save_stock_study(stock_name: str, analysis: dict):
    """종목 분석 결과 저장"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    history = load_stock_history()
    entry = {
        "date": date.today().isoformat(),
        "stock_name": stock_name,
        "analysis": analysis,
    }
    # 같은 날 같은 종목이면 덮어쓰기
    history = [
        h for h in history
        if not (h.get("date") == entry["date"] and h.get("stock_name") == stock_name)
    ]
    history.append(entry)
    with open(STOCK_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)


def load_stock_history() -> list:
    """저장된 종목 공부 히스토리"""
    if not STOCK_HISTORY_PATH.exists():
        return []
    try:
        with open(STOCK_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def get_stock_analysis_text(analysis) -> str:
    """종목 분석에서 텍스트 추출"""
    if not analysis:
        return ""
    if isinstance(analysis, str):
        return analysis
    return analysis.get("full_text", "")
