# AI 종목 필터 — 디베이트 시스템 인터페이스 명세

> **목적**: 기존 3-에이전트 합의 시스템을 5단계 디베이트 시스템으로 확장.
> 이 문서는 구현 전 함수 시그니처/JSON 스키마/스트리밍 규약을 고정하기 위한 사양서다.
>
> **확정 사항** (사용자 선택)
> - 라운드 간 정보 공개: **전체 의견 + 소멸 테이블(diff)**
> - UI 표시: **토큰 스트리밍**
> - 합의 임계: **일치율 ≥ 0.8 OR 5라운드 도달**

---

## 1. 전체 흐름

```
[1] 데이터 수집     기존 collect_technical_data + collect_news_data
[2] 초기 의견 R0    기존 run_technical_agent + run_news_agent (Top10 각각)
[3] 디베이트 라운드 NEW  R1..R_max
     - 각 AI에게 "상대의 R(n-1) 결과 + 소멸 테이블" 제시
     - 반박/동의/질문 후 수정된 Top10 산출
     - 매 라운드 끝에 일치율 계산
     - 종료 조건: agreement_rate >= 0.8  OR  round == 5
[4] 최종 합의 NEW   run_judge (기존 consensus_agent의 디베이트-aware 버전)
     - 자연 합의(>=0.8): 합산 + 비중 배분만
     - 미합의: 분쟁 종목에 대해 중재 + 비중 결정
[5] 대화 모드 NEW   run_chat_turn (대화형 후속 처리)
     - 설명형 질문 → 자연어 응답
     - 재교섭 요청 → 디베이트 R+1 트리거
```

---

## 2. 파일 구조

| 파일 | 변경 종류 |
|------|----------|
| `lib/ai_stock_filter.py` | **확장** — 디베이트/judge/chat 함수 + 스키마 빌더 추가 |
| `lib/ai_filter_cli.py` | **신규** — Rich 기반 인터랙티브 CLI |
| `cache/ai_filter_logs/ai_filter_{date}.json` | **스키마 확장** — `debate`, `chat_history` 키 추가 |

기존 `run_ai_filter()`는 호환성 유지(legacy mode), 새 진입점 `run_ai_filter_with_debate()` 추가.

---

## 3. 함수 시그니처

### 3.1 디베이트 라운드

```python
def run_debate_round(
    round_num: int,
    prev_tech_result: dict,           # 직전 라운드 tech 결과 (R0 = run_technical_agent 결과)
    prev_news_result: dict,           # 직전 라운드 news 결과
    tech_data: dict,                  # 30종목 지표 (불변)
    news_data: dict,                  # 30종목 뉴스 (불변)
    on_token: Callable[[str, str], None] | None = None,  # (agent, token) → None
) -> dict:
    """
    한 디베이트 라운드 실행: tech-AI와 news-AI가 상대의 직전 라운드 결과를
    보고 반박/동의/질문 후 수정된 Top10 제시.

    Returns
    -------
    {
        "round": int,
        "tech": {
            "rebuttal": str,                  # 상대 의견에 대한 자연어 반박/동의
            "top_10": [...],                  # 수정된 Top10
            "excluded_notable": [...],
            "_raw": {...},                    # system/user prompt, response, model
        },
        "news": {  ...같은 구조... },
        "agreement_rate": float,              # |tech_top10 ∩ news_top10| / 10
        "agreed_stocks": [stock_code, ...],   # 공통 종목 코드
        "diff_from_prev": {
            "tech_dropped": [stock_code, ...],    # 직전 → 현재 빠진 종목
            "tech_added": [stock_code, ...],      # 직전 → 현재 들어온 종목
            "news_dropped": [...],
            "news_added": [...],
        }
    }
    """
```

### 3.2 디베이트 루프 (오케스트레이터)

```python
def run_debate_loop(
    tech_data: dict,
    news_data: dict,
    initial_tech_result: dict,
    initial_news_result: dict,
    max_rounds: int = 5,
    convergence_threshold: float = 0.8,
    on_token: Callable[[str, str], None] | None = None,
    on_round_end: Callable[[dict], None] | None = None,
) -> dict:
    """
    R1..R_max 디베이트 반복. agreement_rate가 임계 도달 시 조기 종료.

    Returns
    -------
    {
        "rounds": [<run_debate_round 결과>, ...],  # R1, R2, ...
        "converged": bool,                          # 자연 합의 여부
        "final_agreement_rate": float,
        "total_rounds": int,
    }
    """
```

### 3.3 Judge (Risk Manager)

```python
def run_judge(
    tech_data: dict,
    initial_tech_result: dict,
    initial_news_result: dict,
    debate: dict,                     # run_debate_loop 결과
    on_token: Callable[[str], None] | None = None,
) -> dict:
    """
    디베이트 전체 이력 + 팩터 점수를 보고 최종 10종목 + 비중 결정.

    - converged=True: 합산 + 비중 배분만 (기존 consensus와 유사)
    - converged=False: 분쟁 종목 (한쪽만 선정) 우선 중재 → 비중 결정

    Returns
    -------
    {
        "final_portfolio": [
            {
                "stock_code": str,
                "stock_name": str,
                "weight_pct": float,
                "confidence": "high"|"medium"|"low",
                "tech_selected_final": bool,   # 마지막 라운드 tech_top10 포함 여부
                "news_selected_final": bool,
                "factor_rank": int,
                "reason": str,
            },
            ...
        ],
        "judge_intervened": bool,             # 분쟁 종목에 대해 강제 결정 했는지
        "judge_reasoning": str,               # 자연어 종합 설명
        "_raw": {...},
    }
    """
```

### 3.4 Chat (대화 모드)

```python
def run_chat_turn(
    chat_history: list[dict],         # [{"role":"user"|"assistant", "content":..., "ts":...}, ...]
    full_context: dict,               # tech_data, debate, final_portfolio 등 전체 상태
    user_msg: str,
    on_token: Callable[[str], None] | None = None,
) -> dict:
    """
    설명형 vs 재교섭형 자동 판별 후 응답.

    Returns
    -------
    {
        "response": str,                          # 사용자에게 보일 자연어 답변
        "trigger_renegotiate": bool,              # True면 CLI가 run_debate_round 추가 호출
        "renegotiate_hint": {
            "force_include": [stock_code, ...],   # 사용자가 "Y 빼고 Z 넣어" 한 경우
            "force_exclude": [stock_code, ...],
            "user_intent": str,                   # 다음 라운드 프롬프트에 주입할 자연어
        } | None,
        "_raw": {...},
    }
    """
```

### 3.5 메인 진입점

```python
def run_ai_filter_with_debate(
    stocks: list[tuple[str, float]],
    calc_date: str,
    conn=None,
    max_rounds: int = 5,
    convergence_threshold: float = 0.8,
    on_token: Callable[[str, str], None] | None = None,
    on_round_end: Callable[[dict], None] | None = None,
) -> dict:
    """전체 파이프라인 실행. 결과는 §4 스키마."""
```

---

## 4. JSON 스키마 (`ai_filter_{date}.json`)

```jsonc
{
  "calc_date": "2026-05-20",
  "input_stocks": 30,
  "schema_version": "2.0",

  "tech_data": { /* 기존 collect_technical_data 결과 */ },
  "news_data": { /* 기존 collect_news_data 결과 */ },

  // R0: 초기 의견 (기존 tech_result, news_result와 동일)
  "initial": {
    "tech_result": { "top_10": [...], "excluded_notable": [...], "_raw": {...} },
    "news_result": { "top_10": [...], "excluded_notable": [...], "_raw": {...} }
  },

  // R1..R_max
  "debate": {
    "rounds": [
      {
        "round": 1,
        "tech": {
          "rebuttal": "뉴스 AI가 제시한 X 종목은 ADX 12로 추세 약함...",
          "top_10": [...],
          "excluded_notable": [...],
          "_raw": {...}
        },
        "news": { /* 같은 구조 */ },
        "agreement_rate": 0.6,
        "agreed_stocks": ["005930", "000660", ...],
        "diff_from_prev": {
          "tech_dropped": ["XXXXXX"],
          "tech_added":   ["YYYYYY"],
          "news_dropped": [...],
          "news_added":   [...]
        }
      }
      /* R2, R3, ... */
    ],
    "converged": true,
    "final_agreement_rate": 0.9,
    "total_rounds": 3
  },

  // Judge 결과
  "judge": {
    "final_portfolio": [
      {
        "stock_code": "XXXXXX",
        "stock_name": "종목A",
        "weight_pct": 15.0,
        "confidence": "high",
        "tech_selected_final": true,
        "news_selected_final": true,
        "factor_rank": 1,
        "reason": "..."
      }
    ],
    "judge_intervened": false,
    "judge_reasoning": "...",
    "_raw": {...}
  },

  // 최상위 편의 키 (judge.final_portfolio와 동일)
  "final_portfolio": [...],

  // 대화 모드 이력
  "chat_history": [
    {
      "role": "user",
      "content": "왜 두산밥캣을 14%로 잡았어?",
      "ts": "2026-05-20T14:23:11+09:00"
    },
    {
      "role": "assistant",
      "content": "기술/뉴스 모두 동의했고 팩터점수도...",
      "trigger_renegotiate": false,
      "ts": "..."
    }
  ]
}
```

**참고**: 기존 `ai_filter_2026-04-30.json` 등은 `schema_version` 키 없음 → 로딩 시 v1로 판단.

---

## 5. 스트리밍 규약

**선택**: 콜백 방식 (generator는 dict 반환과 어긋남).

```python
# 백엔드
def run_debate_round(..., on_token: Callable[[str, str], None] | None = None):
    with client.messages.stream(...) as stream:
        for text in stream.text_stream:
            if on_token:
                on_token("tech", text)   # 또는 "news", "judge", "chat"
    ...
```

```python
# CLI (Rich)
from rich.live import Live
def on_token(agent: str, token: str):
    state.buffers[agent] += token
    live.update(render(state))
```

**라운드 종료 콜백** (`on_round_end`): UI가 일치율 / diff 테이블 표시할 시점.

---

## 6. 일치율 (agreement_rate) 정의

**v1 (기본)**: 단순 집합 교집합
```python
agreement_rate = len(set(tech_top10_codes) & set(news_top10_codes)) / 10
```

**v2 (선택지, 추후 검토)**: 상위 가중 — 상위 5종목 일치는 1.5배 가중
- 현재는 v1로 시작. 디베이트 수렴이 너무 빠르거나 느리면 v2로 전환.

---

## 7. 라운드 간 정보 공개 (확정: "전체 + 소멸 테이블")

각 AI에게 직전 라운드에서 제시되는 정보:

1. **상대의 전체 의견** — top_10 + reason + excluded_notable
2. **소멸 테이블 (diff)** — 자신과 상대가 직전 → 현재로 뺀/넣은 종목
3. **자신의 직전 의견** — 일관성 유지용
4. **(선택) 상대의 rebuttal** — 이전 라운드에서 자신을 향한 반박

프롬프트 템플릿은 `lib/ai_stock_filter.py` 안에 상수로 정의:
- `TECHNICAL_DEBATE_PROMPT_TEMPLATE`
- `NEWS_DEBATE_PROMPT_TEMPLATE`

---

## 8. 구현 순서 (단독 작업)

| 순서 | 작업 | 검증 방법 |
|------|------|----------|
| 1 | 스키마 빌더 + diff/일치율 유틸 (`_compute_agreement`, `_compute_diff`) | 단위 테스트 (mock top10 입력) |
| 2 | `run_debate_round` (스트리밍 콜백 없이 먼저) | 실데이터 1라운드 호출 |
| 3 | 디베이트 프롬프트 튜닝 (반박/동의/질문 톤) | 결과 JSON 응답 quality 점검 |
| 4 | `run_debate_loop` + 조기 종료 | 2-3라운드 돌려보고 수렴 케이스 확인 |
| 5 | `run_judge` (converged / non-converged 두 분기) | 인위적으로 non-converged 케이스 만들어 중재 확인 |
| 6 | `run_chat_turn` + 재교섭 트리거 판별 | "왜 X 골랐어?" / "Y 빼고 Z 넣어" 두 케이스 |
| 7 | 스트리밍 콜백 추가 | print 콜백으로 단순 검증 |
| 8 | `lib/ai_filter_cli.py` Rich UI | 통합 실행 |
| 9 | `run_ai_filter_with_debate` 통합 + 로그 저장 | E2E |

---

## 9. 결정 미루기 (later)

- 일치율 v2 가중치는 v1 결과 보고 결정
- chat 모드의 재교섭 한도(R6, R7... 무한 허용?) — 일단 +3까지로
- judge 모델 분리 (consensus는 opus / debate는 sonnet) — 비용/품질 보고 결정

---

## 10. UI 화면 목업 (Rich CLI)

대화 흐름 8단계. `lib/ai_filter_cli.py` 구현 시 참조.

### 10.1 Welcome — 진입 화면

```
╭─────────────────────────────────────────────────────────────────────╮
│           AI 종목 필터 — 디베이트 시스템 v2.0                       │
│                                                                     │
│   기준일:  2026-05-20                                               │
│   전략:    EVIC/ROIC Attractiveness                                 │
│   후보:    30종목 (팩터 상위)                                       │
│                                                                     │
│   파이프라인:                                                       │
│     [1] 데이터 수집  ━━━━━━━━━━  완료 (30종목 지표 + 뉴스)          │
│     [2] 초기 의견 R0                                                │
│     [3] 디베이트 R1..R5  (합의 임계 0.8)                            │
│     [4] Risk Manager 중재                                           │
│     [5] 대화 모드                                                   │
│                                                                     │
│              [Enter] 시작    [q] 종료                               │
╰─────────────────────────────────────────────────────────────────────╯
```

### 10.2 R0 — 초기 의견 (좌우 분할)

```
╭─ R0 · 기술적 분석 AI ──────────╮ ╭─ R0 · 뉴스 분석 AI ────────────╮
│ Top 10                         │ │ Top 10                         │
│  1. 두산밥캣  (241560)  92     │ │  1. HL만도   (204320)  88      │
│  2. 삼성전자  (005930)  88     │ │  2. 두산밥캣 (241560)  85      │
│  3. SK하이닉스(000660)  85     │ │  3. 현대차   (005380)  82      │
│  4. HL만도   (204320)  84     │ │  4. NAVER    (035420)  80      │
│  5. ...                        │ │  5. ...                        │
│                                │ │                                │
│ Excluded notable               │ │ Excluded notable               │
│  · 종목X — RSI 82 과매수       │ │  · 종목Y — 소송 진행 중        │
╰────────────────────────────────╯ ╰────────────────────────────────╯
╭─ 일치율 ─────────────────────────────────────────────────────────╮
│  R0  공통 4종목  ████░░░░░░ 40%                  [Enter] R1 시작 │
╰──────────────────────────────────────────────────────────────────╯
```

### 10.3 R1 — 스트리밍 중 (실시간 토큰 흐름)

```
╭─ R1 · 기술적 AI ⚡ ────────────╮ ╭─ R1 · 뉴스 AI ⚡ ──────────────╮
│ [반박]                         │ │ [반박]                         │
│ 뉴스 AI가 제시한 NAVER는       │ │ 기술 AI의 SK하이닉스 선정은    │
│ ADX 12로 추세 약함. RSI도      │ │ 동의. 다만 변동성 종목 X는     │
│ 53 중립이라 진입 매력 낮음▌    │ │ 1분기 어닝쇼크 우려 있음▌      │
│                                │ │                                │
│ [수정 Top 10]                  │ │ [수정 Top 10]                  │
│  (스트리밍 대기 중…)           │ │  (스트리밍 대기 중…)           │
╰────────────────────────────────╯ ╰────────────────────────────────╯
╭─ 진행 ───────────────────────────────────────────────────────────╮
│  R1 디베이트 진행 중...  ━━━━━━━━━━━━━░░░░░  62%                 │
╰──────────────────────────────────────────────────────────────────╯
```

### 10.4 R1 종료 — 일치율 + 소멸 테이블

```
╭─ R1 · 결과 비교 ─────────────────────────────────────────────────╮
│                                                                  │
│  일치율  ███████░░░ 70%  (+30%p ↑)                               │
│                                                                  │
│  ┌── 소멸 테이블 ─────────────────────────────────────────────┐  │
│  │              직전(R0)  →  현재(R1)                         │  │
│  │  Tech  드롭:  NAVER(035420), 종목Z                         │  │
│  │  Tech  추가:  현대차(005380), LG에너지(373220)             │  │
│  │  News  드롭:  종목Y                                        │  │
│  │  News  추가:  SK하이닉스(000660)                           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  공통 7종목: 두산밥캣, 삼성전자, SK하이닉스, HL만도, 현대차, ... │
│                                                                  │
│  합의 임계 0.8 미달 → R2 진행                  [Enter] 계속      │
╰──────────────────────────────────────────────────────────────────╯
```

### 10.5 수렴 — 합의 도달

```
╭─ R3 · 합의 ✓ ────────────────────────────────────────────────────╮
│                                                                  │
│  일치율  █████████░ 90%   ≥ 0.8 → 자연 합의                      │
│                                                                  │
│  라운드 추이:  R0 40% → R1 70% → R2 80% → R3 90%                 │
│                                                                  │
│  공통 9종목 (회색=R2 추가, 녹색=처음부터 합의):                  │
│   ● 두산밥캣  ● 삼성전자  ● SK하이닉스  ● HL만도  ● 현대차       │
│   ● NAVER    ● 카카오    ○ LG에너지    ○ 셀트리온                │
│                                                                  │
│           Risk Manager 중재 단계로 진행 →                        │
╰──────────────────────────────────────────────────────────────────╯
```

### 10.6 Judge — 최종 포트폴리오

```
╭─ Risk Manager · 최종 포트폴리오 ─────────────────────────────────╮
│                                                                  │
│  judge_intervened: false (자연 합의 → 비중 배분만 수행)          │
│                                                                  │
│  ┌────┬──────────┬────────┬───────┬──────┬──────┬─────────────┐  │
│  │ #  │ 종목명   │ 코드   │ 비중  │ 확신 │ 팩터 │ 사유 요약   │  │
│  ├────┼──────────┼────────┼───────┼──────┼──────┼─────────────┤  │
│  │  1 │ 두산밥캣 │ 241560 │ 14.0% │ High │  19  │ R0~R3 합의… │  │
│  │  2 │ HL만도   │ 204320 │ 13.0% │ High │  23  │ 어닝서프…   │  │
│  │  3 │ 삼성전자 │ 005930 │ 12.0% │ High │   3  │ 추세+모멘…  │  │
│  │ …  │   …      │   …    │   …   │  …   │  …   │     …       │  │
│  └────┴──────────┴────────┴───────┴──────┴──────┴─────────────┘  │
│                                                                  │
│  합계: 100.0%       섹터분산: 6 | 평균확신도: High               │
╰──────────────────────────────────────────────────────────────────╯
            [c] 대화 모드   [s] 저장   [q] 종료
```

### 10.7 대화 모드 — 설명형 질문

```
╭─ 대화 모드 ──────────────────────────────────────────────────────╮
│  💬 you  > 왜 두산밥캣을 14%로 잡았어?                            │
│                                                                  │
│  🤖 consensus                                                    │
│  두산밥캣은 R0부터 R3까지 양쪽 AI 모두 일관되게 Top10에 포함했고,│
│  팩터 점수는 19위로 상위권은 아니지만 합의도가 가장 높았습니다.  │
│  기술적으로 ADX 30.8(강한 추세), MACD 양수, OBV 양의 기울기에다  │
│  뉴스 측에서도 KB·키움·다올 목표가 동시 상향, 1분기 어닝서프…   │
│                                                                  │
│  > _                                                             │
╰──────────────────────────────────────────────────────────────────╯
            trigger_renegotiate = false → 디베이트 재개 없음
```

### 10.8 대화 모드 — 재교섭 트리거 → R+1

```
╭─ 대화 모드 ──────────────────────────────────────────────────────╮
│  💬 you  > 셀트리온 빼고 카카오뱅크 넣어                         │
│                                                                  │
│  🤖 consensus                                                    │
│  사용자 요청을 디베이트 재개로 전달합니다:                       │
│    force_exclude: 셀트리온(068270)                               │
│    force_include: 카카오뱅크(323410)                             │
│  두 AI에게 이 제약 하에 R4 재논의를 요청합니다…                  │
│                                                                  │
│  ⟳ R4 재교섭 시작 (사용자 제약 주입)                             │
╰──────────────────────────────────────────────────────────────────╯
       ↓ run_debate_round(round_num=4, ..., user_hint=...)
╭─ R4 · 기술적 AI ⚡ ────────────╮ ╭─ R4 · 뉴스 AI ⚡ ──────────────╮
│ 사용자 요청 반영: 카카오뱅크는 │ │ 카카오뱅크는 1분기 순이익 +37% │
│ MA5/20 골든크로스, MFI 62 양호▌│ │ 발표 후 외국인 매수 지속▌      │
╰────────────────────────────────╯ ╰────────────────────────────────╯
```
