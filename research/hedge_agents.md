# HedgeAgents: Multi-Agent Financial Trading System

- 논문: https://arxiv.org/html/2502.13165v1
- 프로젝트 페이지: https://hedgeagents.github.io/
- 소스 코드: 미공개 (논문 기반 직접 구현 필요)

## 핵심 구조

### 에이전트 (원본)
| 에이전트 | 담당 |
|---|---|
| Dave | 비트코인 |
| Bob | 다우존스 주식 |
| Emily | 외환 (FX) |
| Otto | 헤지펀드 매니저 (비중 배분) |

### 3가지 컨퍼런스
1. **BAC (Budget Allocation Conference)** — 30일마다, 섹터별 수익 보고 → Otto가 비중 최적화
2. **EMC (Extreme Market Conference)** — 일일 5% or 3일간 10% 변동 시, 긴급 대응
3. **ESC (Experience Sharing Conference)** — 투자 사이클 종료 시, 과거 경험 공유 → 메모리 저장

### 일일 루프
메모리에서 유사 케이스 검색 → LLM으로 매매 판단 → 결과 리플렉션 저장

### 백테스트 결과 (논문)
- 연 수익률: 71.6%, 총 수익률: 405% (2015~2023, 3년)
- 주의: 백테스트 전용, LLM 데이터 누수 가능성 있음

---

## 한국 KOSPI+KOSDAQ 적용 설계안

### 헤징 축: 섹터 기반
| 에이전트 | 담당 | 근거 |
|---|---|---|
| 반도체 애널리스트 | 삼성전자, SK하이닉스 등 | 경기순환, 글로벌 수요 |
| 바이오 애널리스트 | 셀트리온, 삼성바이오 등 | 임상/규제, 코스닥 비중 |
| 2차전지 애널리스트 | LG에너지, 에코프로 등 | 테마/모멘텀 |
| 내수/금융 애널리스트 | 은행, 유통, 통신 등 | 방어적, 배당 |
| Otto (펀드 매니저) | 섹터 간 비중 배분 | 전체 리스크 관리 |

### 컨퍼런스 한국 버전
| 컨퍼런스 | 트리거 | 비고 |
|---|---|---|
| BAC | 월 1회 | 섹터 비중 조절 |
| EMC | KOSPI 일일 3%↓ or 3일간 5%↓ | 한국 시장 변동성 반영 |
| ESC | 분기 말 | 성공/실패 케이스 공유 |

### 데이터 소스
- 시세: pykrx, FinanceDataReader
- 공시/뉴스: DART API, 네이버 뉴스
- 재무제표: OpenDART API
- 기술지표: TA-Lib / pandas-ta

### TODO
- [ ] 섹터 구분 확정
- [ ] 섹터당 종목 수 결정
- [ ] LLM 선택 (GPT-4 / Claude / 로컬)
- [ ] 프레임워크 선택 (LangGraph / CrewAI / 직접 구현)
- [ ] 프로토타입 구현
