# Alpha Lab

한국 대형주 멀티팩터 퀀트 전략 시스템 + 웹 대시보드

> 애널리스트 컨센서스 기반 Forward 팩터와 회귀 매력도 모델을 결합하여,
> KOSPI 200 대비 초과수익을 추구하는 전략을 설계, 백테스트, 운용합니다.

**대시보드**: https://alpha-dashboard-production-e16d.up.railway.app/portfolio

---

## 목적

1. **팩터 전략 백테스트** — 14개 팩터(밸류/매력도/성장) 조합으로 KOSPI 대형주 30종목 포트폴리오 구성
2. **AI 레짐 예측** — 매크로 지표 + 뉴스를 기반으로 상승/하락장을 판단, 전략 전환에 활용
3. **웹 대시보드** — 성과 비교, 포트폴리오 보유종목, 통계 검증 결과를 실시간 확인
4. **자동화 파이프라인** — 매일 데이터 수집 → 백테스트 → 캐시 갱신까지 자동 실행

---

## 시스템 구조

```
┌─ 데이터 수집 (파이프라인) ──────────────────────────────────────────┐
│                                                                     │
│  FnSpace API ──→ collect_macro.py ──────→ macro_indicators 테이블   │
│  FnSpace API ──→ step7_collect_consensus ──→ fnspace_forward/       │
│                                              fnspace_consensus_daily│
│  네이트 뉴스 ──→ collect_news_nate.py ──→ news 테이블               │
│  FnSpace API ──→ run_pipeline.py ───────→ fnspace_master/finance    │
│                                                                     │
│  LG 그램 ─────→ (별도 업로드) ──────────→ daily_price, marketcap    │
│                                                                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              ▼
┌─ 계산 엔진 ────────────────────────────────────────────────────────┐
│                                                                     │
│  step5b_calc_ttm.py ────→ TTM 재무 계산                             │
│  step6_build_universe.py ─→ 유니버스 구축                            │
│  step7_backtest.py ───────→ 4콤보 백테스트 실행                      │
│  step8_robustness.py ─────→ 강건성 검증 (IS/OOS, bootstrap)         │
│  factor_engine.py ────────→ 팩터 점수 계산                           │
│                                                                     │
│           ▼ 결과 저장                                                │
│  ┌─────────────────────────────────────┐                            │
│  │  Railway PostgreSQL (PG)            │                            │
│  │  ├─ backtest_cache                  │                            │
│  │  ├─ daily_price                     │                            │
│  │  ├─ universe                        │                            │
│  │  ├─ fnspace_forward/consensus_daily │                            │
│  │  ├─ fnspace_master/finance          │                            │
│  │  ├─ macro_indicators                │                            │
│  │  └─ news                            │                            │
│  └─────────────────┬───────────────────┘                            │
│                     │                                                │
│  cache/*.json ◄─────┘ (로컬 캐시 사본, Railway 배포용)               │
│                                                                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              ▼
┌─ AI 레짐 예측 ─────────────────────────────────────────────────────┐
│                                                                     │
│  regime_agent.py ──→ macro_indicators + news (PG 읽음)              │
│  regime_agent_v2.py    │                                            │
│                        ▼                                            │
│            regime_agent_results.json ──→ lib/data.py가 읽어서       │
│            regime_agent_v2_results.json   레짐 모드에 반영           │
│                                                                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              ▼
┌─ 서비스 (웹 대시보드) ─────────────────────────────────────────────┐
│                                                                     │
│  ┌─ Frontend (Next.js) ─┐     ┌─ Backend (FastAPI) ──────────┐    │
│  │                       │     │                               │    │
│  │  /performance ────────┼──→  │  /api/results ← lib/data.py  │    │
│  │  /portfolio ──────────┼──→  │  /api/holdings    ↕           │    │
│  │  /statistics ─────────┼──→  │  /api/robustness  PG DB      │    │
│  │  /lab ────────────────┼──→  │  /api/backtest               │    │
│  │  /monthly ────────────┼──→  │  /api/monthly                │    │
│  │  /chat ───────────────┼──→  │  /api/chat ← lib/ai.py      │    │
│  │                       │     │              (Anthropic API)  │    │
│  └───────────────────────┘     └───────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Railway 배포 구조

```
로컬 (git push origin main)
        │
        ▼
   GitHub repo (dango-sk/alpha_lab)
        │
        ▼ Railway가 자동 감지
   Railway 빌드 & 배포
   ├─ Dockerfile로 이미지 빌드
   ├─ start.sh 실행 (FastAPI + Next.js 동시 기동)
   └─ https://alpha-dashboard-production-e16d.up.railway.app 서빙
```

- **배포 방법**: `git push origin main` 하면 Railway가 자동으로 재배포
- **DB**: Railway에서 호스팅하는 PostgreSQL (metro.proxy.rlwy.net)
- **환경변수**: Railway 대시보드에서 설정 (DATABASE_URL, API 키 등)

---

## 대시보드 페이지

| 페이지 | URL | 기능 |
|--------|-----|------|
| 성과 비교 | /performance | 멀티 전략 누적수익률 차트, 핵심 지표 |
| 포트폴리오 | /portfolio | 현재 보유종목, 섹터/시총 분포, 종목별 수익률 |
| 월별 분석 | /monthly | 월별 수익률 히트맵, 분포, 롤링 성과 |
| 통계 검증 | /statistics | IS/OOS 비교, 롤링 Sharpe, 팩터 유효성 |
| 전략 실험실 | /lab | AI 자연어로 전략 설계 + 즉시 백테스트 |
| AI 채팅 | /chat | Claude 기반 시장 분석 챗봇 |

---

## 프로젝트 구조

```
alpha_lab/
├── app.py                     # Streamlit 대시보드 (레거시)
├── backend/
│   └── main.py                # FastAPI 백엔드 (API 서버)
├── frontend/                  # Next.js 프론트엔드
│   └── src/
│       ├── app/               # 페이지 (performance, portfolio, ...)
│       ├── components/        # 공통 컴포넌트
│       └── lib/               # API 호출, 유틸리티
├── lib/                       # 핵심 비즈니스 로직
│   ├── data.py                # 데이터 로더, 백테스트 실행, 캐싱
│   ├── factor_engine.py       # 팩터 계산 및 스코어링
│   ├── db.py                  # DB 연결 (PostgreSQL)
│   ├── ai.py                  # Claude API 연동
│   ├── market_ai.py           # AI 시장 분석
│   ├── views.py               # Streamlit UI 컴포넌트
│   ├── charts.py              # Plotly 차트
│   └── style.py               # CSS
├── config/
│   └── settings.py            # 팩터 정의, 필터 조건, 백테스트 설정
├── scripts/                   # 데이터 파이프라인
│   ├── run_pipeline.py        # 메인 파이프라인 (매일/월초 실행)
│   ├── collect_macro.py       # 매크로 지표 수집 (FnSpace + VIX)
│   ├── collect_news_nate.py   # 뉴스 수집 (네이트)
│   ├── step5b_calc_ttm.py     # TTM 재무 계산
│   ├── step6_build_universe.py# 유니버스 구축
│   ├── step7_backtest.py      # 백테스트 엔진
│   ├── step7_collect_consensus.py # Forward/Consensus 수집
│   └── step8_robustness.py    # 강건성 검증
├── analysis/                  # AI 레짐 예측
│   ├── regime_agent.py        # 레짐 에이전트 V1
│   ├── regime_agent_v2.py     # 레짐 에이전트 V2
│   ├── regime_agent_results.json      # V1 예측 결과 (lib/data.py가 읽음)
│   └── regime_agent_v2_results.json   # V2 예측 결과
├── cache/                     # 백테스트/팩터 캐시 JSON (Railway에서 읽음)
├── Dockerfile                 # 배포용 Docker 설정
├── start.sh                   # FastAPI + Next.js 동시 실행
└── .env.example               # 환경변수 템플릿
```

---

## 실행 방법

### 1. 환경 설정

```bash
git clone https://github.com/dango-sk/alpha_lab.git
cd alpha_lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# .env 파일 생성 (.env.example 참고, 실제 키는 별도 전달)
cp .env.example .env
```

### 2. 파이프라인 실행

```bash
# 일일 실행 (Forward/Consensus 수집 → 백테스트 → 강건성)
python scripts/run_pipeline.py

# 월초 전체 실행 (마스터 + 재무 + TTM + 유니버스 포함)
python scripts/run_pipeline.py --monthly

# 백테스트만 실행 (수집 스킵)
python scripts/run_pipeline.py --only-backtest

# 커스텀 전략 재계산만
python scripts/run_pipeline.py --only-custom
```

### 3. 로컬 대시보드

```bash
# FastAPI 백엔드
cd backend && uvicorn main:app --reload --port 8000

# Next.js 프론트엔드 (별도 터미널)
cd frontend && npm install && npm run dev
```

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| Backend | FastAPI, Python, pandas, numpy, scipy |
| Frontend | Next.js, TypeScript, Plotly |
| DB | PostgreSQL (Railway 호스팅) |
| AI | Claude API (전략 설계, 시장 분석, 레짐 예측) |
| 데이터 | FnSpace API (한국 주식/재무/컨센서스) |
| 배포 | Railway (Docker), GitHub 연동 자동 배포 |

---

## 협업 가이드

### 브랜치 전략

```
main ← 항상 안정 버전 (Railway 자동 배포)
 ├── feature/xxx ← 새 기능 작업
 └── fix/xxx ← 버그 수정
```

1. `git checkout -b feature/내작업` 으로 브랜치 생성
2. 작업 후 `git push origin feature/내작업`
3. GitHub에서 PR 생성 → 리뷰 → main에 merge
4. main에 merge되면 Railway 자동 배포

### 주의사항

- **main에 직접 push 주의** — push하면 바로 배포됨
- 작업 전 `git pull origin main`으로 최신 코드 반영
- `.env` 파일은 git에 안 올라감 — 별도 전달받아서 루트에 배치
