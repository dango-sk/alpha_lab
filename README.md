# Alpha Lab

한국 대형주 멀티팩터 퀀트 백테스트 시스템

> 애널리스트 컨센서스 기반 Forward 팩터와 회귀 매력도 모델을 결합하여,
> KOSPI 200 대비 연환산 +7%p 초과수익(CAGR 23.1%, Sharpe 0.97)을 달성한 전략을 설계하고 검증합니다.

---

## 주요 성과

| 지표 | A0 전략 (14팩터) | KOSPI 200 |
|------|:-:|:-:|
| **총 수익률** | 193.2% (2.93x) | 116.4% (2.16x) |
| **CAGR** | 23.1% | 16.1% |
| **Sharpe Ratio** | 0.97 | 0.74 |
| **MDD** | -35.8% | -33.9% |
| **월평균 수익률** | 1.99% | — |
| **월 변동성** | 7.10% | — |

- 백테스트 기간: 2021.01 ~ 2026.02 (62개월)
- 월간 리밸런싱 / 거래비용 30bp 반영 / 종목당 최대 15% 비중 제한

---

## 전략 비교

| 전략 | 팩터 구성 | 총 수익 | CAGR | Sharpe | MDD |
|------|----------|:-------:|:----:|:------:|:---:|
| **A0** | 밸류 7 + 매력도 4 + 성장 3 | 1.93x | 23.1% | 0.97 | 35.8% |
| Core7 | 밸류 + 매력도 + 성장 | 1.76x | 21.7% | 0.84 | 40.6% |
| VR4 | 밸류 + 매력도 + 모멘텀 | 1.74x | 21.5% | 0.88 | 36.3% |
| ATT2 | 매력도만 | 1.33x | 17.8% | 0.88 | 28.3% |
| Core5 | 매력도 + 성장 + 모멘텀 | 1.29x | 17.4% | 0.75 | 46.2% |
| KOSPI 200 | 벤치마크 | 1.16x | 16.1% | 0.74 | 33.9% |

**핵심 발견:** 밸류 팩터를 제거하면(Core5) CAGR은 유지되나 MDD가 46.2%로 확대. 밸류 팩터가 직접적 알파보다 **밸류에이션 앵커**(하방 리스크 억제) 역할을 함.

---

## 팩터 시스템

### 14개 팩터 (A0 전략)

**밸류 (7개)** — 낮을수록 저평가
| 팩터 | 설명 | 비중 |
|------|------|:----:|
| T_PER | Trailing P/E | 5% |
| F_PER | Forward P/E | 5% |
| T_EV/EBITDA | Trailing EV/EBITDA | 5% |
| F_EV/EBITDA | Forward EV/EBITDA | 5% |
| T_PBR | Trailing P/B | 5% |
| F_PBR | Forward P/B | 5% |
| T_PCF | Price/Cash Flow | 5% |

**회귀 매력도 (4개)** — 이익 대비 밸류에이션 괴리 포착
| 팩터 | 회귀 모델 | 비중 |
|------|----------|:----:|
| ATT_PBR | PBR ~ ROE | 5% |
| ATT_EVIC | EV/IC ~ ROIC | 5% |
| ATT_PER | F.PER ~ EPS Growth | 10% |
| ATT_EVEBIT | F.EV/EBIT ~ EBIT Growth | 10% |

**성장/모멘텀 (3개)** — 높을수록 긍정적
| 팩터 | 설명 | 비중 |
|------|------|:----:|
| T_SPSG | 매출 YoY 성장률 | 10% |
| F_SPSG | Forward 매출 성장률 | 10% |
| F_EPS_M | Forward EPS 3개월 모멘텀 | 15% |

### 스코어링
- 대형주(시총 상위 200) 대상 4분위 스코어링 (0~4점)
- 팩터별 가중합산 → 0~100 정규화 → 상위 30종목 편입

---

## 종목 필터

- SPAC, ETF, REITs 제외
- 금융업종 제외 (은행, 보험, 증권 등)
- 영업이익 > 0, ROE > 0
- 20일 평균 거래대금 > 5억원
- 부채비율 < 200%

---

## 강건성 검증

- **In-Sample / Out-of-Sample 분할**: IS 2021.01~2024.06 / OOS 2024.07~2026.02
- **롤링 윈도우 분석**: 24개월 롤링 Sharpe Ratio
- **통계적 유의성**: 월별 초과수익 t-test
- **팩터별 5분위 롱숏 분석**: 14개 팩터 개별 유효성 검증

---

## 대시보드

Streamlit 기반 인터랙티브 대시보드 (5개 탭):

| 탭 | 기능 |
|----|------|
| **성과 비교** | 멀티 전략 누적수익률 차트, 핵심 지표 테이블 |
| **월별 분석** | 월별 수익률 히트맵, 수익률 분포, 롤링 성과 |
| **포트폴리오** | 현재 보유종목, 섹터/시총 분포, 개별 비중 |
| **통계 검증** | IS/OOS 비교, 롤링 Sharpe, 팩터 롱숏 결과 |
| **전략 실험실** | AI 자연어 전략 설계 + 즉시 백테스트 실행 |

**AI 전략 설계**: Claude API를 활용하여 자연어로 투자 전략을 설명하면, 팩터 가중치와 필터 조건을 자동 생성하여 백테스트 실행

---

## 프로젝트 구조

```
alpha_lab/
├── app.py                    # Streamlit 대시보드
├── config/
│   └── settings.py           # 백테스트 설정, 팩터 정의, 필터 조건
├── lib/
│   ├── factor_engine.py      # 팩터 계산 및 전략 파이프라인
│   ├── data.py               # 데이터 로더 및 캐싱
│   ├── views.py              # 대시보드 UI 컴포넌트
│   ├── charts.py             # 시각화
│   ├── ai.py                 # Claude API 연동
│   ├── chat.py               # AI 챗봇 UI
│   └── style.py              # CSS 스타일링
├── scripts/
│   ├── run_pipeline.py       # 전체 파이프라인 실행
│   ├── step1_update_prices.py
│   ├── step3_calc_value_factors.py
│   ├── step6_generate_signals.py
│   ├── step7_backtest.py     # 백테스트 엔진
│   └── step8_robustness.py   # 강건성 검증
├── analysis/
│   ├── factor_longshort.py   # 단일 팩터 롱숏 분석
│   └── strategy_compare.py   # 전략 비교 분석
└── cache/                    # 캐시 결과 (JSON)
```

---

## 데이터 파이프라인

```
Step 1: 주가 업데이트 (FnSpace API → SQLite)
  ↓
Step 3: 밸류 팩터 계산 (월별)
  ↓
Step 6: 시그널 생성 (스코어링)
  ↓
Step 7: 백테스트 실행
  ↓
Step 8: 강건성 검증
```

- 월~금 17:00 자동 실행 (cron)
- 전체 소요 시간: 약 20~40분

---

## 실행 방법

```bash
# 파이프라인 실행 (데이터 수집 → 백테스트)
python scripts/run_pipeline.py

# 대시보드 실행
streamlit run app.py
```

---

## 기술 스택

- **Python** — pandas, numpy, scipy, scikit-learn
- **Streamlit** — 대시보드
- **SQLite** — 주가/재무 데이터 저장
- **Plotly** — 차트 시각화
- **Claude API** — AI 자연어 전략 설계
- **FnSpace API** — 한국 주식 데이터 소스
