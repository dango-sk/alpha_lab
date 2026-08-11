
# usquant — 미국 중소형주 레짐-적응형 퀀트 시스템

HSMM 레짐 판단 → BOS 스크리닝 → Agent 결격제거 → 정량랭킹 → 비중배분

---

## 1. 설계 원칙

이 시스템은 두 가지 판단 위에 세워져 있습니다.

**(1) 반등을 놓치는 것이 하락을 맞는 것보다 비싸다**

그래서 변동성 스파이크에 즉각 반응하지 않습니다. vol-targeting은 기본적으로 잠겨 있고,
국면이 기대수명에 근접했을 때(d/D ≥ 0.6)만 열립니다. 같은 급락이라도 강세장 초반의
급락과 후반의 급락은 정보량이 다르다 — 이것이 HMM 대신 HSMM을 쓰는 유일한 정당화입니다.

**(2) LLM Agent는 '선택자'가 아니라 '제거자'여야 한다**

"유망한 30개를 골라라"는 미래 예측이고, 백테스트에서 학습데이터 오염을 검증할 방법이
없습니다. "결격 사유가 있는 것을 빼라"는 과거 사실 판정이고, 재현·검증이 가능합니다.
최종 30종목 선택은 결정론적 팩터 랭킹이 담당합니다.

---

## 2. 파이프라인

```
[HSMM]  IWM/SPY/HYG/IEF/VIX → 6개 피처 → filtered prob + d/D
   │       walk-forward 재학습 (6개월 주기), filtered만 사용 (Viterbi 금지)
   │
   ├── 약세 판정 → 노출 40%, 신규 편입 중단, 기존 보유 축소만
   │
   └── 강세 판정 → 노출 100%
        │  d/D < 0.60 → vol-target 잠금 (100% 고정)
        │  d/D ≥ 0.60 → vol-target 활성 (하한 60%)
        ▼
   [BOS]  RSI + 볼린저 z + 52주 낙폭 → 상위 100
        │  하락의 '질' 필터: MA200 대비 -25% 이내, 고점대비 -45% 이내
        │  유동성 필터: AUM 기준 진입 가능 종목만
        ▼
   [Agent]  마스킹된 재무·공시 → 8개 결격코드 이진 판정
        │  회사명/티커/연도/금액 제거, 근거 필드 명시 강제, 전건 로그
        ▼
   [Rank]  Value 35% + Quality 40% + Momentum 25%, 섹터 중립
        │  랭킹 유니버스 = BOS 후보 ∪ 현재 보유  ← 중요
        ▼
   [TurnoverCtrl]  버퍼 2.0×N, 최소보유 3개월, 월 회전율 상한 25%
        ▼
   [Allocate]  역변동성 + 랭킹 틸트 0.3, 개별 6%, 섹터 25%, ADV 8% 제약
```

---

## 3. 검증된 것

### EDHMM 엔진 (합성데이터, 알려진 정답 대비)

| 항목 | 결과 |
|---|---|
| 국면 수명 복원 | 51.6일 / 118.4일 (실제 45 / 120) |
| filtered 상태 복원 정확도 | 99.5% |
| 국면 전환 횟수 | 실제 47 / **EDHMM 61** / **GMM 755** |
| Hazard 비율 (d/D>0.8 ÷ d/D<0.4) | **293배** (HMM은 1.0배 상수) |
| BIC 상태개수 선택 | K=2 정확히 선택 |

GMM이 755회 전환한 것이 GMM을 실전 신호로 쓰지 않는 이유입니다.
Hazard 293배가 HSMM을 쓰는 이유입니다.

### 파이프라인 배선 (합성데이터 — 성과 해석 금지)

전 구간 오류 없이 완주, 85회 리밸런싱.

**발견된 치명적 결함과 수정:**

| | 수정 전 | 수정 후 |
|---|---|---|
| 월 회전율 | 1.15 | **0.45** |
| 리밸런싱당 비용 | 2.80% | **0.70%** |
| 연 비용 드래그 | 33.6% | **8.3%** |
| 보유종목 안정성 | 3~25개 요동 | 25개 고정 |

원인은 **랭킹 유니버스가 BOS 후보군으로만 한정된 것**이었습니다. 보유 종목이 회복되면
과매도가 풀려 BOS에서 빠지고, 랭킹이 없으니 버퍼 로직이 작동하지 못해 강제 전량교체가
발생했습니다. 랭킹 대상을 `BOS 후보 ∪ 현재 보유`로 바꾸어 해결했습니다.

이 결함은 실데이터로 갔으면 알파를 전부 먹었을 것입니다.

---

## 4. 실전 전환 전 필수 작업

### ★ 데이터 (가장 중요)

`SyntheticAdapter`는 배선 검증 전용입니다. 실전에는 아래 중 하나가 필요합니다.

| 소스 | 상폐종목 | PIT 재무 | 비용 | 판단 |
|---|---|---|---|---|
| Sharadar SEP/SF1 (Nasdaq Data Link) | ○ | ○ | ~$150/월 | **권장** |
| Polygon.io | ○ | △ | ~$200/월 | 가격은 우수, 재무 약함 |
| CRSP / Compustat | ○ | ○ | 기관가 | 최고 품질 |
| yfinance | **✗** | ✗ | 무료 | 백테스트 사용 금지 |

**중소형주에서 survivorship bias는 성과를 통째로 조작합니다.** 상폐된 종목이 유니버스에
없으면 "과매도 종목 매수" 전략의 백테스트는 반드시 거짓말이 됩니다 — 가장 크게 빠진 뒤
사라진 종목들이 전부 빠져 있기 때문입니다. `DataAdapter.assert_survivorship_safe()`를
구현하지 않은 어댑터로는 백테스트를 돌리지 마십시오.

### Agent 오염 진단

```python
bt.ablation()
```

| 관찰 | 해석 |
|---|---|
| CAGR만 크게 개선, MaxDD 변화 없음 | ⚠ **오염 의심** — Agent가 미래를 알고 있다 |
| MaxDD / P10_Month / WorstMonth 개선 | ✓ 정상 — tail risk 제거가 Agent의 본업 |
| 차이 미미 | · Agent 제거 검토 — API 비용만 나감 |

**수익률이 아니라 낙폭 지표를 보십시오.** 이것이 핵심입니다.

### 파라미터 튜닝 원칙

- `dd_gate = 0.60`은 임의값입니다. 0.4~0.8 구간에서 성과가 급변하면 과적합입니다
- 국면 샘플이 10회 미만이면 duration 분포 추정이 무의미합니다. 이 경우 HSMM을 포기하고
  HMM + 명시적 3개월 lock 규칙으로 대체하는 것이 정직합니다
- 모든 튜닝은 walk-forward로. in-sample 최적화 금지

---

## 5. 사용법

```python
from usquant.config import Config
from usquant.backtest.engine import Backtester
from usquant.agent.disqualifier import Disqualifier

cfg = Config()
adapter = SharadarAdapter(api_key=...)      # 직접 구현 필요
adapter.assert_survivorship_safe()          # 통과 못 하면 진행 금지

def call_claude(prompt: str) -> str:
    ...                                      # Anthropic API 호출

bt = Backtester(cfg, adapter, Disqualifier(cfg.agent, api_call=call_claude))

result = bt.run(use_agent=True)
print(result.summary())

diag = bt.ablation()                         # 오염 진단
print(diag, diag.attrs["verdict"])
```

---

## 6. 파일 구조

```
usquant/
├── config.py                     모든 튜닝 파라미터
├── data/
│   ├── base.py                   어댑터 인터페이스 + survivorship 가드
│   └── synthetic.py              배선 검증용 (실전 사용 금지)
├── regime/
│   ├── edhmm.py                  ★ HSMM 엔진 (검증 완료)
│   ├── features.py               causal 피처 + rolling 표준화
│   └── exposure.py               d/D 게이트 + vol-target 정책
├── screen/bos.py                 기술적 과매도 스크리너
├── agent/disqualifier.py         결격 제거기 (마스킹 + 캐시)
├── rank/factors.py               Value/Quality/Momentum 랭킹
├── portfolio/
│   ├── allocate.py               역변동성 + 유동성 제약
│   └── turnover_control.py       ★ 랭킹 버퍼 (필수)
└── backtest/engine.py            walk-forward + ablation
```

---

## 7. 다음 단계

1. **Sharadar 어댑터 구현** — 이게 없으면 나머지는 전부 무의미합니다
2. 실데이터로 EDHMM 재학습 → 미국 시장 국면 개수를 BIC로 재확인 (2개가 아닐 수 있음)
3. Agent 프롬프트를 실제 10-K/8-K로 튜닝, 판정 100건 수동 검증
4. Ablation 실행 → Agent 유지/제거 결정
5. 한국 시장 어댑터 추가 (DART API + KRX 상폐이력)

