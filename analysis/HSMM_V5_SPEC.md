# HSMM V5 사양 — newlow 축 bounded-influence 학습

`analysis/hsmm_final.py`(production) 에서 **M-step 의 평균·공분산 기여만** 교체한다.
피처·상태수·전이·duration·필터·노출 규칙은 전부 동일하다. production 무수정.

---

## 0. 왜 V5 인가 — V4(노이즈 성분)의 감사 결과

`analysis/hsmm_noise_audit.py` (2026-09-10) 결과:

- 노이즈 성분이 z>0.5 로 격리한 달은 **전체 newlow 상위 10개월 중 9개**였다.
  (2020-02 0.466 / 2022-09 0.301 / 2021-11 0.287 / 2020-01 0.250 / 2025-03 0.168 …)
- 즉 노이즈가 제거한 것은 "극단값"이 아니라 **newlow 위험축 그 자체**였다.
- 결과: Bear 상태의 newlow 평균 0.158 → 0.048, Bear 는 '완만한 약세' 전용으로 좁아지고
  Bull 평균이 +1.5σ 로 밀려나 **초강세 vs 나머지** 구도가 됐다(Bear 점유 77%, 리프트 1.04).
- 2026-05 newlow 스파이크(0.168, 상위 3%)에 무반응 → 익월 −22% 미탐지.
- γ(분산) 는 원인이 아니다. 분산비 bear/bull 최대 5.09, 2026-01 은 0.44 로 Bear 가 오히려 좁다.
  π_max 는 한 번도 걸리지 않았다.

**COVID 진단은 유효했다.** V4 에서 COVID 는 γ_bear=1.000 이면서 z=1.000 이었다 —
Bear 로 인식되면서 학습 기여만 0. 디코딩 P_bear 0.96 을 유지한 채 평균 오염만 차단했다.

## 0.1 기각된 대안 — z=0 강제(비대칭 완전 복원)

"newlow↑ & breadth↓ 관측은 노이즈로 보내지 않는다"는 규칙은 **실행하지 않는다.**
COVID(newlow 0.466, breadth 0.117)가 정확히 그 조건에 해당하므로 z=0 이 되어
학습 기여가 100% 로 되돌아가고, V4 가 해결한 평균 오염이 그대로 재발한다.
0 이냐 1 이냐의 이분법이 문제의 원인이다.

---

## 1. V5 의 원리 — 방향은 유지, 크기만 제한

newlow 의 Bear 방향 정보(부호)는 **전부 보존**하고, 극단적 크기가 상태 평균을 끌고 가는
영향력만 사전 고정 상한 `c` 로 자른다. 표준 bounded-influence(Huber M-estimator) 다.

M-step 에서 상태 `k` 의 평균 갱신 시, 현재 평균으로부터의 편차를 제한 축에서만 winsorize:

```
D_i      = x_i − μ_k^{(m)}
D̃_i,j   = clip(D_i,j, −c, +c)        j ∈ BOUNDED_AXES (기본 = {newlow})
D̃_i,j   = D_i,j                       그 외 축은 그대로

μ_k^{(m+1)} = μ_k^{(m)} + Σ_i r_i·D̃_i / Σ_i r_i          r_i = γ_ik · w_i
```

- **z=0/1 이분법이 아니다.** COVID 는 여전히 Bear 방향으로 기여하되 `c` 만큼만 기여한다.
- **부호가 살아있다.** newlow 스파이크는 계속 Bear 평균을 newlow↑ 쪽으로 민다.
  → V4 가 잃은 2026-05 신호가 회복될 수 있다.
- `c → ∞` 이면 production 과 수학적으로 동일하다(nesting 유지).
- 영향함수가 `c` 로 유계 → 창에 어떤 극단 관측치가 들어와도 평균 이동폭이 사전에 상한된다.

공분산도 같은 winsorized 편차로 계산하되, 축소편향을 상수 보정한다.

```
C_k = Σ_i r_i·(D̃_i/s) (D̃_i/s)ᵀ / Σ_i r_i
s_j = sqrt( E[clip(Z,−c,c)²] ),  Z~N(0,1)                # 제한 축만. 그 외 s_j = 1
    = sqrt( (2Φ(c)−1) − 2c·φ(c) + 2c²·(1−Φ(c)) )
```

이후 Ledoit-Wolf 축소·ridge 는 production 과 동일하게 적용한다.

## 2. 하이퍼파라미터

| 항목 | 값 |
|---|---|
| BOUNDED_AXES | `{newlow}` (기본). `--axes` 로 all 대조군 가능 |
| c (표준화 공간) | 그리드 1.0 / 1.5 / 2.0 / 3.0. c→∞ = production |
| 일관성 보정 | on (`--no-consistency` 로 해제 가능) |
| 그 외 전부 | production 과 동일 (SPEC §1~§10) |

## 3. 판정 순서 — ★ 진단 먼저, 성과는 그다음

성과부터 보면 상시 저노출의 부수효과에 속는다(t·V4 에서 두 번 반복됐다).
아래 4개 진단을 먼저 통과한 c 만 성과 평가로 넘긴다.

| # | 진단 | 기준 |
|---|---|---|
| 1 | COVID 이후 Bear 상태 newlow 평균 (2022-01·2023-01 재적합) | production 0.158/0.193 대비 낮아지되 **V4 의 0.048 처럼 소거되지 않을 것** (목표 0.07~0.12) |
| 2 | 2022 slow bear 탐지 (2021-10~2022-09 평균 P_bear) | production 0.129 → **≥ 0.5** |
| 3 | 2026-05 신호 (newlow 0.168 스파이크) | V4 0.021 → **≥ 0.4** (production 0.503) |
| 4 | Bear 점유율 / 분별력 | bear_ratio ≤ 45%, 리프트 ≥ 1.2, bear_ret_pos ≤ 0.45 |

4개 통과 시에만 성과 표(CAGR/Sharpe/MDD/Calmar + 동일 평균노출 상수노출 null)를 본다.
채택은 null 대조군 Sharpe·Calmar 동시 상회까지 확인한 뒤에 판단한다.

## 4. 산출

```
analysis/results/hsmm_v5_diag.csv     재적합별 진단 (Bear newlow 평균·점유율·평균이동)
analysis/results/hsmm_v5_metrics.csv  성과·분별력
analysis/results/hsmm_v5_path.csv     월별 P_bear / 노출
```
