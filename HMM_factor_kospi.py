# 자동 변환: HMM_factor.ipynb → .py (코드 셀만 추출)
# KOSPI 버전: HMM 입력 = [ret_1m(1개월모멘텀), ret_6m(6개월모멘텀)] 2개.
#   - 자동탐색에서 모멘텀이 Bear 포착(recall~72%)·디리스킹 성과 1위 → 채택.
#   - (변동성/VIX/환율/SOX 추가는 모두 Bear recall 악화 → 제외. 모멘텀이 최선.)
#   - KOSPI 가격 파생이라 풀히스토리(1996~) → 외부 데이터 의존 없음.
#   - 2-state(Bull/Bear): Sideways 없이 매월 둘 중 하나 → bull_key/bear_key 전환에 직결, Bear recall↑.
#   - 예측: filtered(그 달까지 prefix의 predict_proba 마지막 행) → 미래 누수 없음.
#   - 평가: 워크포워드 + 다음달 채점. 학습 15년, OOS 2012~.

# ===== Cell 1 =====
import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm  # 진행 상황 표시

# --------------------------------------------------------------------------
# 1. 데이터 준비 (KOSPI 단일 시계열: 가격 → 실현변동성 / 모멘텀)
# --------------------------------------------------------------------------
# NOTE: 실현변동성은 KOSPI 자체 거래일 위에서 계산해야 한다. (VIX 등 다른 캘린더와 합치면
#       한쪽만 쉰 날에 NaN이 생겨 21일 rolling 결과가 거의 전부 NaN이 됨)
ks = yf.download('^KS11', start='1992-01-01', auto_adjust=False)
ks.columns = [c[0] if isinstance(c, tuple) else c for c in ks.columns]  # MultiIndex 평탄화

# 실현 변동성 (Garman-Klass 분산 → 21일 평균·연율화 → 마지막에 √ 한 번)
def garman_klass_variance(high, low, close, open):
    return 0.5 * np.log(high / low)**2 - (2 * np.log(2) - 1) * np.log(close / open)**2

ks['gk_var'] = garman_klass_variance(ks['High'], ks['Low'], ks['Close'], ks['Open'])
ks['realized_vol'] = np.sqrt(ks['gk_var'].rolling(window=21).mean() * 252)
ks.dropna(subset=['realized_vol'], inplace=True)

# 월별 리샘플링
monthly_df = ks.resample('ME').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'realized_vol': 'last'
})
monthly_df.rename(columns={'Close': 'sp500_Close', 'realized_vol': 'realized_volatility'}, inplace=True)

# ── 구리/금 비율 월말값 — yfinance (cg_chg 입력용, 경기·위험선호 선행, 2000~) ──
def _ym(ticker):
    d = yf.download(ticker, start='1990-01-01', auto_adjust=False, progress=False)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    return d['Close'].resample('ME').last()

_copper = _ym('HG=F')   # 구리 선물
_gold = _ym('GC=F')     # 금 선물
cg_m = _copper / _gold  # 구리/금 비율

# HMM 입력 / 매핑용 피처  (★ 입력 = [ret_3m, cg_chg] — 균형정확도 1위)
features = monthly_df.copy()
features['monthly_return'] = features['sp500_Close'].pct_change()        # 국면 이름 매핑용 (KOSPI)
features['ret_3m'] = features['sp500_Close'].pct_change(3)               # KOSPI 3개월 모멘텀
features['cg_chg'] = cg_m.pct_change(3)                                  # 구리/금 비율 3개월 변화율(경기·위험선호 선행)
features.dropna(inplace=True)

# ==============================================================================
# 2. 워크 포워드 검증 (Walk-Forward Validation)
# ==============================================================================
train_window_years = 15   # 전부 KOSPI 가격 파생이라 풀히스토리 → 15년 학습, OOS 2012~
test_window_months = 12   # 테스트(예측) 기간 (1년)
n_states = 3              # HMM 국면 개수 (Bull / Sideways / Bear 3개)
INPUT_COLS = ['ret_3m', 'cg_chg']  # ★ KOSPI 3개월 모멘텀 + 구리/금 비율 3개월변화
ACT_TH = 0.01             # 정답(다음달 수익률) 임계: >+1% Bull / <-1% Bear / 그사이 Sideways

out_of_sample_regimes = []
out_of_sample_conf = []   # 각 달의 filtered 확신도(최대 posterior)
start_index = train_window_years * 12
end_index = len(features)

print("워크 포워드 검증을 시작합니다...")
for i in tqdm(range(start_index, end_index, test_window_months)):
    # 1. 학습/테스트 분할 (확장 윈도우)
    train_features = features.iloc[:i]
    test_features = features.iloc[i : i + test_window_months]
    if len(test_features) == 0:
        break

    X_train = train_features[INPUT_COLS].values
    X_test = test_features[INPUT_COLS].values

    # 2-0. 표준화 (데이터 누수 방지: 학습 구간으로만 fit)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. HMM 학습 (매번 재학습)
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(X_train)

    # 3. 국면 해석 및 매핑 (학습구간 평균 월수익률 기준)
    train_hidden_states = model.predict(X_train)
    regime_stats = {}
    for state in range(n_states):
        state_mask = (train_hidden_states == state)
        regime_stats[state] = {'return': train_features[state_mask]['monthly_return'].mean()}

    # 상태 명명 (학습구간 평균 월수익률 rank 기반)
    _sorted = sorted(regime_stats.items(), key=lambda kv: kv[1]['return'])
    if n_states == 2:
        regime_map = {_sorted[0][0]: 'Bear Market', _sorted[-1][0]: 'Bull Market'}
    else:
        # 3-state: 최저=Bear, 중간=Sideways, 최고=Bull
        regime_map = {_sorted[0][0]: 'Bear Market', _sorted[1][0]: 'Sideways', _sorted[-1][0]: 'Bull Market'}

    # 4. Out-of-Sample 예측 및 저장
    #    ★ 전이행렬 1-step 예측: 현재 레짐 확률 → 다음 달 레짐 확률
    #      (a) filtered posterior  P(S_t | X_1:t) = predict_proba(t까지 prefix)의 마지막 행
    #      (b) 전이행렬 곱          P(S_{t+1} | X_1:t) = P(S_t|X_1:t) @ A   (A=model.transmat_)
    #      → 다음 달 분포의 argmax를 '다음 달 예측 레짐'으로. t 이후 관측 안 봄(누수 없음).
    oos_states = []
    oos_conf = []
    for j in range(len(test_features)):
        X_upto_t = np.vstack([X_train, X_test[:j + 1]])      # 학습구간 + test의 0..j월 (= t시점까지)
        filtered_post = model.predict_proba(X_upto_t)[-1]    # P(S_t | X_1:t) 현재 레짐 확률
        next_post = filtered_post @ model.transmat_          # P(S_{t+1} | X_1:t) 전이행렬로 다음달 예측
        oos_states.append(int(next_post.argmax()))           # 다음 달 예측 레짐(미래 관측 안 봄)
        oos_conf.append(float(next_post.max()))              # 다음 달 예측 확신도
    predicted_regimes = pd.Series(
        [regime_map[state] for state in oos_states],
        index=test_features.index, name='regime')
    out_of_sample_regimes.append(predicted_regimes)
    out_of_sample_conf.append(pd.Series(oos_conf, index=test_features.index, name='conf'))

walk_forward_results = pd.concat(out_of_sample_regimes)
features['walk_forward_regime'] = walk_forward_results
features['regime_conf'] = pd.concat(out_of_sample_conf)   # 확신도(최대 posterior)
print("\n워크 포워드 검증 완료!")

# ── 결과값 요약 출력 ──────────────────────────────────────────────
#    ★ 신호를 한 칸 미래에 적용: regime[t](t월 말 filtered 국면)을 t+1월 수익률로 평가한다.
valid = features.dropna(subset=['walk_forward_regime']).copy()
valid['fwd_return'] = valid['monthly_return'].shift(-1)   # regime[t] → 다음 달(t+1) 수익률로 채점
valid = valid.dropna(subset=['fwd_return'])               # 마지막 달(미래 수익률 없음) 제외
print(f"\nOOS 구간: {valid.index.min().date()} ~ {valid.index.max().date()} ({len(valid)}개월, 다음달 채점)")
print("\n[예측 레짐별 '다음 달' 실현 월수익률]")
print(f"{'레짐':<14}{'개월':>5}{'평균%':>9}{'표준편차%':>11}{'+비율':>8}")
for r in ['Bull Market', 'Sideways', 'Bear Market']:
    x = valid[valid['walk_forward_regime'] == r]['fwd_return']
    if len(x):
        print(f"{r:<14}{len(x):>5}{x.mean()*100:>9.2f}{x.std()*100:>11.2f}{(x>0).mean()*100:>7.0f}%")

# ── 정답 = 다음 달 수익률 ±1% 3분류 (>+1% Bull / <-1% Bear / 그사이 Sideways) ──
def actual_regime(ret):
    return 'Bull Market' if ret > ACT_TH else ('Bear Market' if ret < -ACT_TH else 'Sideways')
valid['actual_regime'] = valid['fwd_return'].apply(actual_regime)
correct = (valid['walk_forward_regime'] == valid['actual_regime'])

print(f"\n[3-way 적중률]  정답 = 다음달 수익률 (>+1% Bull / <-1% Bear / 그사이 Sideways)")
print(f"{'예측레짐':<14}{'개월':>5}{'적중':>6}{'적중률':>9}")
for r in ['Bull Market', 'Sideways', 'Bear Market']:
    m = valid['walk_forward_regime'] == r
    n = int(m.sum()); hit = int((m & correct).sum())
    if n:
        print(f"{r:<14}{n:>5}{hit:>6}{hit/n*100:>8.1f}%")
print(f"{'전체':<14}{len(valid):>5}{int(correct.sum()):>6}{correct.mean()*100:>8.1f}%")

# ── 클래스별 Recall (실제 레짐 중 올바르게 예측한 비율) ──
print("\n[클래스별 Recall]  실제 레짐 → 같은 레짐으로 예측한 비율")
for r in ['Bull Market', 'Sideways', 'Bear Market']:
    m = valid['actual_regime'] == r
    n = int(m.sum()); hit = int((m & (valid['walk_forward_regime'] == r)).sum())
    if n:
        print(f"  실제 {r:<13} {hit:>3}/{n:<3} = {hit/n*100:5.1f}%")

# ── Bear precision + 3x3 혼동행렬 ──
_pb = valid['walk_forward_regime'] == 'Bear Market'; _ab = valid['actual_regime'] == 'Bear Market'
_tp = int((_pb & _ab).sum()); _fp = int((_pb & ~_ab).sum())
print(f"\n[Bear precision] {_tp}/{_tp+_fp} = {(_tp/(_tp+_fp)*100 if _tp+_fp else 0):.1f}%  (Bear 예측 중 실제 Bear)")
_labels = ['Bull Market', 'Sideways', 'Bear Market']
print("\n[혼동행렬]  행=예측 / 열=정답")
_hdr = '예측|정답'
print(f"{_hdr:<12}" + ''.join(f"{l.split()[0]:>9}" for l in _labels))
for p in _labels:
    row = [int(((valid['walk_forward_regime'] == p) & (valid['actual_regime'] == a)).sum()) for a in _labels]
    print(f"{p.split()[0]:<12}" + ''.join(f"{c:>9}" for c in row))

# ── 확신도 게이팅 (3-way): 확신한 달만 채점 ──
va = valid.copy()
print("\n[확신도 게이팅]  P(예측국면) >= 임계인 달만 3-way 채점")
print(f"{'임계':>6}{'해당월':>7}{'비중':>7}{'적중률':>8}")
for th in [0.40, 0.50, 0.60, 0.70, 0.80]:
    sub = va[va['regime_conf'] >= th]
    if len(sub) == 0:
        continue
    acc = (sub['walk_forward_regime'] == sub['actual_regime']).mean() * 100
    print(f"{th:>6.2f}{len(sub):>7}{len(sub)/len(va)*100:>6.0f}%{acc:>7.1f}%")

# ===== Cell 2 =====
# ==============================================================================
# 3. 결과 시각화
# ==============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(20, 10))

# 전체 기간 KOSPI 지수 플롯
ax.plot(features.index, features['sp500_Close'], label='KOSPI 종가', color='black', linewidth=1.5)

# 워크 포워드 예측 국면에 따라 배경색 칠하기
regime_colors = {'Bear Market': 'red', 'Sideways': 'orange', 'Bull Market': 'green'}
valid_regimes = features.dropna(subset=['walk_forward_regime'])

for i in range(len(valid_regimes)):
    regime = valid_regimes['walk_forward_regime'].iloc[i]
    color = regime_colors[regime]
    start_date = valid_regimes.index[i]
    end_date = valid_regimes.index[i+1] if i + 1 < len(valid_regimes) else valid_regimes.index[i] + pd.DateOffset(months=1)
    ax.axvspan(start_date, end_date, color=color, alpha=0.3)

ax.set_title('HMM 시장 국면 분석 - KOSPI [3-state, 입력: ret_3m + cg_chg(구리/금), 정답±1%, 전이행렬 1-step] (Walk-Forward OOS)', fontsize=14)
ax.set_xlabel('날짜', fontsize=15)
ax.set_ylabel('KOSPI 지수', fontsize=15)
ax.set_yscale('log')  # 로그 스케일

handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.3) for name, color in regime_colors.items()]
labels = regime_colors.keys()
ax.legend(handles, labels, title="시장 국면", fontsize=12)

ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y년'))
plt.grid(True)
plt.savefig('analysis/results/HMM_factor_kospi_regime.png', dpi=120, bbox_inches='tight')  # 결과 그래프 저장
plt.show()
