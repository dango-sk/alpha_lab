# 자동 변환: HMM_factor.ipynb → .py (코드 셀만 추출)
# KOSPI 버전: 지수만 한국 시장(^KS11)으로 교체, 변동성지수는 원본처럼 미국 VIX(^VIX) 사용.
# 그 외 분석 로직/시각화는 원본(HMM_factor.py)을 그대로 사용.

# ===== Cell 1 =====
import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm # 진행 상황을 보여주기 위한 라이브러리

# --------------------------------------------------------------------------
# 1. 데이터 준비 (한국 시장: KOSPI / 미국 VIX)
# --------------------------------------------------------------------------
# NOTE: 지수만 KOSPI(^KS11)로 교체하고, 변동성지수는 원본처럼 미국 VIX(^VIX)를 그대로 사용.
#       (KOSPI는 글로벌 위험선호에 강하게 연동되므로 VIX를 위험-레짐 대리변수로 사용)
#       이후 분석 코드는 원본 그대로 두기 위해 내부 컬럼명은 원본과 동일하게(sp500_*, vix_*) 유지한다.
#       (실제로는 sp500_* = KOSPI, vix_* = 미국 VIX 데이터가 담긴다.)
tickers = ['^KS11', '^VIX']
df = yf.download(tickers, start='1992-01-01')
df = df.swaplevel(axis='columns')
df.columns = [f"{col[1]}_{col[0].replace('^', '')}" for col in df.columns]
cols_to_drop = [col for col in df.columns if 'Adj Close' in col or 'Volume' in col]
df.drop(columns=cols_to_drop, inplace=True)

# 실현 변동성 계산
# Garman-Klass는 본질적으로 '분산(variance)' 추정량이므로,
# 일별 분산을 21일 평균·연율화(×252)한 뒤 마지막에 한 번만 √를 취한다.
def garman_klass_variance(high, low, close, open):
    return 0.5 * np.log(high / low)**2 - (2 * np.log(2) - 1) * np.log(close / open)**2

df['gk_var_KS11'] = garman_klass_variance(df['High_KS11'], df['Low_KS11'], df['Close_KS11'], df['Open_KS11'])
df['realized_vol_KS11'] = np.sqrt(df['gk_var_KS11'].rolling(window=21).mean() * 252)
df.dropna(inplace=True)

# 월별 리샘플링
agg_rules = {
    'Open_KS11': 'first', 'High_KS11': 'max', 'Low_KS11': 'min', 'Close_KS11': 'last', 'realized_vol_KS11': 'last',
    'Open_VIX': 'first', 'High_VIX': 'max', 'Low_VIX': 'min', 'Close_VIX': 'last'
}
monthly_df = df.resample('M').agg(agg_rules)
final_column_names = {
    'Open_KS11': 'sp500_Open', 'High_KS11': 'sp500_High', 'Low_KS11': 'sp500_Low', 'Close_KS11': 'sp500_Close', 'realized_vol_KS11': 'sp500_realized_vol',
    'Open_VIX': 'vix_Open', 'High_VIX': 'vix_High', 'Low_VIX': 'vix_Low', 'Close_VIX': 'vix_Close'
}
monthly_df.rename(columns=final_column_names, inplace=True)
monthly_df['vix_level'] = yf.download('^VIX', start='1990-01-01')['Close'].resample('M').last()


# HMM 모델 입력을 위한 데이터 준비
features = monthly_df.copy()
features['monthly_return'] = features['sp500_Close'].pct_change()
features.rename(columns={'sp500_realized_vol': 'realized_volatility'}, inplace=True)
features.dropna(inplace=True)

# ==============================================================================
# 2. 워크 포워드 검증 (Walk-Forward Validation) - 핵심 변경 부분
# ==============================================================================

# 워크 포워드 파라미터 설정
train_window_years = 15  # 초기 학습 기간 (15년)
test_window_months = 12  # 테스트(예측) 기간 (1년)
n_states = 3             # HMM 국면 개수

# Out-of-Sample 예측 결과를 저장할 리스트
out_of_sample_regimes = []

# 전체 데이터 기간을 test_window_months 단위로 반복
start_index = train_window_years * 12
end_index = len(features)

print("워크 포워드 검증을 시작합니다...")
for i in tqdm(range(start_index, end_index, test_window_months)):
    # 1. 학습 및 테스트 데이터 분할 (확장 윈도우 방식)
    train_features = features.iloc[:i]
    test_features = features.iloc[i : i + test_window_months]

    if len(test_features) == 0:
        break

    X_train = train_features[['monthly_return', 'realized_volatility', 'vix_level']].values
    X_test = test_features[['monthly_return', 'realized_volatility', 'vix_level']].values

    # 2-0. 표준화 (데이터 누수 방지: 학습 구간으로만 fit, 테스트는 transform만)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. HMM 모델 학습 (매번 새로운 데이터로 재학습)
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(X_train)

    # 3. 국면 해석 및 매핑 (매우 중요!)
    # HMM은 상태(0,1,2)를 임의로 할당하므로, 매번 재학습 시 의미를 찾아줘야 함
    # 학습 데이터(train_features)를 기반으로 각 국면의 특성을 분석
    train_hidden_states = model.predict(X_train)
    regime_stats = {}
    for state in range(n_states):
        state_mask = (train_hidden_states == state)
        # 평균 수익률을 기준으로 국면을 식별
        regime_stats[state] = {'return': train_features[state_mask]['monthly_return'].mean()}

    # 평균 수익률 순서대로 정렬하여 '하락장', '횡보장', '상승장'으로 매핑
    sorted_by_return = sorted(regime_stats.items(), key=lambda item: item[1]['return'])
    regime_map = {
        sorted_by_return[0][0]: 'Bear Market',
        sorted_by_return[1][0]: 'Sideways',
        sorted_by_return[-1][0]: 'Bull Market'
    }

    # 4. Out-of-Sample 국면 예측 및 결과 저장
    out_of_sample_states = model.predict(X_test)
    # 예측된 숫자(0,1,2)를 해석된 국면 이름으로 변환하여 저장
    predicted_regimes = pd.Series([regime_map[state] for state in out_of_sample_states], index=test_features.index, name='regime')
    out_of_sample_regimes.append(predicted_regimes)

# 모든 Out-of-Sample 예측 결과를 하나로 합치기
walk_forward_results = pd.concat(out_of_sample_regimes)

# 원본 데이터프레임에 워크 포워드 결과 합치기
features['walk_forward_regime'] = walk_forward_results
print("\n워크 포워드 검증 완료!")

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
# walk_forward_regime 열에 값이 있는(NaN이 아닌) 기간만 시각화
valid_regimes = features.dropna(subset=['walk_forward_regime'])

for i in range(len(valid_regimes)):
    regime = valid_regimes['walk_forward_regime'].iloc[i]
    color = regime_colors[regime]
    start_date = valid_regimes.index[i]
    # 다음 데이터 포인트까지 또는 마지막 데이터 포인트인 경우 한 달 뒤까지
    end_date = valid_regimes.index[i+1] if i + 1 < len(valid_regimes) else valid_regimes.index[i] + pd.DateOffset(months=1)
    ax.axvspan(start_date, end_date, color=color, alpha=0.3)

ax.set_title('HMM 시장 국면 분석 - KOSPI (Walk-Forward Out-of-Sample 검증 결과)', fontsize=20)
ax.set_xlabel('날짜', fontsize=15)
ax.set_ylabel('KOSPI 지수', fontsize=15)
ax.set_yscale('log') # 로그 스케일로 변경하여 장기 추세를 더 잘보이게 함

# 범례 설정
handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.3) for name, color in regime_colors.items()]
labels = regime_colors.keys()
ax.legend(handles, labels, title="시장 국면", fontsize=12)

ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y년'))
plt.grid(True)
plt.show()
