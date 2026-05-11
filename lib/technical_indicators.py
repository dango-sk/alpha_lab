"""
기술적 지표 계산 모듈: AI 종목 필터에서 사용.

daily_price 테이블의 가격/거래량 데이터로 모든 지표를 직접 계산한다.
"""
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════
# 추세 / 모멘텀
# ═══════════════════════════════════════════════════════

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index). 0~100, 70 이상 과매수, 30 이하 과매도."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD, Signal, Histogram 반환."""
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3):
    """Stochastic %K, %D. 80 이상 과매수, 20 이하 과매도."""
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX (Average Directional Index). 25 이상이면 추세 강함."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period).mean()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """CCI (Commodity Channel Index). +100 이상 과매수, -100 이하 과매도."""
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md).replace(0, np.nan)


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    """ROC (Rate of Change). %단위."""
    return 100 * (close - close.shift(period)) / close.shift(period).replace(0, np.nan)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R. -80 이하 과매도, -20 이상 과매수."""
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def moving_average_gap(close: pd.Series, windows: list[int] = None) -> dict[str, pd.Series]:
    """이동평균 이격도 (%). windows 기본 [5, 20, 60, 120, 200]."""
    if windows is None:
        windows = [5, 20, 60, 120, 200]
    result = {}
    for w in windows:
        ma = close.rolling(w).mean()
        result[f"ma{w}_gap"] = 100 * (close - ma) / ma.replace(0, np.nan)
    return result


# ═══════════════════════════════════════════════════════
# 변동성
# ═══════════════════════════════════════════════════════

def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """볼린저밴드: upper, middle, lower, %b(위치), bandwidth."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / middle.replace(0, np.nan) * 100
    return upper, middle, lower, pct_b, bandwidth


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR (Average True Range)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """역사적 변동성 (연환산 %)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252) * 100


def keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                    ema_period: int = 20, atr_period: int = 14, multiplier: float = 2.0):
    """Keltner Channel: upper, middle, lower."""
    middle = close.ewm(span=ema_period, min_periods=ema_period).mean()
    atr_val = atr(high, low, close, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower


# ═══════════════════════════════════════════════════════
# 거래량
# ═══════════════════════════════════════════════════════

def volume_change_ratio(volume: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """거래량 변화율: 단기 평균 / 장기 평균."""
    return volume.rolling(short).mean() / volume.rolling(long).mean().replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV (On Balance Volume)."""
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 14) -> pd.Series:
    """MFI (Money Flow Index). RSI의 거래량 가중 버전. 0~100."""
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0).rolling(period).sum()
    neg_mf = mf.where(delta <= 0, 0.0).rolling(period).sum()
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def ad_line(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series) -> pd.Series:
    """A/D Line (Accumulation/Distribution)."""
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (clv.fillna(0) * volume).cumsum()


def cmf(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 20) -> pd.Series:
    """CMF (Chaikin Money Flow). -1 ~ +1."""
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (clv.fillna(0) * volume).rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


# ═══════════════════════════════════════════════════════
# 가격 위치
# ═══════════════════════════════════════════════════════

def week52_high_ratio(close: pd.Series) -> pd.Series:
    """52주(252일) 고점 대비 현재가 비율 (%). 100이면 신고가."""
    high_252 = close.rolling(252).max()
    return 100 * close / high_252.replace(0, np.nan)


def week52_low_ratio(close: pd.Series) -> pd.Series:
    """52주(252일) 저점 대비 현재가 비율 (%). 100이면 저점."""
    low_252 = close.rolling(252).min()
    return 100 * close / low_252.replace(0, np.nan)


def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series):
    """Pivot Point + 지지/저항선 (전일 기준)."""
    pp = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    r1 = 2 * pp - low.shift(1)
    s1 = 2 * pp - high.shift(1)
    r2 = pp + (high.shift(1) - low.shift(1))
    s2 = pp - (high.shift(1) - low.shift(1))
    return pp, r1, s1, r2, s2


def fibonacci_retracement(close: pd.Series, period: int = 252) -> dict[str, pd.Series]:
    """Fibonacci 되돌림 수준 (0%, 23.6%, 38.2%, 50%, 61.8%, 100%)."""
    high = close.rolling(period).max()
    low = close.rolling(period).min()
    diff = high - low
    levels = {
        "fib_0": low,
        "fib_236": low + diff * 0.236,
        "fib_382": low + diff * 0.382,
        "fib_500": low + diff * 0.500,
        "fib_618": low + diff * 0.618,
        "fib_100": high,
    }
    # 현재가가 어떤 레벨에 있는지 (0~1 스케일)
    levels["fib_position"] = (close - low) / diff.replace(0, np.nan)
    return levels


# ═══════════════════════════════════════════════════════
# 패턴 / 구조
# ═══════════════════════════════════════════════════════

def cross_signals(close: pd.Series) -> dict[str, pd.Series]:
    """골든크로스/데드크로스 시그널. 1=골든, -1=데드, 0=없음."""
    pairs = [(5, 20), (20, 60), (60, 120)]
    result = {}
    for short, long in pairs:
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        cross = pd.Series(0, index=close.index)
        cross[(ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))] = 1   # 골든
        cross[(ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))] = -1  # 데드
        result[f"cross_{short}_{long}"] = cross
    return result


def new_high_low(close: pd.Series, period: int = 252) -> dict[str, pd.Series]:
    """신고가/신저가 여부. 1=해당, 0=아님."""
    return {
        "new_high": (close >= close.rolling(period).max()).astype(int),
        "new_low": (close <= close.rolling(period).min()).astype(int),
    }


def consecutive_days(close: pd.Series) -> pd.Series:
    """연속 상승/하락 일수. 양수=연속상승, 음수=연속하락."""
    direction = np.sign(close.diff())
    groups = (direction != direction.shift(1)).cumsum()
    counts = direction.groupby(groups).cumcount() + 1
    return counts * direction


def candle_patterns(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series) -> dict[str, pd.Series]:
    """기본 캔들 패턴: 도지, 망치형, 장악형."""
    body = (close - open_).abs()
    total_range = (high - low).replace(0, np.nan)
    upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low

    # 도지: 몸통이 전체 범위의 10% 이하
    doji = (body / total_range < 0.1).astype(int)

    # 망치형: 아래꼬리가 몸통의 2배 이상, 윗꼬리 짧음
    hammer = ((lower_shadow > body * 2) & (upper_shadow < body * 0.5)).astype(int)

    # 장악형(불리시): 전일 음봉 → 오늘 양봉이 전일 몸통 완전히 감싸기
    prev_bearish = close.shift(1) < open_.shift(1)
    curr_bullish = close > open_
    bullish_engulf = (
        prev_bearish & curr_bullish &
        (open_ <= close.shift(1)) & (close >= open_.shift(1))
    ).astype(int)

    return {
        "doji": doji,
        "hammer": hammer,
        "bullish_engulfing": bullish_engulf,
    }


# ═══════════════════════════════════════════════════════
# 통합 계산 함수
# ═══════════════════════════════════════════════════════

def calc_all_indicators(df: pd.DataFrame) -> dict[str, float]:
    """단일 종목의 OHLCV DataFrame → 최신 시점 기술적 지표 딕셔너리.

    Parameters
    ----------
    df : DataFrame
        columns: ['close', 'high', 'low', 'volume'] (+ optional 'open')
        index: DatetimeIndex 또는 trade_date 정렬 상태

    Returns
    -------
    dict : {지표명: 최신값}
    """
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    o = df["open"] if "open" in df.columns else c.shift(1)  # open 없으면 전일종가로 대체

    result = {}

    # 추세/모멘텀
    result["rsi_14"] = rsi(c, 14).iloc[-1]
    macd_l, macd_s, macd_h = macd(c)
    result["macd"] = macd_l.iloc[-1]
    result["macd_signal"] = macd_s.iloc[-1]
    result["macd_histogram"] = macd_h.iloc[-1]
    k, d = stochastic(h, l, c)
    result["stoch_k"] = k.iloc[-1]
    result["stoch_d"] = d.iloc[-1]
    result["adx_14"] = adx(h, l, c, 14).iloc[-1]
    result["cci_20"] = cci(h, l, c, 20).iloc[-1]
    result["roc_12"] = roc(c, 12).iloc[-1]
    result["williams_r_14"] = williams_r(h, l, c, 14).iloc[-1]

    for key, val in moving_average_gap(c).items():
        result[key] = val.iloc[-1]

    # 변동성
    _, _, _, pct_b, bw = bollinger_bands(c)
    result["bb_pct_b"] = pct_b.iloc[-1]
    result["bb_bandwidth"] = bw.iloc[-1]
    result["atr_14"] = atr(h, l, c, 14).iloc[-1]
    result["hv_20"] = historical_volatility(c, 20).iloc[-1]

    # 거래량
    result["vol_change_ratio"] = volume_change_ratio(v).iloc[-1]
    result["obv_slope"] = _slope(obv(c, v), 20)
    result["mfi_14"] = mfi(h, l, c, v, 14).iloc[-1]
    result["cmf_20"] = cmf(h, l, c, v, 20).iloc[-1]

    # 가격 위치
    result["week52_high_ratio"] = week52_high_ratio(c).iloc[-1]
    result["week52_low_ratio"] = week52_low_ratio(c).iloc[-1]
    fib = fibonacci_retracement(c)
    result["fib_position"] = fib["fib_position"].iloc[-1]

    # 패턴
    for key, val in cross_signals(c).items():
        # 최근 5일 내 시그널 있으면 반환
        result[key] = val.iloc[-5:].sum()
    nh_nl = new_high_low(c)
    result["new_high"] = nh_nl["new_high"].iloc[-1]
    result["new_low"] = nh_nl["new_low"].iloc[-1]
    result["consecutive_days"] = consecutive_days(c).iloc[-1]
    candles = candle_patterns(o, h, l, c)
    for key, val in candles.items():
        result[key] = val.iloc[-3:].sum()  # 최근 3일 내 패턴

    # NaN → None
    return {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in result.items()}


def _slope(series: pd.Series, period: int = 20) -> float:
    """최근 N일 기울기 (선형회귀 계수). 방향 판단용."""
    tail = series.dropna().tail(period)
    if len(tail) < period:
        return np.nan
    x = np.arange(len(tail))
    coeffs = np.polyfit(x, tail.values, 1)
    return coeffs[0]
