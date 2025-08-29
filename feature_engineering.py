# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


def qqe_indicator(df: pd.DataFrame, rsi_period=14, smoothing_factor=5) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = RSIIndicator(close=df['close'], window=rsi_period).rsi()
    wilders_period = (rsi_period * 2) - 1
    df['rsi_ema'] = df['rsi'].ewm(alpha=1 / wilders_period, adjust=False).mean()
    df['atr_rsi'] = abs(df['rsi_ema'].diff())
    df['atr_rsi_ema'] = df['atr_rsi'].ewm(alpha=1 / wilders_period, adjust=False).mean()
    dar_factor = 4.236
    df['dar'] = df['atr_rsi_ema'] * dar_factor
    df['banda_superior'] = df['rsi_ema'] + df['dar']
    df['banda_inferior'] = df['rsi_ema'] - df['dar']
    df['qqe_line'] = np.nan
    prev_qqe_line = 0
    for i in range(1, len(df)):
        rsi_ema_atual = df.loc[i, 'rsi_ema']
        prev_qqe_line = df.loc[i - 1, 'qqe_line']
        if pd.isna(prev_qqe_line): prev_qqe_line = rsi_ema_atual
        new_qqe_line = prev_qqe_line
        if rsi_ema_atual > prev_qqe_line:
            new_qqe_line = max(df.loc[i, 'banda_inferior'], prev_qqe_line)
        elif rsi_ema_atual < prev_qqe_line:
            new_qqe_line = min(df.loc[i, 'banda_superior'], prev_qqe_line)
        df.loc[i, 'qqe_line'] = new_qqe_line
    df['qqe_signal'] = df['qqe_line'].ewm(span=smoothing_factor, adjust=False).mean()
    return df


def criar_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    df = df.copy()

    for window in [9, 21, 50, 200]:
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
    for window in [9, 21, 50, 200]:
        df[f'ema_{window}_slope'] = df[f'ema_{window}'].pct_change()

    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()

    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df['rsi_trend_adjusted'] = df['rsi'] * (df['close'] / df['ema_200'])

    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['normalized_volatility'] = df['atr'] / df['close']

    df['atr_acceleration'] = df['atr'].pct_change()

    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_hband'] = bb.bollinger_hband()
    df['bb_lband'] = bb.bollinger_lband()
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_pband'] = (df['close'] - df['bb_lband']) / (df['bb_hband'] - df['bb_lband'])

    candle_range = df['high'] - df['low']
    df['close_pos_in_range'] = (df['close'] - df['low']) / candle_range
    df['range_to_atr'] = candle_range / df['atr']

    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    df['volume_climax'] = (df['volume'] > 3 * df['volume_ma_20']).astype(int)

    for p in [1, 3, 5, 15]:
        df[f'return_{p}'] = df['close'].pct_change(p)

    df = qqe_indicator(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if drop_na:
        df = df.dropna().reset_index(drop=True)

    return df