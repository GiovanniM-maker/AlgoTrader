"""
Feature engineering pipeline for the algorithmic trading ML model.

Input  : Normalised OHLCV DataFrame  +  sentiment dict
Output : Feature matrix DataFrame ready for model training / inference

All NaN rows are dropped after feature construction.

Dependencies
------------
* pandas
* numpy
* pandas_ta           (pip install pandas_ta)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import pandas_ta; raise a clear error if missing
# ---------------------------------------------------------------------------

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.error(
        "pandas_ta is not installed. Install with: pip install pandas_ta"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentiment weights for the composite score
SENTIMENT_WEIGHTS: Dict[str, float] = {
    "fear_greed":       0.35,
    "cryptopanic":      0.40,
    "google_trends":    0.25,
}

# Lag periods (in candles) for return features
LAG_PERIODS = [1, 3, 6, 12, 24]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    sentiment: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix from OHLCV data and optional sentiment.

    Parameters
    ----------
    df:
        Normalised OHLCV DataFrame.  Must contain columns:
        open_time, open, high, low, close, volume, close_time.
        Should be sorted ascending by open_time.
    sentiment:
        Optional dict with any of these keys (values must be numeric):
            - ``fear_greed_value``       float in [0, 100]
            - ``cryptopanic_sentiment``  float in [-1, 1]
            - ``google_trends_score``    float in [0, 100]

    Returns
    -------
    pd.DataFrame
        Feature matrix with all engineered columns.  Rows with any NaN are
        dropped.  The index is reset.
    """
    if not PANDAS_TA_AVAILABLE:
        raise ImportError(
            "pandas_ta is required for feature engineering. "
            "Install with: pip install pandas_ta"
        )

    if df is None or df.empty:
        return pd.DataFrame()

    feat = df.copy()

    feat = _add_technical_features(feat)
    feat = _add_volume_features(feat)
    feat = _add_temporal_features(feat)
    feat = _add_lag_features(feat)
    feat = _add_sentiment_features(feat, sentiment or {})

    # Drop rows containing any NaN (from indicator warm-up periods)
    before = len(feat)
    feat = feat.dropna().reset_index(drop=True)
    after = len(feat)
    if before != after:
        logger.debug("build_features: dropped %d rows with NaN (warm-up).", before - after)

    return feat


class FeatureEngineer:
    """Wrapper for build_features with optional sentiment."""

    def compute_features(
        self,
        df: pd.DataFrame,
        sentiment: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Compute feature matrix from OHLCV data. Alias for build_features."""
        return build_features(df, sentiment=sentiment)


def create_target(
    df: pd.DataFrame,
    horizon_candles: int = 6,
    threshold: float = 0.01,
) -> pd.Series:
    """
    Compute the binary classification target variable.

    target_t = 1  if  close_{t+horizon} / close_t  > (1 + threshold)
    target_t = 0  otherwise

    Parameters
    ----------
    df:
        Feature-engineered DataFrame that includes a ``close`` column.
    horizon_candles:
        Number of candles ahead to look for the price move.
    threshold:
        Minimum upward return (e.g. 0.01 = 1 %) to label as positive.

    Returns
    -------
    pd.Series
        Binary integer Series (0 or 1), same index as *df*.
        The last *horizon_candles* rows will be NaN (no future data).
    """
    future_close = df["close"].shift(-horizon_candles)
    pct_change   = (future_close - df["close"]) / df["close"]
    target = (pct_change > threshold).astype("Int64")
    # Last horizon rows have no valid target
    target.iloc[-horizon_candles:] = pd.NA
    return target


# ---------------------------------------------------------------------------
# Internal – technical features
# ---------------------------------------------------------------------------


def _add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, Bollinger Bands, EMA, Ichimoku, VWAP dev, ATR, Stochastic."""

    # --- RSI ---
    df["rsi_7"]  = ta.rsi(df["close"], length=7)
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["rsi_21"] = ta.rsi(df["close"], length=21)

    # --- MACD ---
    macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd_result is not None and not macd_result.empty:
        df["macd_line"]   = macd_result.iloc[:, 0]   # MACD_12_26_9
        df["macd_signal"] = macd_result.iloc[:, 2]   # MACDs_12_26_9
        df["macd_hist"]   = macd_result.iloc[:, 1]   # MACDh_12_26_9
    else:
        df["macd_line"] = df["macd_signal"] = df["macd_hist"] = np.nan

    # --- Bollinger Bands ---
    bb_result = ta.bbands(df["close"], length=20, std=2.0)
    if bb_result is not None and not bb_result.empty:
        df["bb_lower"]     = bb_result.iloc[:, 0]  # BBL
        df["bb_middle"]    = bb_result.iloc[:, 1]  # BBM
        df["bb_upper"]     = bb_result.iloc[:, 2]  # BBU
        df["bb_bandwidth"] = bb_result.iloc[:, 3]  # BBB
        df["bb_pct_b"]     = bb_result.iloc[:, 4]  # BBP
    else:
        for col in ["bb_lower", "bb_middle", "bb_upper", "bb_bandwidth", "bb_pct_b"]:
            df[col] = np.nan

    # --- EMA ---
    for period in [9, 21, 50, 200]:
        col = f"ema_{period}"
        df[col] = ta.ema(df["close"], length=period)

    # --- Price position vs EMAs ---
    for period in [9, 21, 50]:
        ema_col = f"ema_{period}"
        df[f"close_vs_ema{period}"] = (df["close"] - df[ema_col]) / df[ema_col].replace(0, np.nan)

    # --- Ichimoku ---
    ichimoku_result = ta.ichimoku(df["high"], df["low"], df["close"])
    # pandas_ta returns a tuple of DataFrames: (ichimoku_df, forward_df)
    if ichimoku_result is not None:
        ich_df = ichimoku_result[0] if isinstance(ichimoku_result, (list, tuple)) else ichimoku_result
        if isinstance(ich_df, pd.DataFrame) and not ich_df.empty:
            cols = ich_df.columns.tolist()
            # Map to standard names based on column order: tenkan, kijun, span_a, span_b, chikou
            ich_map = {
                0: "ichimoku_tenkan",
                1: "ichimoku_kijun",
                2: "ichimoku_span_a",
                3: "ichimoku_span_b",
            }
            for idx, name in ich_map.items():
                if idx < len(cols):
                    df[name] = ich_df[cols[idx]].values[:len(df)]
                else:
                    df[name] = np.nan
        else:
            for name in ["ichimoku_tenkan", "ichimoku_kijun", "ichimoku_span_a", "ichimoku_span_b"]:
                df[name] = np.nan
    else:
        for name in ["ichimoku_tenkan", "ichimoku_kijun", "ichimoku_span_a", "ichimoku_span_b"]:
            df[name] = np.nan

    # --- VWAP deviation ---
    # VWAP is a session-level indicator; here we compute rolling VWAP
    # using a 20-period window as a proxy.
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol    = (typical_price * df["volume"]).rolling(20).sum()
    cum_vol       = df["volume"].rolling(20).sum().replace(0, np.nan)
    rolling_vwap  = cum_tp_vol / cum_vol
    df["vwap_deviation"] = (df["close"] - rolling_vwap) / rolling_vwap.replace(0, np.nan)

    # --- ATR ---
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14) / df["close"].replace(0, np.nan)
    df["atr_7"]  = ta.atr(df["high"], df["low"], df["close"], length=7)  / df["close"].replace(0, np.nan)

    # --- Stochastic ---
    stoch_result = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    if stoch_result is not None and not stoch_result.empty:
        df["stoch_k"] = stoch_result.iloc[:, 0]
        df["stoch_d"] = stoch_result.iloc[:, 1]
    else:
        df["stoch_k"] = df["stoch_d"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Internal – volume features
# ---------------------------------------------------------------------------


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume z-score, OBV, volume ratio, and price-volume trend."""

    vol = df["volume"]

    # Volume z-score (rolling 20)
    vol_mean = vol.rolling(20).mean()
    vol_std  = vol.rolling(20).std().replace(0, np.nan)
    df["volume_zscore"] = (vol - vol_mean) / vol_std

    # OBV
    obv = ta.obv(df["close"], df["volume"])
    df["obv"] = obv if obv is not None else np.nan

    # Volume ratio: current volume / 20-period EMA of volume
    vol_ema20 = ta.ema(vol, length=20)
    df["volume_ratio"] = vol / vol_ema20.replace(0, np.nan)

    # Price-volume trend (PVT)
    close_pct = df["close"].pct_change()
    df["pvt"] = (close_pct * vol).cumsum()

    return df


# ---------------------------------------------------------------------------
# Internal – temporal features
# ---------------------------------------------------------------------------


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day, day-of-week, weekend flag, and cyclical encodings."""

    # Convert open_time (Unix ms) to datetime for temporal extraction
    dt_index = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    df["hour_of_day"]  = dt_index.dt.hour.astype("float64")
    df["day_of_week"]  = dt_index.dt.dayofweek.astype("float64")
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype("float64")

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
    df["day_sin"]  = np.sin(2 * np.pi * df["day_of_week"]  / 7.0)
    df["day_cos"]  = np.cos(2 * np.pi * df["day_of_week"]  / 7.0)

    return df


# ---------------------------------------------------------------------------
# Internal – lag features
# ---------------------------------------------------------------------------


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged return features for recent price momentum."""

    for period in LAG_PERIODS:
        df[f"return_{period}"] = df["close"].pct_change(period)

    # Forward return for target creation (trainer expects future_return_1)
    df["future_return_1"] = df["close"].pct_change(-1)

    return df


# ---------------------------------------------------------------------------
# Internal – sentiment features
# ---------------------------------------------------------------------------


def _add_sentiment_features(
    df: pd.DataFrame,
    sentiment: Dict[str, Any],
) -> pd.DataFrame:
    """
    Join sentiment values to every row of the DataFrame.

    Sentinel values are used when a key is missing:
        - fear_greed_value:       50.0  (neutral)
        - cryptopanic_sentiment:   0.0  (neutral)
        - google_trends_score:    50.0  (neutral)
    """
    fg   = float(sentiment.get("fear_greed_value",      50.0))
    cp   = float(sentiment.get("cryptopanic_sentiment",   0.0))
    gt   = float(sentiment.get("google_trends_score",    50.0))

    df["fear_greed_value"]      = fg
    df["cryptopanic_sentiment"] = cp
    df["google_trends_score"]   = gt

    # Normalise all sources to [0, 100] for the composite
    fg_norm = max(0.0, min(100.0, fg))
    cp_norm = (max(-1.0, min(1.0, cp)) + 1.0) / 2.0 * 100.0
    gt_norm = max(0.0, min(100.0, gt))

    w = SENTIMENT_WEIGHTS
    composite = (
        w["fear_greed"]    * fg_norm
        + w["cryptopanic"] * cp_norm
        + w["google_trends"] * gt_norm
    )
    df["sentiment_composite"] = composite

    return df
