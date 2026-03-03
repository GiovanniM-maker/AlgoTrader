"""
AlgoTrader Pro - Configuration Loader
======================================
Loads settings.yaml, merges environment variable overrides from .env,
validates with Pydantic v2 models, and exposes a thread-safe singleton
via get_config().
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/utils -> src -> project root
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
_ENV_PATH = _PROJECT_ROOT / ".env"


# ---------------------------------------------------------------------------
# Pydantic Models - bottom-up (leaf models first)
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    name: str = "AlgoTrader Pro"
    version: str = "1.0.0"
    environment: str = "paper"
    log_level: str = "INFO"
    timezone: str = "UTC"
    debug: bool = False

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"paper", "live", "backtest", "test"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got '{v}'")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///./database/algotrader.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


class CoinGeckoConfig(BaseModel):
    base_url: str = "https://api.coingecko.com/api/v3"
    symbols: List[str] = ["bitcoin", "ethereum", "solana", "binancecoin"]
    vs_currency: str = "usd"
    rate_limit_calls: int = 10
    rate_limit_window: int = 60
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    timeout: int = 30


class CryptoPanicConfig(BaseModel):
    base_url: str = "https://cryptopanic.com/api/developer/v2"
    filter: str = "hot"
    kind: str = "news"
    regions: str = "en"
    items_per_request: int = 20
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    timeout: int = 30


class FearGreedConfig(BaseModel):
    base_url: str = "https://api.alternative.me/fng/"
    limit: int = 1
    date_format: str = "us"
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    timeout: int = 30


class DataProvidersConfig(BaseModel):
    coingecko: CoinGeckoConfig = Field(default_factory=CoinGeckoConfig)
    cryptopanic: CryptoPanicConfig = Field(default_factory=CryptoPanicConfig)
    fear_greed: FearGreedConfig = Field(default_factory=FearGreedConfig)


class TimeframesConfig(BaseModel):
    primary: str = "1h"
    secondary: List[str] = ["15m", "4h", "1d"]


class PaperTradingConfig(BaseModel):
    initial_capital: float = 10000.0
    currency: str = "USDT"
    simulate_slippage: bool = True
    simulate_fees: bool = True
    default_fee_rate: float = 0.001
    order_fill_delay_ms: int = 500

    @field_validator("initial_capital")
    @classmethod
    def validate_capital(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("initial_capital must be positive")
        return v


class LayerWeightsConfig(BaseModel):
    layer1_technical: float = 0.30
    layer2_volume: float = 0.25
    layer3_sentiment: float = 0.15
    layer4_ml: float = 0.30

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "LayerWeightsConfig":
        total = (
            self.layer1_technical
            + self.layer2_volume
            + self.layer3_sentiment
            + self.layer4_ml
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"layer_weights must sum to 1.0, got {total:.6f}"
            )
        return self


class StrategyConfig(BaseModel):
    name: str = "HybridV1"
    description: str = ""
    min_confidence_long: float = 65.0
    min_confidence_short: float = 70.0
    max_concurrent_positions: int = 3
    max_portfolio_risk_pct: float = 20.0
    allow_short: bool = False
    layer_weights: LayerWeightsConfig = Field(default_factory=LayerWeightsConfig)
    combination_method: str = "weighted_average"
    trade_cooldown_minutes: int = 60
    min_layer_agreement: float = 0.6


class TrailingStopConfig(BaseModel):
    enabled: bool = True
    type: str = "atr"
    atr_multiplier: float = 1.5
    percentage: float = 2.0


class TimeExitConfig(BaseModel):
    enabled: bool = True
    max_hold_hours: int = 48
    min_profit_to_hold_pct: float = 0.5


class ExpectedValueConfig(BaseModel):
    auto_pause_if_negative: bool = True
    rolling_window_trades: int = 30
    min_trades_to_evaluate: int = 10


class TakeProfitConfig(BaseModel):
    enabled: bool = True
    risk_reward_ratio: float = 2.5


class RiskConfig(BaseModel):
    kelly_fraction_multiplier: float = 0.5
    kelly_min_fraction: float = 0.01
    kelly_max_fraction: float = 0.15
    kelly_rolling_window_trades: int = 50
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    atr_high_volatility_multiplier: float = 3.0
    volatility_threshold_pct: float = 5.0
    trailing_stop: TrailingStopConfig = Field(default_factory=TrailingStopConfig)
    time_exit: TimeExitConfig = Field(default_factory=TimeExitConfig)
    expected_value: ExpectedValueConfig = Field(default_factory=ExpectedValueConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    max_drawdown_pause_pct: float = 15.0
    daily_loss_limit_pct: float = 5.0
    correlation_threshold: float = 0.8


class EnsembleWeightsConfig(BaseModel):
    xgboost: float = 0.40
    random_forest: float = 0.30
    lstm: float = 0.30

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "EnsembleWeightsConfig":
        total = self.xgboost + self.random_forest + self.lstm
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ensemble_weights must sum to 1.0, got {total:.6f}")
        return self


class XGBoostConfig(BaseModel):
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    use_label_encoder: bool = False
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 50
    n_jobs: int = -1


class RandomForestConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int = 10
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: str = "sqrt"
    n_jobs: int = -1
    random_state: int = 42
    class_weight: str = "balanced"


class LSTMConfig(BaseModel):
    lookback_candles: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    patience: int = 10
    validation_split: float = 0.2


class FeatureEngineeringConfig(BaseModel):
    target_horizon_candles: int = 6
    target_return_threshold: float = 0.01
    normalize_features: bool = True
    normalization_method: str = "zscore"


class RetrainingConfig(BaseModel):
    enabled: bool = True
    interval_days: int = 30
    min_samples: int = 2000
    retrain_on_performance_drop: bool = True
    performance_drop_threshold: float = 0.05


class MLConfig(BaseModel):
    ensemble_weights: EnsembleWeightsConfig = Field(default_factory=EnsembleWeightsConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)
    model_dir: str = "./models"
    save_best_only: bool = True


class MonteCarloConfig(BaseModel):
    n_simulations: int = 1000
    confidence_intervals: List[int] = [5, 25, 50, 75, 95]
    random_seed: int = 42


class WalkForwardConfig(BaseModel):
    enabled: bool = True
    in_sample_months: int = 12
    out_sample_months: int = 3


class BacktestingConfig(BaseModel):
    default_start: str = "2022-01-01"
    default_end: str = "2024-01-01"
    commission: float = 0.001
    slippage_std_pct: float = 0.0005
    initial_capital: float = 10000.0
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    results_dir: str = "./backtests"
    save_trades: bool = True
    generate_report: bool = True


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    cors_origins: List[str] = ["http://localhost:3000"]
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


class RSISignalConfig(BaseModel):
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    extreme_oversold: float = 20.0
    extreme_overbought: float = 80.0
    signal_strength_oversold: float = 0.8
    signal_strength_overbought: float = 0.8
    signal_strength_neutral: float = 0.0


class MACDSignalConfig(BaseModel):
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    histogram_threshold: float = 0.0
    use_zero_crossover: bool = True


class BollingerSignalConfig(BaseModel):
    period: int = 20
    std_dev: float = 2.0
    squeeze_threshold: float = 0.1
    use_bandwidth: bool = True


class EMACrossoverConfig(BaseModel):
    fast_period: int = 9
    medium_period: int = 21
    slow_period: int = 50
    trend_confirmation_candles: int = 2


class VolumeAnomalyConfig(BaseModel):
    zscore_threshold: float = 2.5
    lookback_period: int = 20
    volume_confirm_price: bool = True
    price_move_threshold: float = 0.005


class StochRSIConfig(BaseModel):
    rsi_period: int = 14
    stoch_period: int = 14
    smooth_k: int = 3
    smooth_d: int = 3
    oversold: float = 20.0
    overbought: float = 80.0


class ATRSignalConfig(BaseModel):
    period: int = 14


class OBVConfig(BaseModel):
    ema_period: int = 20


class VWAPConfig(BaseModel):
    enabled: bool = True
    deviation_bands: List[float] = [1.0, 2.0]


class SignalsConfig(BaseModel):
    rsi: RSISignalConfig = Field(default_factory=RSISignalConfig)
    macd: MACDSignalConfig = Field(default_factory=MACDSignalConfig)
    bollinger: BollingerSignalConfig = Field(default_factory=BollingerSignalConfig)
    ema_crossover: EMACrossoverConfig = Field(default_factory=EMACrossoverConfig)
    volume_anomaly: VolumeAnomalyConfig = Field(default_factory=VolumeAnomalyConfig)
    stoch_rsi: StochRSIConfig = Field(default_factory=StochRSIConfig)
    atr: ATRSignalConfig = Field(default_factory=ATRSignalConfig)
    obv: OBVConfig = Field(default_factory=OBVConfig)
    vwap: VWAPConfig = Field(default_factory=VWAPConfig)


class DataConfig(BaseModel):
    cache_dir: str = "./data/cache"
    cache_format: str = "parquet"
    cache_expiry_hours: int = 1
    historical_lookback_years: int = 2
    sentiment_poll_interval_minutes: int = 15
    fear_greed_poll_interval_minutes: int = 60
    google_trends_poll_interval_hours: int = 24
    ohlcv_poll_interval_minutes: int = 5
    min_candles_required: int = 200
    max_gap_hours: int = 4
    fill_missing_method: str = "forward_fill"
    parquet_dir: str = "./data/parquet"
    parquet_compression: str = "snappy"


class SchedulerJobConfig(BaseModel):
    enabled: bool = True
    interval_minutes: Optional[int] = None
    interval_hours: Optional[int] = None
    interval_days: Optional[int] = None


class SchedulerConfig(BaseModel):
    timezone: str = "UTC"
    jobs: Dict[str, Any] = Field(default_factory=dict)


class NotificationsConfig(BaseModel):
    enabled: bool = False
    channels: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Root Config
# ---------------------------------------------------------------------------


class AlgoTraderConfig(BaseModel):
    """Root configuration object for AlgoTrader Pro."""

    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    data_providers: DataProvidersConfig = Field(default_factory=DataProvidersConfig)
    symbols_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "bitcoin": "BTCUSDT",
            "ethereum": "ETHUSDT",
            "solana": "SOLUSDT",
            "binancecoin": "BNBUSDT",
        }
    )
    symbols_reverse: Dict[str, str] = Field(
        default_factory=lambda: {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "SOLUSDT": "solana",
            "BNBUSDT": "binancecoin",
        }
    )
    timeframes: TimeframesConfig = Field(default_factory=TimeframesConfig)
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)

    # Convenience properties
    @property
    def is_paper(self) -> bool:
        return self.app.environment == "paper"

    @property
    def is_live(self) -> bool:
        return self.app.environment == "live"

    @property
    def trading_symbols(self) -> List[str]:
        """Returns the list of trading pairs (e.g. ['BTCUSDT', ...])."""
        return list(self.symbols_mapping.values())

    @property
    def coingecko_symbols(self) -> List[str]:
        """Returns CoinGecko IDs (e.g. ['bitcoin', ...])."""
        return self.data_providers.coingecko.symbols


# ---------------------------------------------------------------------------
# Environment Variable Overrides
# ---------------------------------------------------------------------------


def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to the raw YAML dict.
    Priority: env vars > settings.yaml defaults.
    """
    # App
    if v := os.getenv("APP_ENV"):
        raw.setdefault("app", {})["environment"] = v
    if v := os.getenv("LOG_LEVEL"):
        raw.setdefault("app", {})["log_level"] = v

    # Database
    if v := os.getenv("DATABASE_URL"):
        raw.setdefault("database", {})["url"] = v

    # CryptoPanic
    if v := os.getenv("CRYPTOPANIC_API_KEY"):
        raw.setdefault("data_providers", {}).setdefault("cryptopanic", {})["api_key"] = v
    if v := os.getenv("CRYPTOPANIC_BASE_URL"):
        raw.setdefault("data_providers", {}).setdefault("cryptopanic", {})["base_url"] = v

    # CoinGecko
    if v := os.getenv("COINGECKO_API_KEY"):
        raw.setdefault("data_providers", {}).setdefault("coingecko", {})["api_key"] = v

    # Bybit (Phase 2)
    if v := os.getenv("BYBIT_API_KEY"):
        raw.setdefault("bybit", {})["api_key"] = v
    if v := os.getenv("BYBIT_API_SECRET"):
        raw.setdefault("bybit", {})["api_secret"] = v
    if v := os.getenv("BYBIT_TESTNET"):
        raw.setdefault("bybit", {})["use_testnet"] = v.lower() == "true"

    return raw


# ---------------------------------------------------------------------------
# Loader & Singleton
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_config_instance: Optional[AlgoTraderConfig] = None


def load_config(
    config_path: Optional[Path] = None,
    env_path: Optional[Path] = None,
    force_reload: bool = False,
) -> AlgoTraderConfig:
    """
    Load and validate the full application configuration.

    Args:
        config_path: Path to settings.yaml (defaults to project root/config/settings.yaml)
        env_path:    Path to .env file (defaults to project root/.env)
        force_reload: If True, bypass the singleton cache and reload from disk.

    Returns:
        Validated AlgoTraderConfig instance.
    """
    global _config_instance

    if not force_reload and _config_instance is not None:
        return _config_instance

    with _config_lock:
        # Double-checked locking
        if not force_reload and _config_instance is not None:
            return _config_instance

        # 1. Load .env file (does not override existing OS env vars by default)
        resolved_env = env_path or _ENV_PATH
        if resolved_env.exists():
            load_dotenv(resolved_env, override=False)

        # 2. Load YAML
        resolved_cfg = config_path or _CONFIG_PATH
        if not resolved_cfg.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {resolved_cfg}\n"
                "Ensure config/settings.yaml exists in the project root."
            )

        with open(resolved_cfg, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        # 3. Apply environment variable overrides
        raw = _apply_env_overrides(raw)

        # 4. Validate with Pydantic
        config = AlgoTraderConfig.model_validate(raw)

        _config_instance = config
        return config


@lru_cache(maxsize=1)
def get_config() -> AlgoTraderConfig:
    """
    Return the singleton AlgoTraderConfig instance.

    This function is safe to call from multiple threads and multiple times;
    it loads the config once and caches it for the lifetime of the process.

    Example::

        from src.utils.config_loader import get_config
        cfg = get_config()
        print(cfg.app.environment)       # "paper"
        print(cfg.strategy.name)         # "HybridV1"
        print(cfg.risk.kelly_max_fraction)  # 0.15
    """
    return load_config()


# ---------------------------------------------------------------------------
# Convenience accessors (optional shortcuts)
# ---------------------------------------------------------------------------


def get_db_url() -> str:
    """Return the DATABASE_URL from config (with env override applied)."""
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    return get_config().database.url


def get_trading_symbols() -> List[str]:
    """Return list of trading pairs (e.g. ['BTCUSDT', 'ETHUSDT', ...])."""
    return get_config().trading_symbols


def get_coingecko_symbols() -> List[str]:
    """Return list of CoinGecko IDs (e.g. ['bitcoin', 'ethereum', ...])."""
    return get_config().coingecko_symbols


# ---------------------------------------------------------------------------
# CLI entry point for config validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("Validating configuration...")
    cfg = load_config(force_reload=True)
    print("Configuration valid.")
    print(f"  App:         {cfg.app.name} v{cfg.app.version} [{cfg.app.environment}]")
    print(f"  Strategy:    {cfg.strategy.name}")
    print(f"  Symbols:     {cfg.trading_symbols}")
    print(f"  Timeframes:  {cfg.timeframes.primary} (primary), {cfg.timeframes.secondary}")
    print(f"  Capital:     {cfg.paper_trading.initial_capital} {cfg.paper_trading.currency}")
    print(f"  DB URL:      {cfg.database.url}")
    print(
        f"  ML weights:  XGB={cfg.ml.ensemble_weights.xgboost} "
        f"RF={cfg.ml.ensemble_weights.random_forest} "
        f"LSTM={cfg.ml.ensemble_weights.lstm}"
    )
