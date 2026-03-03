"""
src/api/schemas/models.py
===========================
Pydantic v2 request / response models for the AlgoTrader Pro API.

These models are used for:
  - Response serialisation (type-safe JSON output).
  - Request body validation (POST endpoints).
  - OpenAPI schema generation (auto-docs at /docs).

All monetary values are in USDT unless noted otherwise.
Percentage values are expressed as plain floats (e.g. 3.5 means 3.5 %).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Portfolio / equity models
# ---------------------------------------------------------------------------


class EquityPoint(BaseModel):
    """Single point on the equity curve time-series."""

    timestamp: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of the snapshot.",
        examples=["2024-01-15T12:00:00+00:00"],
    )
    equity: float = Field(
        ...,
        description="Total portfolio value in USDT at this point in time.",
        examples=[10_250.75],
    )
    drawdown_pct: float = Field(
        ...,
        description="Current drawdown from the equity peak as a positive percentage.",
        examples=[2.34],
    )
    cash: Optional[float] = Field(
        default=None,
        description="Available un-invested cash in USDT.",
        examples=[6_500.00],
    )


class Position(BaseModel):
    """An open trading position."""

    trade_id: str = Field(..., description="Unique UUID for this trade/position.")
    symbol: str = Field(..., description="Trading pair symbol, e.g. BTCUSDT.")
    direction: str = Field(..., description="'long' or 'short'.")
    entry_price: float = Field(..., description="Price at which the position was opened.")
    current_price: float = Field(..., description="Most recent market price for mark-to-market.")
    quantity: float = Field(..., description="Quantity of the base asset held.")
    notional_value: float = Field(..., description="entry_price * quantity in USDT.")
    unrealized_pnl: float = Field(
        ...,
        description="Unrealised profit/loss in USDT at the current market price.",
    )
    unrealized_pnl_pct: float = Field(
        ...,
        description="Unrealised P&L as a percentage of the notional entry value.",
    )
    stop_loss: float = Field(..., description="Stop-loss price level.")
    take_profit: Optional[float] = Field(
        default=None, description="Take-profit price level (if set)."
    )
    confidence_score: float = Field(
        ...,
        description="Signal confidence score [0-100] that triggered the trade entry.",
    )
    entry_time: str = Field(
        ..., description="ISO-8601 UTC timestamp when the trade was opened."
    )


# ---------------------------------------------------------------------------
# Trade models
# ---------------------------------------------------------------------------


class TradeResponse(BaseModel):
    """Full trade record returned by /trades and /trades/{trade_id}."""

    trade_id: str
    symbol: str
    direction: str = Field(..., description="'long' or 'short'.")
    status: str = Field(..., description="'open', 'closed', 'cancelled', or 'error'.")

    # Entry
    entry_time: str
    entry_price: float
    entry_slippage: float = Field(0.0, description="Entry slippage percentage.")
    entry_fee: float = Field(0.0, description="Fee paid at entry in USDT.")
    quantity: float
    notional_value: float

    # Position sizing
    kelly_fraction: Optional[float] = None
    risk_amount: Optional[float] = None

    # Exit (nullable for open trades)
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_slippage: Optional[float] = None
    exit_fee: Optional[float] = None
    exit_reason: Optional[str] = Field(
        default=None,
        description=(
            "Reason for exit: 'stop_loss', 'take_profit', 'trailing_stop', "
            "'time_exit', 'signal_reversal', 'manual', 'drawdown_pause', or 'error'."
        ),
    )

    # P&L
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration_minutes: Optional[int] = None

    # Risk levels
    stop_loss_price: float
    take_profit_price: Optional[float] = None
    atr_at_entry: Optional[float] = None

    # Signal metadata
    confidence_score: float
    signal_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-layer signal scores as a JSON object.",
    )

    # Audit
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Signal models
# ---------------------------------------------------------------------------


class SignalResponse(BaseModel):
    """Signal evaluation result for a single symbol."""

    symbol: str = Field(..., description="Trading pair, e.g. BTCUSDT.")
    confidence_score: float = Field(
        ..., description="Composite signal confidence [0-100]."
    )
    direction: str = Field(
        ..., description="Signal direction: 'long', 'short', or 'neutral'."
    )
    layer1_score: float = Field(0.0, description="Technical indicator layer score [0-100].")
    layer2_score: float = Field(0.0, description="Volume analysis layer score [0-100].")
    layer3_score: float = Field(0.0, description="Sentiment layer score [0-100].")
    ml_score: float = Field(0.0, description="ML ensemble layer score [0-100].")
    action_taken: str = Field(
        ...,
        description=(
            "Disposition: 'trade_opened', 'skipped_threshold', 'skipped_max_positions', "
            "'skipped_cooldown', 'skipped_correlation', 'skipped_ev_negative', "
            "'skipped_drawdown_pause', 'skipped_short_disabled', or 'error'."
        ),
    )
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of the signal.")


# ---------------------------------------------------------------------------
# Bot control models
# ---------------------------------------------------------------------------


class BotStatusResponse(BaseModel):
    """Current operational status of the trading engine."""

    status: str = Field(
        ...,
        description="'RUNNING', 'PAUSED', 'STOPPED', or 'ERROR'.",
    )
    started_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 UTC timestamp when the engine last started.",
    )
    uptime_seconds: Optional[float] = Field(
        default=None,
        description="Seconds elapsed since last start.",
    )
    symbols: List[str] = Field(
        default_factory=list,
        description="List of symbols the engine is currently trading.",
    )
    mode: str = Field(
        default="paper",
        description="Trading mode: 'paper' or 'live'.",
    )
    total_trades: int = Field(
        default=0,
        description="Total trades opened in the current session.",
    )


# ---------------------------------------------------------------------------
# Dashboard models
# ---------------------------------------------------------------------------


class DashboardSummary(BaseModel):
    """Top-level portfolio summary for the dashboard header cards."""

    equity: float = Field(..., description="Current total portfolio value in USDT.")
    cash: float = Field(..., description="Available uninvested cash in USDT.")
    initial_capital: float = Field(
        ..., description="Starting capital when the bot was first configured."
    )
    pnl_today_pct: float = Field(
        ..., description="Percentage P&L for the current calendar day."
    )
    pnl_total_pct: float = Field(
        ..., description="Total percentage P&L since inception."
    )
    win_rate: float = Field(
        ..., description="Fraction of closed trades that were profitable [0.0-1.0]."
    )
    open_positions_count: int = Field(
        ..., description="Number of currently open positions."
    )
    bot_status: str = Field(
        ..., description="Current engine status: 'RUNNING', 'PAUSED', or 'STOPPED'."
    )
    fear_greed_value: Optional[float] = Field(
        default=None,
        description="Latest Fear & Greed Index value [0-100].",
    )
    fear_greed_label: Optional[str] = Field(
        default=None,
        description="Human-readable F&G classification, e.g. 'Greed'.",
    )
    last_updated: str = Field(
        ..., description="ISO-8601 UTC timestamp when this summary was computed."
    )


class MarketTick(BaseModel):
    """Real-time market data for a single trading pair."""

    symbol: str = Field(..., description="Trading pair, e.g. BTCUSDT.")
    price: float = Field(..., description="Last traded price in USDT.")
    change_24h_pct: float = Field(
        ..., description="24-hour price change as a percentage."
    )
    volume_24h: float = Field(..., description="24-hour trading volume in USDT.")
    high_24h: float = Field(..., description="24-hour high price in USDT.")
    low_24h: float = Field(..., description="24-hour low price in USDT.")


# ---------------------------------------------------------------------------
# Backtest models
# ---------------------------------------------------------------------------


class BacktestRunRequest(BaseModel):
    """Request body for POST /backtests/run."""

    symbol: str = Field(
        default="bitcoin",
        description=(
            "CoinGecko coin ID used for historical data fetch "
            "(e.g. 'bitcoin', 'ethereum', 'solana')."
        ),
    )
    timeframe: str = Field(
        default="1h",
        description="Candle timeframe: '1h', '4h', or '1d'.",
    )
    start_date: str = Field(
        default="2023-01-01",
        description="Backtest start date in YYYY-MM-DD format.",
    )
    end_date: str = Field(
        default="2024-01-01",
        description="Backtest end date in YYYY-MM-DD format.",
    )
    initial_capital: float = Field(
        default=10_000.0,
        gt=0,
        description="Starting capital in USDT.",
    )


class BacktestStatusResponse(BaseModel):
    """Status of a running or completed backtest job."""

    run_id: str = Field(..., description="UUID assigned to this backtest run.")
    status: str = Field(
        ...,
        description="'pending', 'running', 'complete', or 'failed'.",
    )
    progress_pct: float = Field(
        default=0.0,
        description="Approximate progress percentage [0-100].",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'.",
    )


class BacktestSummary(BaseModel):
    """Summary row returned by GET /backtests (list view)."""

    run_id: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: float
    total_trades: int
    win_rate: Optional[float] = None
    created_at: str


# ---------------------------------------------------------------------------
# Performance metrics model
# ---------------------------------------------------------------------------


class PerformanceMetrics(BaseModel):
    """Full suite of risk-adjusted performance metrics."""

    sharpe_ratio: Optional[float] = Field(
        default=None,
        description="Annualised Sharpe Ratio (excess return / total volatility).",
    )
    sortino_ratio: Optional[float] = Field(
        default=None,
        description="Annualised Sortino Ratio (excess return / downside volatility).",
    )
    max_drawdown_pct: float = Field(
        ...,
        description="Maximum peak-to-trough drawdown as a positive percentage.",
    )
    calmar_ratio: Optional[float] = Field(
        default=None,
        description="CAGR divided by Max Drawdown.",
    )
    profit_factor: Optional[float] = Field(
        default=None,
        description="Gross profit divided by gross loss across all trades.",
    )
    win_rate: float = Field(
        ...,
        description="Fraction of profitable trades [0.0-1.0].",
    )
    total_trades: int = Field(..., description="Total number of closed trades analysed.")
    avg_win_pct: float = Field(
        ...,
        description="Average P&L percentage of winning trades.",
    )
    avg_loss_pct: float = Field(
        ...,
        description="Average P&L percentage of losing trades (typically negative).",
    )
    total_return_pct: float = Field(
        ...,
        description="Total portfolio return from inception as a percentage.",
    )


# ---------------------------------------------------------------------------
# Generic API response wrappers
# ---------------------------------------------------------------------------


class PaginatedTrades(BaseModel):
    """Paginated list of trade records."""

    trades: List[TradeResponse]
    total: int = Field(..., description="Total number of matching records.")
    page: int = Field(..., description="Current page number (1-indexed).")
    pages: int = Field(..., description="Total number of pages.")


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    mode: str = Field(default="paper")
