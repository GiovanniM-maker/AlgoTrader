"""
Layer-3 Sentiment Signal: CryptoPanic News Sentiment.

CryptoPanic is a news aggregator that tags crypto articles as bullish,
bearish, or neutral.  The upstream data provider (in ``data/providers``)
queries the CryptoPanic API and distils the raw vote counts into a single
normalised sentiment score in [-1.0, +1.0].

Signal construction
-------------------
* The sentiment_score is used *directly* as the signal value.
* Strength = abs(sentiment_score) — confident readings are weighted more.
* If no data is available (score is None or provider failed), the signal
  returns neutral (value=0, strength=0) so the aggregator skips it
  gracefully without erroring.

Typical upstream score formula (implemented by the provider):
    score = (bullish_votes - bearish_votes) / max(total_votes, 1)
    score = tanh(score * amplification_factor)   # smooth to [-1, +1]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)


class CryptoPanicSignal(BaseSignal):
    """
    CryptoPanic news-sentiment passthrough signal.

    The signal value is a direct mapping of the upstream sentiment score.
    No secondary computation is performed here; the signal simply packages
    the provider output into a SignalResult for the aggregator.
    """

    name: str = "cryptopanic"
    layer: int = 3
    weight: float = 1.0

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def compute_from_score(
        self,
        sentiment_score: Optional[float],
        article_count: int = 0,
        extra_metadata: Optional[dict] = None,
    ) -> SignalResult:
        """
        Compute the signal from a pre-computed CryptoPanic sentiment score.

        Parameters
        ----------
        sentiment_score : float or None
            Normalised sentiment in [-1.0, +1.0].  Pass None when the
            provider returned no data.
        article_count : int
            Number of articles used to compute the score (for logging).
        extra_metadata : dict, optional
            Any additional metadata to attach to the result.

        Returns
        -------
        SignalResult
        """
        if sentiment_score is None or np.isnan(float(sentiment_score)):
            return self._neutral_result("no CryptoPanic data available")

        score: float = float(np.clip(sentiment_score, -1.0, 1.0))
        strength: float = abs(score)

        meta = {
            "sentiment_score": round(score, 6),
            "article_count": article_count,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        logger.debug(
            "[%s] score=%.4f  articles=%d  strength=%.4f",
            self.name, score, article_count, strength,
        )

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=score,
            strength=strength,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # BaseSignal / aggregator compatibility
    # ------------------------------------------------------------------

    def compute(self, data) -> SignalResult:  # type: ignore[override]
        """
        Compatibility entry-point called by the aggregator.

        Accepts either:
        * A dict with key ``cryptopanic_score`` (float) and optionally
          ``cryptopanic_article_count`` (int).
        * A pandas DataFrame (unsupported — returns neutral).
        """
        import pandas as pd

        if isinstance(data, dict):
            score: Optional[float] = data.get("cryptopanic_score")
            count: int = int(data.get("cryptopanic_article_count", 0))
            extra: dict = {
                k: v
                for k, v in data.items()
                if k not in ("cryptopanic_score", "cryptopanic_article_count")
                and k.startswith("cryptopanic_")
            }
            return self.compute_from_score(score, article_count=count, extra_metadata=extra)

        if isinstance(data, pd.DataFrame):
            return self._neutral_result("CryptoPanicSignal expects a sentiment dict, not a DataFrame")

        return self._neutral_result("unsupported input type")
