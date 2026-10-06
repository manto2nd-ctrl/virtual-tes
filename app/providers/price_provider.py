"""Abstract price provider interface.

Implementations: ``MockPriceProvider`` (now), ``CsvPriceProvider`` and
``EntsoePriceProvider`` (Phase 3), ``NordPoolPriceProvider`` (future). Downstream code
depends only on this interface and on ``PricePoint``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from app.models.domain import PriceFetchResult


class PriceProviderError(RuntimeError):
    """Raised when a provider cannot deliver data. Must never cause data deletion."""


class PriceProvider(ABC):
    #: identifier stored in ``day_ahead_prices.source``
    source: str = "abstract"

    @abstractmethod
    def fetch_day_ahead(self, bidding_zone: str, start_utc: datetime, end_utc: datetime) -> PriceFetchResult:
        """Return day-ahead prices for delivery intervals within ``[start_utc, end_utc)``.

        Implementations must return timezone-aware UTC timestamps and must not assume a
        fixed number of intervals per day.
        """
