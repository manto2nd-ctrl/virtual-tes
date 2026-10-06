"""Market information availability and receding-horizon timeline model.

Models when day-ahead electricity market prices become known to the EMS controller.
Supports:
  1. Live publication timestamps (``PricePoint.published_at``) if recorded.
  2. Configurable day-ahead market auction clearing assumption (e.g. 14:00 local time
     on the calendar day prior to delivery for Lithuania / Nord Pool / ENTSO-E).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from app.core.timegrid import ensure_utc
from app.models.domain import PricePoint


@dataclass(frozen=True)
class MarketInformationConfig:
    """Configuration for market information arrival timeline."""

    day_ahead_assumed_available_local_time: str = "14:00"  # HH:MM
    market_timezone: str = "Europe/Vilnius"
    use_actual_published_at_if_available: bool = True
    is_assumed_timing: bool = True  # Tag marking historical day-ahead availability as an assumption


class MarketInformationModel:
    """Determines when market prices become known and tracks information arrival events."""

    def __init__(self, config: MarketInformationConfig | None = None) -> None:
        self.config = config or MarketInformationConfig()
        self.tz = ZoneInfo(self.config.market_timezone)
        parts = self.config.day_ahead_assumed_available_local_time.split(":")
        self.pub_hour = int(parts[0])
        self.pub_minute = int(parts[1]) if len(parts) > 1 else 0

    def price_known_at_utc(self, point: PricePoint) -> datetime:
        """UTC timestamp when the price point becomes known to the controller."""
        if self.config.use_actual_published_at_if_available and point.published_at is not None:
            return ensure_utc(point.published_at)

        # Day-ahead assumption: published at configured local time on the day prior to delivery
        deliv_local = point.delivery_start_utc.astimezone(self.tz)
        prior_day = deliv_local.date() - timedelta(days=1)
        pub_local = datetime.combine(
            prior_day,
            time(self.pub_hour, self.pub_minute),
            tzinfo=self.tz,
        )
        return pub_local.astimezone(timezone.utc)

    def get_known_prices(
        self,
        prices: list[PricePoint],
        as_of_utc: datetime,
    ) -> list[PricePoint]:
        """Return all price points that are known at or before ``as_of_utc``."""
        as_of = ensure_utc(as_of_utc)
        return [p for p in prices if self.price_known_at_utc(p) <= as_of]

    def get_market_events(
        self,
        prices: list[PricePoint],
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[datetime]:
        """Return sorted unique UTC timestamps where new price information arrives.

        Always includes ``start_utc`` and any event occurring within ``[start_utc, end_utc)``.
        """
        s_utc = ensure_utc(start_utc)
        e_utc = ensure_utc(end_utc)
        known_times = {s_utc}
        for p in prices:
            k_utc = self.price_known_at_utc(p)
            if s_utc < k_utc < e_utc:
                known_times.add(k_utc)
        return sorted(known_times)
