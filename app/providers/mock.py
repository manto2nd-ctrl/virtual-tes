"""Deterministic mock day-ahead prices.

Produces a realistic-looking Baltic daily shape in LOCAL time (night trough, morning
peak, solar dip, evening peak) plus seeded noise. Same date + seed => identical prices,
which keeps simulations and tests reproducible.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.core.timegrid import build_intervals, ensure_utc
from app.models.domain import PriceFetchResult, PricePoint
from app.providers.price_provider import PriceProvider


class MockPriceProvider(PriceProvider):
    source = "mock"

    def __init__(self, tz: ZoneInfo, resolution_minutes: int = 15, seed: int = 42,
                 noise_eur_mwh: float = 8.0, base_eur_mwh: float = 90.0) -> None:
        self.tz = tz
        self.resolution_minutes = resolution_minutes
        self.seed = seed
        self.noise = noise_eur_mwh
        self.base = base_eur_mwh

    @staticmethod
    def _bump(h: float, centre: float, width: float) -> float:
        return math.exp(-((h - centre) ** 2) / (2 * width ** 2))

    def shape(self, local_hour: float) -> float:
        """Deterministic price shape [EUR/MWh] as a function of local hour (0..24)."""
        h = local_hour
        return (
            self.base
            - 50 * self._bump(h, 3.5, 2.0)  # night trough
            + 70 * self._bump(h, 8.0, 1.5)  # morning peak
            - 45 * self._bump(h, 13.5, 2.0)  # midday solar dip
            + 130 * self._bump(h, 19.0, 1.8)  # evening peak
        )

    def fetch_day_ahead(self, bidding_zone: str, start_utc: datetime, end_utc: datetime) -> PriceFetchResult:
        start, end = ensure_utc(start_utc), ensure_utc(end_utc)
        points: list[PricePoint] = []
        published = datetime(2000, 1, 1, tzinfo=timezone.utc)  # placeholder, deterministic
        for iv in build_intervals(start, end, self.resolution_minutes):
            local = iv.start_local(self.tz)
            # Seed per interval from (seed, UTC timestamp) -> independent of query range.
            rng = random.Random(f"{self.seed}-{iv.start_utc.isoformat()}")
            price = self.shape(local.hour + local.minute / 60.0) + rng.uniform(-self.noise, self.noise)
            points.append(PricePoint(
                bidding_zone=bidding_zone,
                delivery_start_utc=iv.start_utc,
                delivery_end_utc=iv.end_utc,
                price_eur_mwh=round(price, 2),
                resolution_minutes=self.resolution_minutes,
                source=self.source,
                published_at=published,
            ))
        raw = json.dumps({
            "provider": self.source, "seed": self.seed, "bidding_zone": bidding_zone,
            "prices": [{"start_utc": p.delivery_start_utc.isoformat(), "price": p.price_eur_mwh} for p in points],
        })
        return PriceFetchResult(
            source=self.source, bidding_zone=bidding_zone, period_start_utc=start, period_end_utc=end,
            points=points, raw_payload=raw, http_status=None,
            request_params={"seed": self.seed, "resolution_minutes": self.resolution_minutes},
        )
