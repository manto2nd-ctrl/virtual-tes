"""Pure domain objects shared between providers, services and the database layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.timegrid import ensure_utc


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class MarketPriceInterval:
    """Canonical normalized market price interval across all market providers.

    Conforms to Phase 5.8 Normalized Market Model requirements:
    - start_utc: UTC interval start
    - end_utc: UTC interval end
    - price_eur_mwh: Day-ahead spot price in EUR/MWh
    - bidding_zone: Bidding zone string (e.g. "LT")
    - source: Provider name (e.g. "LITGRID", "ELERING", "VOLTON", "ENTSOE")
    - original_resolution_minutes: Resolution of the original source interval
    - fetched_at_utc: Timestamp when provider fetched or parsed this data
    - source_record_id: Provider's native record identifier / key
    - is_derived: True if interpolated/converted from a different resolution
    - raw_payload_hash: SHA-256 hash of the raw response payload for provenance
    """

    start_utc: datetime
    end_utc: datetime
    price_eur_mwh: float
    bidding_zone: str = "LT"
    source: str = "LITGRID"
    original_resolution_minutes: int = 15
    fetched_at_utc: datetime = field(default_factory=utcnow)
    source_record_id: str | None = None
    is_derived: bool = False
    raw_payload_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_utc", ensure_utc(self.start_utc))
        object.__setattr__(self, "end_utc", ensure_utc(self.end_utc))
        object.__setattr__(self, "fetched_at_utc", ensure_utc(self.fetched_at_utc))
        object.__setattr__(self, "bidding_zone", self.bidding_zone.upper().strip())
        object.__setattr__(self, "source", self.source.upper().strip())
        if self.end_utc <= self.start_utc:
            raise ValueError(f"end_utc ({self.end_utc}) must be after start_utc ({self.start_utc})")

    @property
    def resolution_minutes(self) -> int:
        return int((self.end_utc - self.start_utc).total_seconds() / 60)

    def to_price_point(self) -> PricePoint:
        """Convert to existing PricePoint domain object for downstream compatibility."""
        return PricePoint(
            bidding_zone=self.bidding_zone,
            delivery_start_utc=self.start_utc,
            delivery_end_utc=self.end_utc,
            price_eur_mwh=self.price_eur_mwh,
            resolution_minutes=self.resolution_minutes,
            source=self.source,
            published_at=self.fetched_at_utc,
            currency="EUR",
            original_resolution_minutes=self.original_resolution_minutes,
            derived_from_id=self.source_record_id if self.is_derived else None,
        )

    @classmethod
    def from_price_point(
        cls,
        point: PricePoint,
        source_record_id: str | None = None,
        is_derived: bool = False,
        raw_payload_hash: str | None = None,
    ) -> MarketPriceInterval:
        """Construct canonical MarketPriceInterval from a PricePoint."""
        return cls(
            start_utc=point.delivery_start_utc,
            end_utc=point.delivery_end_utc,
            price_eur_mwh=point.price_eur_mwh,
            bidding_zone=point.bidding_zone,
            source=point.source,
            original_resolution_minutes=point.original_resolution_minutes or point.resolution_minutes,
            fetched_at_utc=point.published_at or utcnow(),
            source_record_id=source_record_id or point.derived_from_id,
            is_derived=is_derived or (point.derived_from_id is not None),
            raw_payload_hash=raw_payload_hash,
        )


@dataclass(frozen=True, slots=True)
class PricePoint:
    """One day-ahead market price for one delivery interval."""

    bidding_zone: str
    delivery_start_utc: datetime
    delivery_end_utc: datetime
    price_eur_mwh: float
    resolution_minutes: int
    source: str
    published_at: datetime | None = None
    currency: str = "EUR"
    original_resolution_minutes: int | None = None
    derived_from_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "delivery_start_utc", ensure_utc(self.delivery_start_utc))
        object.__setattr__(self, "delivery_end_utc", ensure_utc(self.delivery_end_utc))
        if self.published_at is not None:
            object.__setattr__(self, "published_at", ensure_utc(self.published_at))
        if self.original_resolution_minutes is None:
            object.__setattr__(self, "original_resolution_minutes", self.resolution_minutes)
        if self.delivery_end_utc <= self.delivery_start_utc:
            raise ValueError("delivery_end_utc must be after delivery_start_utc")
        expected = (self.delivery_end_utc - self.delivery_start_utc).total_seconds() / 60
        if int(expected) != self.resolution_minutes:
            raise ValueError(
                f"resolution_minutes={self.resolution_minutes} inconsistent with interval length {expected} min"
            )


@dataclass(frozen=True, slots=True)
class PriceFetchResult:
    """Result of a provider fetch: parsed points plus the raw payload for audit."""

    source: str
    bidding_zone: str
    period_start_utc: datetime
    period_end_utc: datetime
    points: list[PricePoint]
    raw_payload: str | None = None
    raw_content_type: str = "application/json"
    http_status: int | None = None
    request_params: dict = field(default_factory=dict)
    market_intervals: list[MarketPriceInterval] = field(default_factory=list)
