"""Litgrid Open Data API day-ahead price provider.

Primary provider for Lithuanian Gen0 Virtual TES system.

Dataset:
  Nord Pool Lietuva (EUR/MWh)
Official API:
  https://openapi.litgrid.eu/v1/kategorijos/elektros-energijos-kainos/801
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.core.timegrid import ensure_utc
from app.models.domain import MarketPriceInterval, PriceFetchResult, PricePoint
from app.providers.price_provider import PriceProvider, PriceProviderError

log = logging.getLogger(__name__)

DEFAULT_LITGRID_URL = "https://openapi.litgrid.eu/v1/kategorijos/elektros-energijos-kainos/801"


def parse_litgrid_timestamp(utc_str: str) -> datetime:
    """Parse Litgrid UTC timestamp string (e.g. '2025-10-05 19:00:00' or ISO format)."""
    s = utc_str.strip()
    if "T" in s:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    else:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return ensure_utc(dt)


def expand_intervals_to_resolution(
    intervals: list[MarketPriceInterval],
    target_resolution_minutes: int = 15,
) -> list[MarketPriceInterval]:
    """Ensure intervals match target resolution without modifying native intervals of that resolution.

    - 15-minute intervals remain 15-minute intervals (is_derived=False).
    - 60-minute intervals are expanded into four 15-minute sub-intervals with is_derived=True
      and original_resolution_minutes=60 preserved.
    """
    expanded: list[MarketPriceInterval] = []
    for iv in intervals:
        current_res = iv.resolution_minutes
        if current_res == target_resolution_minutes:
            expanded.append(iv)
        elif current_res > target_resolution_minutes and current_res % target_resolution_minutes == 0:
            sub_count = current_res // target_resolution_minutes
            sub_delta = timedelta(minutes=target_resolution_minutes)
            for i in range(sub_count):
                sub_start = iv.start_utc + i * sub_delta
                sub_end = sub_start + sub_delta
                expanded.append(
                    MarketPriceInterval(
                        start_utc=sub_start,
                        end_utc=sub_end,
                        price_eur_mwh=iv.price_eur_mwh,
                        bidding_zone=iv.bidding_zone,
                        source=iv.source,
                        original_resolution_minutes=iv.original_resolution_minutes,
                        fetched_at_utc=iv.fetched_at_utc,
                        source_record_id=f"{iv.source_record_id}-sub{i}" if iv.source_record_id else None,
                        is_derived=True,
                        raw_payload_hash=iv.raw_payload_hash,
                    )
                )
        else:
            # Pass through any interval that cannot be cleanly divided
            expanded.append(iv)
    return expanded


class LitgridPriceProvider(PriceProvider):
    """Primary day-ahead electricity price provider using the Litgrid Open Data API."""

    source = "LITGRID"

    def __init__(
        self,
        base_url: str = DEFAULT_LITGRID_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        initial_backoff_sec: float = 0.5,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.initial_backoff_sec = initial_backoff_sec
        self._custom_client = http_client

    def fetch_market_intervals(
        self,
        bidding_zone: str = "LT",
        start_utc: datetime | None = None,
        end_utc: datetime | None = None,
        target_resolution_minutes: int | None = None,
    ) -> list[MarketPriceInterval]:
        """Fetch and return canonical MarketPriceIntervals for Lithuania."""
        result = self.fetch_day_ahead(
            bidding_zone=bidding_zone,
            start_utc=start_utc or datetime.min.replace(tzinfo=timezone.utc),
            end_utc=end_utc or datetime.max.replace(tzinfo=timezone.utc),
        )
        intervals = result.market_intervals
        if target_resolution_minutes is not None:
            intervals = expand_intervals_to_resolution(intervals, target_resolution_minutes)
        return intervals

    def fetch_day_ahead(
        self,
        bidding_zone: str,
        start_utc: datetime,
        end_utc: datetime,
    ) -> PriceFetchResult:
        """Fetch day-ahead prices from Litgrid and return normalized PriceFetchResult."""
        zone = bidding_zone.upper().strip()
        if zone != "LT":
            raise PriceProviderError(
                f"LitgridPriceProvider only serves bidding zone 'LT' (received '{bidding_zone}')"
            )

        start_utc = ensure_utc(start_utc)
        end_utc = ensure_utc(end_utc)

        raw_text, status_code = self._get_payload()
        payload_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        fetched_at = datetime.now(timezone.utc)

        try:
            data = json.loads(raw_text)
        except Exception as exc:
            raise PriceProviderError(f"Failed to parse Litgrid JSON payload: {exc}") from exc

        if not isinstance(data, list):
            raise PriceProviderError(f"Expected Litgrid JSON list, got {type(data).__name__}")

        intervals = self._parse_items(data, zone, fetched_at, payload_hash, start_utc, end_utc)
        points = [iv.to_price_point() for iv in intervals]

        return PriceFetchResult(
            source=self.source,
            bidding_zone=zone,
            period_start_utc=start_utc,
            period_end_utc=end_utc,
            points=points,
            raw_payload=raw_text,
            raw_content_type="application/json",
            http_status=status_code,
            request_params={"url": self.base_url},
            market_intervals=intervals,
        )

    def _get_payload(self) -> tuple[str, int]:
        """Execute HTTP request with retries."""
        last_exc: Exception | None = None
        backoff = self.initial_backoff_sec

        for attempt in range(1, self.max_retries + 1):
            try:
                if self._custom_client is not None:
                    resp = self._custom_client.get(self.base_url, timeout=self.timeout)
                else:
                    with httpx.Client(timeout=self.timeout) as client:
                        resp = client.get(self.base_url)

                if resp.status_code >= 400:
                    raise PriceProviderError(
                        f"Litgrid API HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                return resp.text, resp.status_code

            except Exception as exc:
                last_exc = exc
                log.warning(
                    "Litgrid fetch failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0

        raise PriceProviderError(
            f"Litgrid API unavailable after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    def _parse_items(
        self,
        items: list[dict[str, Any]],
        zone: str,
        fetched_at: datetime,
        payload_hash: str,
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[MarketPriceInterval]:
        """Parse, deduplicate, sort, compute resolution, and filter items."""
        parsed_entries: list[tuple[datetime, float, str]] = []
        for item in items:
            if not isinstance(item, dict) or "utc" not in item or "value" not in item:
                continue
            try:
                t = parse_litgrid_timestamp(str(item["utc"]))
                val = float(item["value"])
                rec_id = str(item.get("id", "801"))
                parsed_entries.append((t, val, rec_id))
            except (ValueError, TypeError):
                continue

        if not parsed_entries:
            return []

        # Sort chronologically and deduplicate on timestamp
        parsed_entries.sort(key=lambda x: x[0])
        deduped: list[tuple[datetime, float, str]] = []
        seen: set[datetime] = set()
        for t, val, rec_id in parsed_entries:
            if t not in seen:
                seen.add(t)
                deduped.append((t, val, rec_id))

        # Detect interval durations
        durations_minutes: list[int] = []
        for i in range(len(deduped) - 1):
            dt_min = int((deduped[i + 1][0] - deduped[i][0]).total_seconds() / 60)
            durations_minutes.append(dt_min if dt_min in (15, 30, 60) else 60)

        # For the last element, reuse the preceding duration or default to 60
        last_dur = durations_minutes[-1] if durations_minutes else 60
        durations_minutes.append(last_dur)

        result: list[MarketPriceInterval] = []
        for (t, val, rec_id), dur in zip(deduped, durations_minutes, strict=True):
            iv_end = t + timedelta(minutes=dur)
            # Filter strictly within requested bounds if provided
            if iv_end <= start_utc or t >= end_utc:
                continue

            result.append(
                MarketPriceInterval(
                    start_utc=t,
                    end_utc=iv_end,
                    price_eur_mwh=val,
                    bidding_zone=zone,
                    source=self.source,
                    original_resolution_minutes=dur,
                    fetched_at_utc=fetched_at,
                    source_record_id=f"litgrid-{rec_id}-{t.strftime('%Y%m%d%H%M')}",
                    is_derived=False,
                    raw_payload_hash=payload_hash,
                )
            )

        return result
