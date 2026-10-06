"""Elering Nord Pool day-ahead price provider.

Secondary and cross-validation provider for the Lithuanian Gen0 system.

API:
  https://dashboard.elering.ee/api/nps/price?start={start}&end={end}
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

DEFAULT_ELERING_URL = "https://dashboard.elering.ee/api/nps/price"


class EleringPriceProvider(PriceProvider):
    """Secondary / Validation day-ahead price provider using public Elering API."""

    source = "ELERING"

    def __init__(
        self,
        base_url: str = DEFAULT_ELERING_URL,
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
    ) -> list[MarketPriceInterval]:
        """Fetch and return canonical MarketPriceIntervals."""
        s = start_utc or (datetime.now(timezone.utc) - timedelta(days=1))
        e = end_utc or (s + timedelta(days=2))
        res = self.fetch_day_ahead(bidding_zone=bidding_zone, start_utc=s, end_utc=e)
        return res.market_intervals

    def fetch_day_ahead(
        self,
        bidding_zone: str,
        start_utc: datetime,
        end_utc: datetime,
    ) -> PriceFetchResult:
        """Fetch day-ahead prices for specified period and return PriceFetchResult."""
        zone = bidding_zone.upper().strip()
        start_utc = ensure_utc(start_utc)
        end_utc = ensure_utc(end_utc)

        # Elering API requires ISO-8601 UTC query strings
        start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {"start": start_str, "end": end_str}

        raw_text, status_code = self._get_payload(params)
        payload_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        fetched_at = datetime.now(timezone.utc)

        try:
            data = json.loads(raw_text)
        except Exception as exc:
            raise PriceProviderError(f"Failed to parse Elering JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise PriceProviderError(f"Expected Elering JSON object, got {type(data).__name__}")

        if not data.get("success", False) and "data" not in data:
            raise PriceProviderError(f"Elering API response indicated failure: {raw_text[:200]}")

        sub_data = data.get("data", {})
        zone_key = zone.lower()
        if zone_key not in sub_data:
            raise PriceProviderError(
                f"Bidding zone '{zone}' (key '{zone_key}') not found in Elering response keys: {list(sub_data.keys())}"
            )

        items = sub_data[zone_key]
        if not isinstance(items, list):
            raise PriceProviderError(f"Expected list for zone '{zone_key}', got {type(items).__name__}")

        intervals = self._parse_items(items, zone, fetched_at, payload_hash, start_utc, end_utc)
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
            request_params=params,
            market_intervals=intervals,
        )

    def _get_payload(self, params: dict[str, str]) -> tuple[str, int]:
        last_exc: Exception | None = None
        backoff = self.initial_backoff_sec

        for attempt in range(1, self.max_retries + 1):
            try:
                if self._custom_client is not None:
                    resp = self._custom_client.get(self.base_url, params=params, timeout=self.timeout)
                else:
                    with httpx.Client(timeout=self.timeout) as client:
                        resp = client.get(self.base_url, params=params)

                if resp.status_code >= 400:
                    raise PriceProviderError(
                        f"Elering API HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                return resp.text, resp.status_code

            except Exception as exc:
                last_exc = exc
                log.warning("Elering fetch failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0

        raise PriceProviderError(
            f"Elering API unavailable after {self.max_retries} attempts: {last_exc}"
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
        raw_points: list[tuple[datetime, float]] = []
        for item in items:
            if not isinstance(item, dict) or "timestamp" not in item or "price" not in item:
                continue
            try:
                ts = int(item["timestamp"])
                p_val = float(item["price"])
                t_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
                raw_points.append((t_utc, p_val))
            except (ValueError, TypeError):
                continue

        raw_points.sort(key=lambda x: x[0])
        intervals: list[MarketPriceInterval] = []

        for i, (t_utc, p_val) in enumerate(raw_points):
            # Calculate duration: Elering NPS prices are generally 15-minute (900s)
            if i < len(raw_points) - 1:
                dt_sec = int((raw_points[i + 1][0] - t_utc).total_seconds())
                dur_min = dt_sec // 60 if dt_sec in (900, 1800, 3600) else 15
            else:
                dur_min = intervals[-1].resolution_minutes if intervals else 15

            iv_end = t_utc + timedelta(minutes=dur_min)
            if iv_end <= start_utc or t_utc >= end_utc:
                continue

            intervals.append(
                MarketPriceInterval(
                    start_utc=t_utc,
                    end_utc=iv_end,
                    price_eur_mwh=p_val,
                    bidding_zone=zone,
                    source=self.source,
                    original_resolution_minutes=dur_min,
                    fetched_at_utc=fetched_at,
                    source_record_id=f"elering-{zone.lower()}-{int(t_utc.timestamp())}",
                    is_derived=False,
                    raw_payload_hash=payload_hash,
                )
            )

        return intervals
