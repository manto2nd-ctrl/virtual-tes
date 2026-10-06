"""Volton Public Data API day-ahead price provider.

Optional third fallback provider for Lithuanian day-ahead spot market data.

API:
  https://public-data.volton.energy/v1/day-ahead-spot-lt/latest.json
  https://public-data.volton.energy/v1/day-ahead-spot-lt/{date}.json
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

DEFAULT_VOLTON_BASE_URL = "https://public-data.volton.energy/v1/day-ahead-spot-lt"


class VoltonPriceProvider(PriceProvider):
    """Tertiary fallback provider using Volton's public day-ahead spot JSON dataset."""

    source = "VOLTON"

    def __init__(
        self,
        base_url: str = DEFAULT_VOLTON_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        initial_backoff_sec: float = 0.5,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
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
        zone = bidding_zone.upper().strip()
        if zone != "LT":
            raise PriceProviderError(
                f"Volton day-ahead dataset is specific to LT (received '{bidding_zone}')"
            )

        start_utc = ensure_utc(start_utc)
        end_utc = ensure_utc(end_utc)

        # Use latest.json for recent/upcoming queries, or date-specific archive
        now_utc = datetime.now(timezone.utc)
        if start_utc >= (now_utc - timedelta(days=7)):
            target_url = f"{self.base_url}/latest.json"
        else:
            date_str = start_utc.strftime("%Y-%m-%d")
            target_url = f"{self.base_url}/{date_str}.json"

        raw_text, status_code = self._get_payload(target_url)
        payload_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        fetched_at = datetime.now(timezone.utc)

        try:
            data = json.loads(raw_text)
        except Exception as exc:
            raise PriceProviderError(f"Failed to parse Volton JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise PriceProviderError(f"Expected Volton JSON object, got {type(data).__name__}")

        rows = data.get("rows", [])
        meta = data.get("meta", {})
        res_minutes = int(meta.get("resolution_minutes", 15))

        intervals: list[MarketPriceInterval] = []
        for r in rows:
            if not isinstance(r, dict) or "mtu_start" not in r or "price_eur_mwh" not in r:
                continue
            try:
                t_str = str(r["mtu_start"]).replace("Z", "+00:00")
                t_start = datetime.fromisoformat(t_str)
                t_start = ensure_utc(t_start)
                t_end = t_start + timedelta(minutes=res_minutes)
                p_val = float(r["price_eur_mwh"])

                if t_end <= start_utc or t_start >= end_utc:
                    continue

                intervals.append(
                    MarketPriceInterval(
                        start_utc=t_start,
                        end_utc=t_end,
                        price_eur_mwh=p_val,
                        bidding_zone=zone,
                        source=self.source,
                        original_resolution_minutes=res_minutes,
                        fetched_at_utc=fetched_at,
                        source_record_id=f"volton-lt-{t_start.strftime('%Y%m%d%H%M')}",
                        is_derived=False,
                        raw_payload_hash=payload_hash,
                    )
                )
            except (ValueError, TypeError):
                continue

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
            request_params={"url": target_url},
            market_intervals=intervals,
        )

    def _get_payload(self, url: str) -> tuple[str, int]:
        last_exc: Exception | None = None
        backoff = self.initial_backoff_sec

        for attempt in range(1, self.max_retries + 1):
            try:
                if self._custom_client is not None:
                    resp = self._custom_client.get(url, timeout=self.timeout)
                else:
                    with httpx.Client(timeout=self.timeout) as client:
                        resp = client.get(url)

                if resp.status_code >= 400:
                    raise PriceProviderError(
                        f"Volton API HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                return resp.text, resp.status_code

            except Exception as exc:
                last_exc = exc
                log.warning("Volton fetch failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0

        raise PriceProviderError(
            f"Volton API unavailable after {self.max_retries} attempts: {last_exc}"
        ) from last_exc
