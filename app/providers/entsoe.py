"""ENTSO-E Transparency Platform day-ahead price provider.

API Documentation:
  https://transparency.entsoe.eu/content/static_content/download?path=/Static%20content/web%20api/Guide.html

Details:
  - Document type: A44 (Price Document / Day-ahead prices)
  - Time intervals: UTC formatted as YYYYMMDDHHMM
  - Response format: XML (Publication_MarketDocument)
  - Bidding zone EIC for Lithuania (LT): 10YLT-1001A0008Q
  - Supports 15-minute (PT15M) and 60-minute (PT60M) resolutions
  - Exponential backoff with retry on transient network / 5xx errors
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.core.timegrid import ensure_utc
from app.models.domain import PriceFetchResult, PricePoint
from app.providers.price_provider import PriceProvider, PriceProviderError

log = logging.getLogger(__name__)

# Standard bidding zone EIC codes (Area / Bidding Zone Domain)
BIDDING_ZONE_EIC: dict[str, str] = {
    "LT": "10YLT-1001A0008Q",  # Lithuania
    "LV": "10YLV-1001A00074",  # Latvia
    "EE": "10Y1001A1001A39I",  # Estonia
    "FI": "10YFI-1--------U",  # Finland
    "SE4": "10Y1001A1001A47J",  # Sweden 4
    "PL": "10YPL-AREA-----S",  # Poland
}

DEFAULT_BASE_URL = "https://web-api.tp.entsoe.eu/api"


def _strip_ns(tag: str) -> str:
    """Return tag name without XML namespace."""
    return tag.split("}")[-1] if "}" in tag else tag


def _find_elem(root: ET.Element, name: str) -> ET.Element | None:
    """Find first descendant matching tag name ignoring namespace."""
    for elem in root.iter():
        if _strip_ns(elem.tag) == name:
            return elem
    return None


def _find_all(elem: ET.Element, name: str) -> list[ET.Element]:
    """Find all direct or indirect children matching tag name ignoring namespace."""
    return [e for e in elem.iter() if _strip_ns(e.tag) == name]


def _text(elem: ET.Element, child_name: str, default: str = "") -> str:
    child = _find_elem(elem, child_name)
    return child.text.strip() if child is not None and child.text else default


def parse_resolution_to_timedelta(res_str: str) -> timedelta:
    """Parse ISO-8601 duration string (PT15M, PT60M, PT1H)."""
    s = res_str.upper().strip()
    if s in ("PT15M", "15M"):
        return timedelta(minutes=15)
    elif s in ("PT60M", "60M", "PT1H", "1H"):
        return timedelta(minutes=60)
    elif s in ("PT30M", "30M"):
        return timedelta(minutes=30)
    else:
        raise ValueError(f"Unsupported ENTSO-E resolution: {res_str}")


class EntsoePriceProvider(PriceProvider):
    source = "entsoe"

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        initial_backoff_sec: float = 1.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_token = api_token.strip() if api_token else None
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.initial_backoff_sec = initial_backoff_sec
        self._custom_client = http_client

    def _resolve_eic(self, bidding_zone: str) -> str:
        zone = bidding_zone.upper().strip()
        if zone in BIDDING_ZONE_EIC:
            return BIDDING_ZONE_EIC[zone]
        # If user passed a full 16-character EIC code directly
        if len(zone) == 16:
            return zone
        raise PriceProviderError(
            f"Unknown bidding zone: {bidding_zone}. Known zones: {list(BIDDING_ZONE_EIC.keys())}"
        )

    def fetch_day_ahead(
        self,
        bidding_zone: str,
        start_utc: datetime,
        end_utc: datetime,
    ) -> PriceFetchResult:
        if not self.api_token:
            raise PriceProviderError(
                "ENTSO-E API token is not configured. Set TES_ENTSOE_API_TOKEN in .env or pass to EntsoePriceProvider."
            )

        start = ensure_utc(start_utc)
        end = ensure_utc(end_utc)
        if end <= start:
            raise ValueError(f"end_utc must be after start_utc: {start} .. {end}")

        eic = self._resolve_eic(bidding_zone)
        # ENTSO-E format: YYYYMMDDHHMM (UTC)
        p_start_str = start.strftime("%Y%m%d%H%M")
        p_end_str = end.strftime("%Y%m%d%H%M")

        params: dict[str, str] = {
            "documentType": "A44",  # Day-ahead prices
            "in_Domain": eic,
            "out_Domain": eic,
            "periodStart": p_start_str,
            "periodEnd": p_end_str,
            "securityToken": self.api_token,
        }

        # Safe params for logging / audit without leaking the secret token
        audit_params = {k: (v if k != "securityToken" else "***") for k, v in params.items()}

        raw_xml, status_code = self._execute_request(params, audit_params)
        points = self.parse_xml_response(raw_xml, bidding_zone=bidding_zone)

        return PriceFetchResult(
            source=self.source,
            bidding_zone=bidding_zone,
            period_start_utc=start,
            period_end_utc=end,
            points=points,
            raw_payload=raw_xml,
            raw_content_type="application/xml",
            http_status=status_code,
            request_params=audit_params,
        )

    def _execute_request(
        self,
        params: dict[str, str],
        audit_params: dict[str, str],
    ) -> tuple[str, int]:
        backoff = self.initial_backoff_sec
        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            log.info(
                "requesting ENTSO-E prices",
                extra={"ctx": {**audit_params, "attempt": attempt}},
            )
            try:
                client = self._custom_client or httpx.Client(timeout=self.timeout)
                try:
                    response = client.get(self.base_url, params=params)
                finally:
                    if self._custom_client is None:
                        client.close()

                if response.status_code == 200:
                    return response.text, 200

                # 4xx errors (client / auth) are not transient (except rate limit 429)
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    error_msg = f"ENTSO-E API returned HTTP {response.status_code}: {response.text[:300]}"
                    log.error(error_msg)
                    raise PriceProviderError(error_msg)

                log.warning(
                    f"ENTSO-E returned HTTP {response.status_code}, retrying in {backoff:.1f}s (attempt {attempt}/{self.max_retries})"
                )

            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                last_exception = exc
                log.warning(
                    f"ENTSO-E request failed ({exc}), retrying in {backoff:.1f}s (attempt {attempt}/{self.max_retries})"
                )

            if attempt < self.max_retries:
                time.sleep(backoff)
                backoff *= 2.0

        raise PriceProviderError(
            f"Failed to fetch data from ENTSO-E after {self.max_retries} attempts: {last_exception}"
        )

    def parse_xml_response(
        self,
        xml_content: str,
        bidding_zone: str,
    ) -> list[PricePoint]:
        """Parse ENTSO-E XML payload into a list of PricePoint domain models."""
        if not xml_content.strip():
            raise PriceProviderError("Empty response received from ENTSO-E API.")

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as exc:
            raise PriceProviderError(f"Failed to parse ENTSO-E XML: {exc}") from exc

        root_tag = _strip_ns(root.tag)

        # Check for error acknowledgement document
        if root_tag == "Acknowledgement_MarketDocument":
            reason_code = _text(root, "code")
            reason_text = _text(root, "text")
            raise PriceProviderError(
                f"ENTSO-E API Error [{reason_code}]: {reason_text or 'No data found for requested period'}"
            )

        if root_tag != "Publication_MarketDocument":
            raise PriceProviderError(f"Unexpected ENTSO-E XML root element: {root.tag}")

        created_str = _text(root, "createdDateTime")
        published_at = (
            datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            if created_str
            else datetime.now(timezone.utc)
        )

        points: list[PricePoint] = []

        # Find all TimeSeries in document
        time_series_list = [e for e in root if _strip_ns(e.tag) == "TimeSeries"]
        if not time_series_list:
            time_series_list = _find_all(root, "TimeSeries")

        for ts in time_series_list:
            currency = _text(ts, "currency_Unit.name", default="EUR")

            # Parse each Period
            periods = [e for e in ts if _strip_ns(e.tag) == "Period"]
            if not periods:
                periods = _find_all(ts, "Period")

            for period in periods:
                # TimeInterval start and end
                time_interval = _find_elem(period, "timeInterval")
                if time_interval is None:
                    continue

                start_str = _text(time_interval, "start")
                end_str = _text(time_interval, "end")
                if not start_str or not end_str:
                    continue

                period_start_utc = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                period_end_utc = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

                resolution_str = _text(period, "resolution", default="PT15M")
                resolution_td = parse_resolution_to_timedelta(resolution_str)
                resolution_minutes = int(resolution_td.total_seconds() // 60)

                # Find points
                point_elems = [e for e in period if _strip_ns(e.tag) == "Point"]
                if not point_elems:
                    point_elems = _find_all(period, "Point")

                for pt in point_elems:
                    pos_str = _text(pt, "position")
                    price_str = _text(pt, "price.amount")
                    if not pos_str or not price_str:
                        continue

                    pos = int(pos_str)
                    price = float(price_str)

                    # position is 1-indexed
                    pt_start = period_start_utc + (pos - 1) * resolution_td
                    pt_end = pt_start + resolution_td

                    if pt_start >= period_end_utc:
                        continue

                    points.append(
                        PricePoint(
                            bidding_zone=bidding_zone,
                            delivery_start_utc=pt_start,
                            delivery_end_utc=pt_end,
                            price_eur_mwh=round(price, 2),
                            resolution_minutes=resolution_minutes,
                            source=self.source,
                            currency=currency,
                            published_at=published_at,
                        )
                    )

        # Sort chronologically
        points.sort(key=lambda p: p.delivery_start_utc)
        return points
