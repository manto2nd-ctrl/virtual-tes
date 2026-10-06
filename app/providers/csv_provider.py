"""CSV day-ahead price provider.

Reads price data from a CSV file or string.
Supports multiple column naming conventions and interval resolutions (15m, 60m).
Preserves the raw CSV payload for permanent audit storage.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from app.core.timegrid import ensure_utc
from app.models.domain import PriceFetchResult, PricePoint
from app.providers.price_provider import PriceProvider, PriceProviderError

log = logging.getLogger(__name__)


def _parse_timestamp(val: str, default_tz: ZoneInfo | None = None) -> datetime:
    """Parse timestamp string into an aware UTC datetime."""
    s = val.strip()
    # Normalize ISO format with trailing Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Try common alternative formats
        formats = [
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d.%m.%Y %H:%M",
            "%d/%m/%Y %H:%M",
        ]
        parsed = None
        for fmt in formats:
            try:
                parsed = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise PriceProviderError(f"Unable to parse timestamp: {val}")
        dt = parsed

    if dt.tzinfo is None:
        if default_tz is not None:
            dt = dt.replace(tzinfo=default_tz)
        else:
            # Assume UTC if naive and no default timezone provided
            dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


class CsvPriceProvider(PriceProvider):
    source = "csv"

    def __init__(
        self,
        file_path: str | Path | None = None,
        csv_content: str | None = None,
        default_resolution_minutes: int = 15,
        default_tz: ZoneInfo | None = None,
    ) -> None:
        if file_path is None and csv_content is None:
            raise ValueError("Either file_path or csv_content must be provided.")
        self.file_path = Path(file_path) if file_path is not None else None
        self.csv_content = csv_content
        self.default_resolution_minutes = default_resolution_minutes
        self.default_tz = default_tz

    def _load_csv_data(self) -> str:
        if self.csv_content is not None:
            return self.csv_content
        if self.file_path is not None:
            if not self.file_path.exists():
                raise PriceProviderError(f"CSV file not found: {self.file_path}")
            try:
                return self.file_path.read_text(encoding="utf-8")
            except Exception as exc:
                raise PriceProviderError(f"Failed to read CSV file {self.file_path}: {exc}") from exc
        raise PriceProviderError("No CSV content or file available.")

    def fetch_day_ahead(
        self,
        bidding_zone: str,
        start_utc: datetime,
        end_utc: datetime,
    ) -> PriceFetchResult:
        start = ensure_utc(start_utc)
        end = ensure_utc(end_utc)
        if end <= start:
            raise ValueError(f"end_utc must be after start_utc: {start} .. {end}")

        raw_text = self._load_csv_data()
        points = self.parse_csv(raw_text, default_bidding_zone=bidding_zone)

        # Filter strictly within requested range and zone
        filtered_points = [
            p for p in points
            if p.bidding_zone.upper() == bidding_zone.upper()
            and p.delivery_start_utc >= start
            and p.delivery_start_utc < end
        ]

        # Generate a mini CSV audit payload for the matched points
        audit_buffer = io.StringIO()
        writer = csv.writer(audit_buffer)
        writer.writerow(["bidding_zone", "delivery_start_utc", "delivery_end_utc", "price_eur_mwh", "resolution_minutes"])
        for p in filtered_points:
            writer.writerow([
                p.bidding_zone,
                p.delivery_start_utc.isoformat(),
                p.delivery_end_utc.isoformat(),
                f"{p.price_eur_mwh:.2f}",
                p.resolution_minutes,
            ])
        filtered_raw = audit_buffer.getvalue()

        return PriceFetchResult(
            source=self.source,
            bidding_zone=bidding_zone,
            period_start_utc=start,
            period_end_utc=end,
            points=filtered_points,
            raw_payload=filtered_raw if filtered_points else raw_text,
            raw_content_type="text/csv",
            http_status=None,
            request_params={
                "file_path": str(self.file_path) if self.file_path else None,
                "default_resolution_minutes": self.default_resolution_minutes,
            },
        )

    def parse_csv(
        self,
        csv_text: str,
        default_bidding_zone: str = "LT",
    ) -> list[PricePoint]:
        csv_clean = csv_text.strip().lstrip("\ufeff")
        if not csv_clean:
            return []

        # Determine delimiter (comma or semicolon)
        first_line = csv_clean.splitlines()[0]
        delimiter = ";" if ";" in first_line and "," not in first_line else ","

        reader = csv.DictReader(io.StringIO(csv_clean), delimiter=delimiter)
        if not reader.fieldnames:
            return []

        # Normalize fieldnames (strip whitespace, lower-case, strip BOM)
        field_map = {name.strip().lstrip("\ufeff").lower(): name for name in reader.fieldnames}

        # Resolve timestamp column
        start_col = None
        for candidate in ["delivery_start_utc", "start_utc", "timestamp_utc", "timestamp", "start", "datetime"]:
            if candidate in field_map:
                start_col = field_map[candidate]
                break

        if not start_col:
            raise PriceProviderError(
                f"CSV must contain a timestamp column (e.g. 'timestamp', 'start_utc', 'delivery_start_utc'). Found: {reader.fieldnames}"
            )

        # Resolve price column
        price_col = None
        for candidate in ["price_eur_mwh", "price", "spot_price", "eur_mwh"]:
            if candidate in field_map:
                price_col = field_map[candidate]
                break

        if not price_col:
            raise PriceProviderError(
                f"CSV must contain a price column (e.g. 'price', 'price_eur_mwh'). Found: {reader.fieldnames}"
            )

        end_col = None
        for candidate in ["delivery_end_utc", "end_utc", "end"]:
            if candidate in field_map:
                end_col = field_map[candidate]
                break

        zone_col = None
        for candidate in ["bidding_zone", "zone", "area"]:
            if candidate in field_map:
                zone_col = field_map[candidate]
                break

        res_col = None
        for candidate in ["resolution_minutes", "resolution", "interval_minutes"]:
            if candidate in field_map:
                res_col = field_map[candidate]
                break

        rows = list(reader)
        raw_items: list[tuple[datetime, datetime | None, float, str, int | None]] = []

        for row_idx, row in enumerate(rows, start=2):
            raw_start = row.get(start_col)
            raw_price = row.get(price_col)
            if not raw_start or not raw_price:
                continue

            try:
                start_dt = _parse_timestamp(raw_start, self.default_tz)
            except Exception as exc:
                raise PriceProviderError(f"Row {row_idx}: invalid start timestamp '{raw_start}': {exc}") from exc

            try:
                price_val = float(raw_price.replace(",", ".").strip())
            except ValueError as exc:
                raise PriceProviderError(f"Row {row_idx}: invalid price '{raw_price}': {exc}") from exc

            end_dt = None
            if end_col and row.get(end_col):
                end_dt = _parse_timestamp(row[end_col], self.default_tz)

            zone = row.get(zone_col).strip().upper() if zone_col and row.get(zone_col) else default_bidding_zone.upper()
            
            res_minutes = None
            if res_col and row.get(res_col):
                try:
                    res_minutes = int(row[res_col].strip())
                except ValueError:
                    pass

            raw_items.append((start_dt, end_dt, price_val, zone, res_minutes))

        # Sort by start timestamp
        raw_items.sort(key=lambda item: item[0])

        points: list[PricePoint] = []
        for i, (s_dt, e_dt, price, zone, res_m) in enumerate(raw_items):
            if e_dt is not None:
                calc_res = int((e_dt - s_dt).total_seconds() // 60)
            elif res_m is not None:
                calc_res = res_m
                e_dt = s_dt + timedelta(minutes=calc_res)
            elif i + 1 < len(raw_items) and raw_items[i + 1][0] > s_dt:
                delta = raw_items[i + 1][0] - s_dt
                calc_res = int(delta.total_seconds() // 60)
                e_dt = s_dt + timedelta(minutes=calc_res)
            else:
                calc_res = self.default_resolution_minutes
                e_dt = s_dt + timedelta(minutes=calc_res)

            points.append(
                PricePoint(
                    bidding_zone=zone,
                    delivery_start_utc=s_dt,
                    delivery_end_utc=e_dt,
                    price_eur_mwh=round(price, 2),
                    resolution_minutes=calc_res,
                    source=self.source,
                )
            )

        return points
