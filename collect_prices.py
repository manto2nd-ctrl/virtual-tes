"""CLI: Collect and store day-ahead electricity prices.

Supports:
- ENTSO-E Transparency Platform API (real market data)
- CSV file import
- Deterministic mock prices

Stores raw payload audit in `raw_market_data` and versioned prices in `day_ahead_prices`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from app.config.settings import get_settings
from app.database.session import init_db, make_engine, make_session_factory
from app.providers.csv_provider import CsvPriceProvider
from app.providers.entsoe import EntsoePriceProvider
from app.providers.mock import MockPriceProvider
from app.providers.price_provider import PriceProvider, PriceProviderError
from app.services.price_service import collect_day_ahead_prices


def main() -> int:
    settings = get_settings()
    tz = settings.tz
    tomorrow = (datetime.now(tz) + timedelta(days=1)).date()

    parser = argparse.ArgumentParser(description="Collect day-ahead electricity prices and store in database")
    parser.add_argument(
        "--provider",
        choices=["entsoe", "mock", "csv"],
        default="mock" if not settings.entsoe_api_token else "entsoe",
        help="Price source provider (default: entsoe if token configured, else mock)",
    )
    parser.add_argument(
        "--date",
        default=tomorrow.isoformat(),
        help=f"Target calendar date (YYYY-MM-DD), default: tomorrow ({tomorrow.isoformat()})",
    )
    parser.add_argument(
        "--zone",
        default=settings.bidding_zone,
        help=f"Bidding zone code (default: {settings.bidding_zone})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="ENTSO-E security token (overrides TES_ENTSOE_API_TOKEN)",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Path to CSV file (required if --provider csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and display prices without writing to database",
    )
    args = parser.parse_args()

    try:
        target_date = date.fromisoformat(args.date)
    except ValueError:
        print(f"ERROR: Invalid date format '{args.date}'. Expected YYYY-MM-DD.", file=sys.stderr)
        return 1

    bidding_zone = args.zone.upper().strip()

    # Instantiate chosen provider
    provider: PriceProvider
    if args.provider == "entsoe":
        token = args.token or (settings.entsoe_api_token.get_secret_value() if settings.entsoe_api_token else None)
        if not token:
            print(
                "ERROR: ENTSO-E API token is required. Pass --token <TOKEN> or set TES_ENTSOE_API_TOKEN in .env",
                file=sys.stderr,
            )
            return 1
        provider = EntsoePriceProvider(api_token=token)
    elif args.provider == "csv":
        if not args.csv_file:
            print("ERROR: --csv-file is required when --provider csv is selected.", file=sys.stderr)
            return 1
        provider = CsvPriceProvider(file_path=Path(args.csv_file), default_tz=tz)
    elif args.provider == "mock":
        provider = MockPriceProvider(tz=tz, resolution_minutes=settings.resolution_minutes, seed=42)
    else:
        print(f"ERROR: Unknown provider '{args.provider}'", file=sys.stderr)
        return 1

    engine = make_engine(settings.database_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    session = session_factory()

    print("=" * 95)
    print(f"COLLECTING DAY-AHEAD PRICES FOR {target_date.isoformat()} ({tz})")
    print(f"Provider: {provider.source.upper()} | Bidding Zone: {bidding_zone}")
    print("=" * 95)

    try:
        if args.dry_run:
            from app.core.timegrid import local_day_bounds_utc
            s_utc, e_utc = local_day_bounds_utc(target_date, tz)
            fetch_res = provider.fetch_day_ahead(bidding_zone, s_utc, e_utc)
            points = fetch_res.points
            inserted = 0
            duplicates = 0
            corrected: list[dict] = []
            audit_id = None
        else:
            res = collect_day_ahead_prices(
                provider=provider,
                bidding_zone=bidding_zone,
                session=session,
                tz=tz,
                target_date=target_date,
            )
            points = res.points
            inserted = res.inserted
            duplicates = res.duplicates
            corrected = res.corrected_versions
            audit_id = res.raw_market_data_id
    except PriceProviderError as exc:
        print(f"PRICE PROVIDER ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()

    if not points:
        print(f"WARNING: No price points returned for {target_date} in zone {bidding_zone}.")
        return 0

    prices = [p.price_eur_mwh for p in points]
    min_p = min(prices)
    max_p = max(prices)
    avg_p = sum(prices) / len(prices)

    print(f"\nFETCH SUMMARY:")
    print(f"  Total Intervals:    {len(points)}")
    print(f"  Resolution:         {points[0].resolution_minutes} minutes")
    print(f"  Inserted:           {inserted} (dry-run={args.dry_run})")
    print(f"  Duplicates Skipped: {duplicates}")
    print(f"  Price Corrections:  {len(corrected)}")
    print(f"  Raw Audit ID:       {audit_id or 'N/A'}")
    print(f"  Min Price:          {min_p:6.2f} EUR/MWh")
    print(f"  Max Price:          {max_p:6.2f} EUR/MWh")
    print(f"  Average Price:      {avg_p:6.2f} EUR/MWh")
    print("\n" + "-" * 95)
    print(f"{'LOCAL TIME (' + str(tz) + ')':<28} {'UTC TIME':<26} {'RESOLUTION':<12} {'PRICE (EUR/MWh)':>16}")
    print("-" * 95)

    for p in points:
        loc_str = p.delivery_start_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M") + " - " + p.delivery_end_utc.astimezone(tz).strftime("%H:%M")
        utc_str = p.delivery_start_utc.strftime("%Y-%m-%d %H:%M") + "Z"
        print(f"{loc_str:<28} {utc_str:<26} {p.resolution_minutes}m{'':<7} {p.price_eur_mwh:14.2f} EUR")

    print("-" * 95)
    print(f"Successfully collected {len(points)} price intervals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
