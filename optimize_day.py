"""CLI: Optimize next-day Thermal Energy Storage dispatch schedule.

Computes the lowest-cost operation plan for the complete day horizon
using Linear Programming (LP), respecting physical storage constraints,
heat delivery, grid connection limits, and standing losses.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from app.config.settings import get_settings
from app.database.session import init_db, make_engine, make_session_factory
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.providers.csv_provider import CsvPriceProvider
from app.providers.entsoe import EntsoePriceProvider
from app.providers.mock import MockPriceProvider
from app.providers.price_provider import PriceProvider
from app.services.optimization_service import optimize_day
from app.tes.model import ConstantDischargeLimit, LinearDeratingDischargeLimit


def main() -> int:
    settings = get_settings()
    tz = settings.tz
    tomorrow = (datetime.now(tz) + timedelta(days=1)).date()

    parser = argparse.ArgumentParser(description="Optimize Virtual TES dispatch for one 24-hour day")
    parser.add_argument(
        "--date",
        default="2026-10-06",
        help="Optimization date (YYYY-MM-DD), default: 2026-10-06",
    )
    parser.add_argument(
        "--provider",
        choices=["db", "mock", "entsoe", "csv"],
        default="db",
        help="Price source (default: db)",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Path to CSV file (if --provider csv)",
    )
    parser.add_argument(
        "--initial-soc-percent",
        type=float,
        default=None,
        help=f"Initial SOC %% (default: {settings.tes.initial_soc_percent}%%)",
    )
    parser.add_argument(
        "--target-soc-percent",
        type=float,
        default=None,
        help="Target terminal SOC %% (default: equal to initial SOC)",
    )
    parser.add_argument(
        "--derating",
        action="store_true",
        help="Enable heat exchanger linear derating curve below 30%% SOC",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist optimization run and schedule to database",
    )
    args = parser.parse_args()

    try:
        opt_date = date.fromisoformat(args.date)
    except ValueError:
        print(f"ERROR: Invalid date '{args.date}'. Expected YYYY-MM-DD.", file=sys.stderr)
        return 1

    init_soc_pct = args.initial_soc_percent if args.initial_soc_percent is not None else settings.tes.initial_soc_percent
    init_soc_kwh = (init_soc_pct / 100.0) * settings.tes.capacity_kwh

    if args.target_soc_percent is not None:
        target_soc_kwh = (args.target_soc_percent / 100.0) * settings.tes.capacity_kwh
    else:
        # Default Requirement 3: SOC[T] = SOC[0]
        target_soc_kwh = init_soc_kwh

    curve = LinearDeratingDischargeLimit(derate_start_soc_percent=30.0, min_power_at_soc_min_kw=0.5) if args.derating else ConstantDischargeLimit()

    engine = make_engine(settings.database_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    db_session = session_factory()

    prices = None
    provider: PriceProvider | None = None

    if args.provider == "db":
        from app.core.timegrid import local_day_bounds_utc
        from app.database.repositories import PriceRepository
        from app.models.domain import PricePoint
        s_utc, e_utc = local_day_bounds_utc(opt_date, tz)
        repo = PriceRepository(db_session, tz)
        db_pts = repo.get_prices(settings.bidding_zone, s_utc, e_utc, latest_only=True)
        if not db_pts:
            print(
                f"ERROR: No stored prices found in database for {opt_date} in zone {settings.bidding_zone}. "
                "Collect prices first via collect_prices.py or use --provider mock.",
                file=sys.stderr,
            )
            db_session.close()
            return 1
        prices = [
            PricePoint(
                bidding_zone=p.bidding_zone,
                delivery_start_utc=p.delivery_start_utc,
                delivery_end_utc=p.delivery_end_utc,
                price_eur_mwh=p.price_eur_mwh,
                resolution_minutes=p.resolution_minutes,
                source=p.source,
                published_at=p.published_at,
                currency=p.currency,
            )
            for p in db_pts
        ]
    elif args.provider == "mock":
        provider = MockPriceProvider(tz=tz, resolution_minutes=settings.resolution_minutes, seed=42)
    elif args.provider == "csv":
        if not args.csv_file:
            print("ERROR: --csv-file is required when --provider csv is selected.", file=sys.stderr)
            db_session.close()
            return 1
        provider = CsvPriceProvider(file_path=Path(args.csv_file), default_tz=tz)
    elif args.provider == "entsoe":
        token = settings.entsoe_api_token.get_secret_value() if settings.entsoe_api_token else None
        if not token:
            print("ERROR: ENTSO-E API token is not configured.", file=sys.stderr)
            db_session.close()
            return 1
        provider = EntsoePriceProvider(api_token=token)

    heat_profile = ConstantHeatDemand(value_kw=settings.site.process_heat_demand_kw)
    site_profile = ConstantSiteLoad(value_kw=settings.site.other_loads_kw)

    try:
        outcome = optimize_day(
            day=opt_date,
            tz=tz,
            resolution_minutes=settings.resolution_minutes,
            bidding_zone=settings.bidding_zone,
            tes=settings.tes,
            site=settings.site,
            tariff=settings.tariff,
            heat_profile=heat_profile,
            site_profile=site_profile,
            initial_soc_kwh=init_soc_kwh,
            target_terminal_soc_kwh=target_soc_kwh,
            terminal_soc_condition="exact",
            discharge_limit_curve=curve,
            provider=provider,
            prices=prices,
            session=db_session if not args.no_save else None,
            note="CLI Day-Ahead Optimization",
        )
    except Exception as exc:
        print(f"OPTIMIZATION ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        db_session.close()

    res = outcome.result
    m = res.metrics

    print("=" * 135)
    print(f"VIRTUAL TES GEN0 — DAY-AHEAD LP DISPATCH OPTIMIZATION ({opt_date.isoformat()}, Timezone: {tz})")
    print(
        f"Status: {res.status} | Solver: {res.solver} | "
        f"Initial SOC: {res.initial_soc_kwh:.2f} kWh ({res.initial_soc_kwh / settings.tes.capacity_kwh * 100:.1f}%) | "
        f"Terminal Target: {res.target_terminal_soc_kwh:.2f} kWh ({res.target_terminal_soc_kwh / settings.tes.capacity_kwh * 100:.1f}%)"
    )
    print(
        f"TES Capacity: {settings.tes.capacity_kwh:.1f} kWh | Charge Limit: {settings.tes.max_charge_power_kw:.1f} kW | "
        f"Discharge Limit: {settings.tes.max_discharge_power_kw:.1f} kW | Grid Limit: {settings.site.grid_connection_limit_kw:.1f} kW | "
        f"Derating: {'ON' if args.derating else 'OFF'}"
    )
    print("=" * 135)

    header = (
        f"{'TIME (Local)':<14} "
        f"{'SPOT (EUR)':<11} "
        f"{'EFF (EUR)':<10} "
        f"{'CHARGE (kW)':<12} "
        f"{'DISCH (kW)':<11} "
        f"{'UNMET (kW)':<11} "
        f"{'SOC (kWh)':<10} "
        f"{'SOC (%)':<8} "
        f"{'HEAT (kW)':<10} "
        f"{'SITE (kW)':<10} "
        f"{'GRID (kW)':<10} "
        f"{'COST (EUR)':<10}"
    )
    print(header)
    print("-" * 135)

    for r in res.intervals:
        t_loc = r.start_utc.astimezone(tz).strftime("%H:%M")
        print(
            f"{t_loc:<14} "
            f"{r.spot_price_eur_mwh:10.2f} "
            f"{r.effective_price_eur_mwh:9.2f} "
            f"{r.charge_power_kw:11.2f} "
            f"{r.discharge_power_kw:10.2f} "
            f"{r.unmet_heat_kw:10.2f} "
            f"{r.soc_kwh:9.2f} "
            f"{r.soc_percent:7.1f}% "
            f"{r.heat_demand_kw:9.2f} "
            f"{r.other_site_load_kw:9.2f} "
            f"{r.grid_power_kw:9.2f} "
            f"{r.cost_eur:9.4f}"
        )

    print("=" * 135)
    print("OPTIMIZATION METRICS & ENERGY BALANCE:")
    print(f"  Total Electricity Charged:          {m.total_charge_kwh:8.4f} kWh")
    print(f"  Useful Process Heat Delivered:      {m.useful_heat_delivered_kwh:8.4f} kWh")
    print(f"  Total Unmet Heat:                   {m.total_unmet_heat_kwh:8.4f} kWh")
    print(f"  Heat Supply Reliability:            {m.heat_supply_reliability_percent:8.2f} %")
    print(f"  Initial SOC -> Terminal SOC:        {res.initial_soc_kwh:8.4f} -> {m.terminal_soc_kwh:8.4f} kWh")
    print(f"  Total Storage Energy Withdrawn:     {m.total_storage_energy_withdrawn_kwh:8.4f} kWh")
    print(f"  Charge Conversion Losses (1-eta_c): {m.charge_conversion_losses_kwh:8.4f} kWh")
    print(f"  Discharge Conversion Losses (1/eta_d): {m.discharge_conversion_losses_kwh:8.4f} kWh")
    print(f"  Standing Thermal Losses:            {m.standing_losses_kwh:8.4f} kWh")
    print(f"  Energy Balance Residual:            {m.energy_balance_residual_kwh:8.6e} kWh (tolerance: < 1e-5)")
    print(f"  Total Electricity Cost:             {m.electricity_cost_eur:8.4f} EUR")
    print(f"  Effective Cost per MWh Useful Heat: {m.cost_eur_per_mwh_useful_heat:8.2f} EUR/MWh")
    if outcome.run_id:
        print(f"  Database Optimization Run ID:       {outcome.run_id}")
    print("=" * 135)
    return 0


if __name__ == "__main__":
    sys.exit(main())
