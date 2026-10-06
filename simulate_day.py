"""CLI: Run a 24-hour simulation demonstration of the Virtual TES.

Outputs the required table:
    timestamp
    mock electricity price
    charge power
    discharge power
    SOC kWh
    SOC %
    heat demand
    site load
    grid power
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from zoneinfo import ZoneInfo

from app.config.settings import get_settings
from app.database.session import init_db, make_engine, make_session_factory
from app.process.heat_demand import ConstantHeatDemand
from app.process.site_load import ConstantSiteLoad
from app.providers.mock import MockPriceProvider
from app.services.simulation_service import simulate_day
from app.simulation.strategies import CheapestIntervalsStrategy, HeatFollowingStrategy


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate Virtual TES for one 24-hour day")
    parser.add_argument("--date", default="2026-10-06", help="Simulation date (YYYY-MM-DD), default: 2026-10-06")
    parser.add_argument("--strategy", choices=["cheapest_n", "heat_following"], default="cheapest_n",
                        help="Dispatch strategy (default: cheapest_n)")
    parser.add_argument("--n-cheapest", type=int, default=None,
                        help="Number of cheapest intervals to charge (for cheapest_n strategy)")
    parser.add_argument("--save", action="store_true", default=False,
                        help="Persist simulation run and prices to database")
    args = parser.parse_args()

    sim_date = date.fromisoformat(args.date)
    settings = get_settings()
    tz = settings.tz

    # Provider and profiles matching requirements for first demonstration:
    # TES capacity = 15 kWh, max charge = 9 kW, max discharge = 3 kW, initial SOC = 50%
    # heat demand = 1.5 kW, grid limit = 12 kW, time resolution = 15 minutes
    provider = MockPriceProvider(tz=tz, resolution_minutes=settings.resolution_minutes, seed=42)
    heat_profile = ConstantHeatDemand(value_kw=settings.site.process_heat_demand_kw)
    site_profile = ConstantSiteLoad(value_kw=settings.site.other_loads_kw)

    if args.strategy == "cheapest_n":
        strategy = CheapestIntervalsStrategy(n_intervals=args.n_cheapest)
    else:
        strategy = HeatFollowingStrategy()

    session = None
    if args.save:
        engine = make_engine(settings.database_url)
        init_db(engine)
        session_factory = make_session_factory(engine)
        session = session_factory()

    try:
        outcome = simulate_day(
            day=sim_date,
            tz=tz,
            resolution_minutes=settings.resolution_minutes,
            bidding_zone=settings.bidding_zone,
            tes=settings.tes,
            site=settings.site,
            tariff=settings.tariff,
            provider=provider,
            heat_profile=heat_profile,
            site_profile=site_profile,
            strategy=strategy,
            initial_soc_kwh=settings.tes.initial_soc_kwh,
            session=session,
            note="CLI 24-hour simulation demo",
        )
    finally:
        if session is not None:
            session.close()

    result = outcome.result
    summary = result.summary

    # Print formatted output table
    print("=" * 110)
    print(f"VIRTUAL TES GEN0 — 24-HOUR SIMULATION ({sim_date.isoformat()}, Timezone: {tz})")
    print(f"Strategy: {strategy.name} | Initial SOC: {result.initial_soc_kwh:.2f} kWh ({result.initial_soc_kwh / settings.tes.capacity_kwh * 100:.1f}%)")
    print(f"TES Capacity: {settings.tes.capacity_kwh:.1f} kWh | Charge Limit: {settings.tes.max_charge_power_kw:.1f} kW | Discharge Limit: {settings.tes.max_discharge_power_kw:.1f} kW | Grid Limit: {settings.site.grid_connection_limit_kw:.1f} kW")
    print("=" * 110)
    header = (
        f"{'TIME (Local)':<14} "
        f"{'PRICE (EUR/MWh)':<16} "
        f"{'CHARGE (kW)':<13} "
        f"{'DISCHARGE (kW)':<15} "
        f"{'SOC (kWh)':<11} "
        f"{'SOC (%)':<9} "
        f"{'HEAT (kW)':<11} "
        f"{'SITE (kW)':<11} "
        f"{'GRID (kW)':<10}"
    )
    print(header)
    print("-" * 110)

    for row in result.rows:
        local_time_str = row.interval_start_utc.astimezone(tz).strftime("%H:%M")
        print(
            f"{local_time_str:<14} "
            f"{row.spot_price_eur_mwh:>15.2f} "
            f"{row.charge_power_kw:>13.2f} "
            f"{row.discharge_power_kw:>15.2f} "
            f"{row.soc_end_kwh:>11.2f} "
            f"{row.soc_end_percent:>8.1f}% "
            f"{row.heat_demand_kw:>11.2f} "
            f"{row.other_loads_kw:>11.2f} "
            f"{row.grid_power_kw:>10.2f}"
        )

    print("=" * 110)
    print("DAILY SUMMARY:")
    print(f"  Total Heat Demand:          {summary.heat_demand_kwh:.2f} kWh")
    print(f"  Total Useful Heat Delivered: {summary.heat_delivered_kwh:.2f} kWh")
    print(f"  Total Unmet Heat:           {summary.unmet_heat_kwh:.2f} kWh")
    print(f"  Total Electricity Consumed: {summary.electricity_kwh:.2f} kWh")
    print(f"  Total Electricity Cost:     {summary.cost_eur:.3f} EUR")
    if summary.avg_price_paid_eur_mwh is not None:
        print(f"  Avg Electricity Price Paid: {summary.avg_price_paid_eur_mwh:.2f} EUR/MWh")
    print(f"  Average Market Spot Price:  {summary.avg_spot_price_eur_mwh:.2f} EUR/MWh")
    if summary.cost_per_mwh_heat_eur is not None:
        print(f"  Cost per MWh useful heat:   {summary.cost_per_mwh_heat_eur:.2f} EUR/MWh_th")
    print(f"  Standing Thermal Losses:    {summary.standing_losses_kwh:.3f} kWh")
    print(f"  Charge Conversion Losses:   {summary.charge_conversion_losses_kwh:.3f} kWh")
    print(f"  Discharge Conv Losses:      {summary.discharge_conversion_losses_kwh:.3f} kWh")
    print(f"  SOC Range [min, max]:       [{summary.soc_min_kwh:.2f}, {summary.soc_max_kwh:.2f}] kWh ([{summary.soc_min_percent:.1f}%, {summary.soc_max_percent:.1f}%])")
    print(f"  Charging Hours:             {summary.charging_hours:.2f} h ({summary.charging_intervals} intervals)")
    print(f"  Equivalent Cycles:          {summary.equivalent_cycles:.2f}")
    print(f"  Peak Grid Power:            {summary.max_grid_power_kw:.2f} kW (limit: {summary.grid_limit_kw:.2f} kW)")
    print(f"  Grid Limit Violations:      {summary.grid_limit_violations}")
    print(f"  Energy Balance Residual:    {summary.energy_balance_residual_kwh:.2e} kWh (must be near 0)")
    if outcome.run_id:
        print(f"  Database Run ID:            {outcome.run_id}")
    print("=" * 110)

    return 0


if __name__ == "__main__":
    sys.exit(main())
