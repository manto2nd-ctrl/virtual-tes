"""CLI: Historical Backtest, Physical Model Validation, and Baseline Comparison.

Phase 5.5 features:
  1. gen0_engineering_estimate_v1 discharge derating curve (NOT MEASURED -- ENGINEERING ESTIMATE).
  2. Sizing study matrix (10, 15, 20, 30 kWh x 6, 9, 12 kW) across:
     - Mode A: Normalized benchmark (same Pmax(SOC%) for all capacities)
     - Mode B: Fixed Gen0 HX (absolute Pmax(SOC_kwh) curve for 1.5-1.6 m2 helical coil).
  3. Standing loss sensitivity expanded up to 30%/day (labeled as assumptions).
  4. Efficiency sensitivity across RTE (LOW 76.5%, NOMINAL 85.5%, HIGH 93.1%).
  5. Auxiliary load sensitivity (0, 50, 100, 200 W continuous).
  6. Configurable heat demand scenarios (1.0 kW, 1.5 kW, 2.0 kW, variable wood drying profile).
  7. Strict validation against silent fallback to mock data when real DB prices are requested.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from app.backtest.domain import (
    BacktestConfig,
    BacktestReport,
    BacktestStrategyMetrics,
    SizingCombinationResult,
    SizingStudyReport,
)
from app.backtest.engine import BacktestRunner, check_physical_viability
from app.backtest.sensitivity import SensitivityAnalyzer, SizingStudyAnalyzer
from app.config.settings import get_settings
from app.core.timegrid import local_day_bounds_utc
from app.database.session import init_db, make_engine, make_session_factory
from app.models.domain import PricePoint
from app.process.heat_demand import get_heat_demand_scenario
from app.process.profiles import PowerProfile
from app.process.site_load import ConstantSiteLoad
from app.providers.csv_provider import CsvPriceProvider
from app.providers.entsoe import EntsoePriceProvider
from app.providers.mock import MockPriceProvider
from app.tes.model import (
    ConstantDischargeLimit,
    DischargeLimitCurve,
    GEN0_CURVE_LABEL,
    GEN0_CURVE_NAME,
    get_gen0_discharge_curve,
)


def load_backtest_prices(
    start_date: date,
    end_date: date,
    provider_name: str,
    tz: ZoneInfo,
    bidding_zone: str,
    csv_file: str | None = None,
) -> list[PricePoint]:
    settings = get_settings()
    s_utc, _ = local_day_bounds_utc(start_date, tz)
    _, e_utc = local_day_bounds_utc(end_date - timedelta(days=1), tz)

    if provider_name == "db":
        from sqlalchemy import func, select
        from app.database.models import DayAheadPrice
        from app.database.repositories import PriceRepository

        engine = make_engine(settings.database_url)
        init_db(engine)
        factory = make_session_factory(engine)
        with factory() as session:
            repo = PriceRepository(session, tz)
            db_pts = repo.get_prices(bidding_zone, s_utc, e_utc, latest_only=True)
            has_full_coverage = False
            if db_pts:
                p_min = min(p.delivery_start_utc for p in db_pts)
                p_max = max(p.delivery_end_utc for p in db_pts)
                if p_min <= s_utc and p_max >= e_utc:
                    has_full_coverage = True

            if not has_full_coverage:
                # Query available range for diagnostic
                min_max = session.execute(
                    select(
                        func.min(DayAheadPrice.delivery_start_utc),
                        func.max(DayAheadPrice.delivery_end_utc),
                        func.count(DayAheadPrice.id),
                    ).where(DayAheadPrice.bidding_zone == bidding_zone)
                ).first()
                min_dt, max_dt, count = min_max if min_max else (None, None, 0)
                avail_days = (max_dt.date() - min_dt.date()).days if (min_dt and max_dt) else 0
                raise ValueError(
                    f"INSUFFICIENT REAL DATA IN DB: requested {start_date} to {end_date} ({(end_date - start_date).days} days) for zone {bidding_zone}, "
                    f"but database only contains {count} prices from {min_dt} to {max_dt} (~{avail_days} days available for {bidding_zone}). "
                    "Cannot fulfill requested backtest without missing data. "
                    "Per Phase 5.5 rules, refusing silent fallback to mock data."
                )
            return [
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
    elif provider_name == "mock":
        prov = MockPriceProvider(tz=tz, resolution_minutes=15, seed=42)
        return prov.fetch_day_ahead(bidding_zone, s_utc, e_utc).points
    elif provider_name == "csv":
        if not csv_file:
            raise ValueError("--csv-file is required when --provider csv is selected.")
        prov = CsvPriceProvider(file_path=Path(csv_file), default_tz=tz)
        return prov.fetch_day_ahead(bidding_zone, s_utc, e_utc).points
    elif provider_name == "entsoe":
        token = settings.entsoe_api_token.get_secret_value() if settings.entsoe_api_token else None
        if not token:
            raise ValueError("ENTSO-E API token is not configured in settings or environment.")
        prov = EntsoePriceProvider(api_token=token)
        return prov.fetch_day_ahead(bidding_zone, s_utc, e_utc).points
    else:
        raise ValueError(f"Unknown provider '{provider_name}'")


def format_row(label: str, vals: list[str]) -> str:
    s = f"  {label:<42}"
    for v in vals:
        s += f"{v:>26}"
    return s


def print_comparison_table(report: BacktestReport, foresite_metrics: BacktestStrategyMetrics | None = None) -> None:
    direct = report.strategies["direct"]
    heuristic = report.strategies["heuristic"]
    rolling = report.strategies["optimized"]

    strat_list = [direct, heuristic, rolling]
    headers = ["DIRECT ELECTRIC", f"CHEAPEST-{report.config.cheapest_n_hours}h HEURISTIC", "OPTIMIZED (ROLLING)"]
    if foresite_metrics is not None:
        strat_list.append(foresite_metrics)
        headers.append("FORESIGHT (BENCHMARK)")

    print("-" * (44 + 26 * len(headers)))
    header_str = f"  {'PERFORMANCE / ECONOMIC METRIC':<42}"
    for h in headers:
        header_str += f"{h:>26}"
    print(header_str)
    print("-" * (44 + 26 * len(headers)))

    # Metrics rows
    print(format_row("Full-Span Physical Capacity", [f"{m.thermal_capacity_full_span_kwh:8.2f} kWh" if m.mode != "baseline_direct" else "N/A" for m in strat_list]))
    print(format_row("Dispatchable Thermal Capacity", [f"{m.dispatchable_capacity_kwh:8.2f} kWh" if m.mode != "baseline_direct" else "N/A" for m in strat_list]))
    print(format_row("Useful Heat Delivered", [f"{m.useful_heat_kwh:8.2f} kWh" for m in strat_list]))
    print(format_row("Unmet Process Heat", [f"{m.unmet_heat_kwh:8.2f} kWh" for m in strat_list]))
    print(format_row("Heat Supply Reliability", [f"{m.heat_supply_reliability_percent:8.2f} %" for m in strat_list]))
    print(format_row("Equivalent Sand Mass", [f"{m.sand_mass_kg:8.1f} kg" if m.sand_mass_kg is not None else "N/A" for m in strat_list]))
    print(format_row("Configured Charge Power Limit", [f"{m.configured_charge_power_limit_kw:8.2f} kW" if m.mode != "baseline_direct" else "N/A" for m in strat_list]))
    print(format_row("Peak Actual Charge Power", [f"{m.peak_actual_charge_power_kw:8.2f} kW" if m.mode != "baseline_direct" else "N/A" for m in strat_list]))
    print(format_row("Grid Connection Limit", [f"{m.grid_connection_limit_kw:8.2f} kW" for m in strat_list]))
    print(format_row("Peak Total Grid Power", [f"{m.peak_total_grid_power_kw:8.2f} kW" for m in strat_list]))
    print(format_row("Backup Heat Delivered", [f"{m.backup_heat_kwh:8.2f} kWh" for m in strat_list]))
    print(format_row("Backup Electricity Consumed", [f"{m.backup_electricity_kwh:8.2f} kWh" for m in strat_list]))
    print(format_row("Backup Electricity Cost", [f"{m.backup_cost_eur:8.2f} EUR" for m in strat_list]))
    print(format_row("TES Electricity Charged", [f"{m.electricity_kwh:8.2f} kWh" for m in strat_list]))
    print(format_row("TES Auxiliary Electricity", [f"{m.tes_auxiliary_kwh:8.2f} kWh" for m in strat_list]))
    print(format_row("Thermal Standing Losses", [f"{m.thermal_standing_loss_kwh:8.2f} kWh" for m in strat_list]))
    print(format_row("Conversion Losses", [f"{m.conversion_loss_kwh:8.2f} kWh" for m in strat_list]))
    print(format_row("Average Paid Electricity Price", [f"{m.average_paid_electricity_price:8.2f} EUR/MWh" for m in strat_list]))
    print(format_row("Raw Electricity Cost", [f"{m.raw_electricity_cost_eur:8.2f} EUR" for m in strat_list]))
    print(format_row("Terminal Inventory Adjustment", [f"{m.terminal_inventory_adjustment_eur:+8.2f} EUR" for m in strat_list]))
    print(format_row("Net Total Evaluated Cost", [f"{m.total_cost_eur:8.2f} EUR" for m in strat_list]))
    print(format_row("Cost per MWh Useful Heat", [f"{m.cost_eur_per_mwh_heat:8.2f} EUR/MWh" for m in strat_list]))
    print(format_row("Savings vs Direct Heating (EUR)", [f"{m.savings_vs_direct_eur:8.2f} EUR" for m in strat_list]))
    print(format_row("Savings vs Direct Heating (%)", [f"{m.savings_vs_direct_percent:8.2f} %" for m in strat_list]))
    print(format_row("Equivalent Full Cycles", [f"{m.equivalent_discharge_cycles:8.2f}" for m in strat_list]))
    print(format_row("Charging Hours", [f"{m.charging_hours:8.2f} h" if m.mode != "baseline_direct" else "N/A" for m in strat_list]))
    print(format_row("Average SOC (kWh / %)", [f"{m.average_soc_kwh:5.2f} kWh ({m.average_soc_percent:4.1f}%)" if m.mode != "baseline_direct" else "N/A" for m in strat_list]))
    print(format_row("Minimum SOC (kWh / %)", [f"{m.minimum_soc_kwh:5.2f} kWh ({m.minimum_soc_percent:4.1f}%)" if m.mode != "baseline_direct" else "N/A" for m in strat_list]))
    print(format_row("Maximum SOC (kWh / %)", [f"{m.maximum_soc_kwh:5.2f} kWh ({m.maximum_soc_percent:4.1f}%)" if m.mode != "baseline_direct" else "N/A" for m in strat_list]))
    print(format_row("Hours Charge OFF During Peak Prices", [f"{m.hours_charge_off_during_high_price_periods:8.2f} h" if m.mode != "baseline_direct" else "N/A" for m in strat_list]))
    print(format_row("Energy Balance Residual", [f"{m.energy_balance_residual_kwh:8.2e} kWh" for m in strat_list]))
    print("-" * (44 + 26 * len(headers)))


def print_sizing_table(report: SizingStudyReport, mode_title: str) -> None:
    print(f"\n{mode_title.upper()} (All capacities refer to FULL-SPAN physical thermal capacity across 80–300 °C; charging subject to site grid headroom):")
    print(f"  {'FULL CAP':<10} {'DISPATCH':<10} {'SAND MASS':<10} {'CONFIG CHG':<12} {'PEAK CHG':<10} {'RELIABILITY':>14} {'UNMET HEAT':>12} {'COST/MWh_HEAT':>16} {'SAVINGS %':>12} {'AVG SOC':>18} {'CYCLES':>10}")
    print("  " + "-" * 140)
    for c in report.combinations:
        soc_str = f"{c.average_soc_kwh:.1f} kWh ({c.average_soc_percent:.1f}%)"
        print(f"  {c.thermal_capacity_full_span_kwh:5.1f} kWh   {c.dispatchable_capacity_kwh:5.1f} kWh   {c.sand_mass_kg:5.1f} kg  {c.configured_charge_power_limit_kw:5.1f} kW    {c.peak_actual_charge_power_kw:5.2f} kW    {c.heat_supply_reliability_percent:13.2f} % {c.unmet_heat_kwh:10.2f} kWh {c.cost_eur_per_mwh_heat:13.2f} EUR/MWh {c.savings_vs_direct_percent:10.2f} % {soc_str:>18} {c.equivalent_cycles:9.2f}")
    print("  " + "-" * 140)

    # Key benchmark indicators (Section 16)
    if report.lowest_cost_combination:
        lc = report.lowest_cost_combination
        print(f"  * Benchmark - Lowest Cost/MWh: {lc.thermal_capacity_full_span_kwh:.0f} kWh full-span ({lc.dispatchable_capacity_kwh:.1f} kWh disp) / {lc.charge_power_kw:.0f} kW -> {lc.cost_eur_per_mwh_heat:.2f} EUR/MWh ({lc.savings_vs_direct_percent:.1f}% savings)")
    if report.highest_reliability_combination:
        hr = report.highest_reliability_combination
        print(f"  * Benchmark - Highest Reliability: {hr.thermal_capacity_full_span_kwh:.0f} kWh full-span ({hr.dispatchable_capacity_kwh:.1f} kWh disp) / {hr.charge_power_kw:.0f} kW -> {hr.heat_supply_reliability_percent:.2f}% reliability")
    if report.lowest_capacity_meeting_99_5_rel:
        lcr = report.lowest_capacity_meeting_99_5_rel
        print(f"  * Benchmark - Lowest Capacity Meeting >=99.5% Reliability: {lcr.thermal_capacity_full_span_kwh:.0f} kWh full-span ({lcr.dispatchable_capacity_kwh:.1f} kWh disp) / {lcr.charge_power_kw:.0f} kW ({lcr.heat_supply_reliability_percent:.2f}% rel)")


def main() -> int:
    settings = get_settings()
    tz = settings.tz

    parser = argparse.ArgumentParser(description="Multi-day Historical Backtest, Physical Model Validation, and Baseline Comparison")
    parser.add_argument("--start-date", default="2026-10-01", help="Backtest start date (YYYY-MM-DD), default: 2026-10-01")
    parser.add_argument("--end-date", default="2026-10-08", help="Backtest end date (YYYY-MM-DD), default: 2026-10-08 (7 days)")
    parser.add_argument("--mode", choices=["realistic_rolling", "perfect_foresight", "both"], default="both", help="Backtest mode (default: both)")
    parser.add_argument("--provider", choices=["db", "mock", "csv", "entsoe"], default="mock", help="Price provider (default: mock)")
    parser.add_argument("--csv-file", type=str, default=None, help="Path to CSV if --provider csv")
    parser.add_argument("--constant-limit", action="store_true", help="Use ConstantDischargeLimit (software test only, not physical Gen0)")
    parser.add_argument("--sizing-mode", choices=["fixed_gen0_hx", "scaled_hx_benchmark", "normalized_benchmark", "both"], default="fixed_gen0_hx", help="Discharge derating sizing mode (default: fixed_gen0_hx)")
    parser.add_argument("--hx-mode", choices=["fixed_gen0_hx", "scaled_hx_benchmark", "normalized_benchmark", "constant"], default="fixed_gen0_hx", help="Heat exchanger mode (default: fixed_gen0_hx)")
    parser.add_argument("--u-val", type=float, default=9.0, help="Overall heat transfer coefficient U [W/(m2*K)] (default: 9.0)")
    parser.add_argument("--airflow", type=float, default=80.0, help="Airflow rate [m3/h] (default: 80.0)")
    parser.add_argument("--backup-heat", action="store_true", help="Enable hypothetical backup electric heater (default: False)")
    parser.add_argument("--heat-demand", choices=["1.0", "1.5", "2.0", "variable", "dryer"], default="1.5", help="Process heat demand scenario (default: 1.5 kW constant)")
    parser.add_argument("--loss-pct", type=float, default=None, help="Standing loss %%/day (default: settings)")
    parser.add_argument("--aux-kw", type=float, default=None, help="Auxiliary power kW (default: settings)")
    parser.add_argument("--sensitivity", action="store_true", help="Run standing loss and efficiency sensitivity analysis")
    parser.add_argument("--physical-sensitivity", action="store_true", help="Run complete physical sensitivity suite (standing losses, RTE, aux load, heat demands)")
    parser.add_argument("--sizing", action="store_true", help="Run systematic TES capacity & charging power sizing matrix")
    args = parser.parse_args()

    try:
        s_date = date.fromisoformat(args.start_date)
        e_date = date.fromisoformat(args.end_date)
    except ValueError as exc:
        print(f"ERROR: Invalid date format: {exc}", file=sys.stderr)
        return 1

    if e_date <= s_date:
        print(f"ERROR: end-date ({e_date}) must be after start-date ({s_date})", file=sys.stderr)
        return 1

    days_count = (e_date - s_date).days

    effective_hx_mode = "constant" if args.constant_limit else args.hx_mode
    u_val = args.u_val if args.u_val is not None else settings.tes.overall_u_w_m2k
    airflow = args.airflow if args.airflow is not None else settings.tes.airflow_m3_h

    # Pluggable discharge limit curve
    curve: DischargeLimitCurve
    curve_name: str
    if effective_hx_mode == "constant":
        curve = ConstantDischargeLimit()
        curve_name = "ConstantDischargeLimit"
    else:
        # Physical Gen0 Helical Coil Air Heat Exchanger model
        curve = get_gen0_discharge_curve(
            mode=effective_hx_mode,
            capacity_kwh=settings.tes.capacity_kwh,
            reference_capacity_kwh=15.0,
            overall_u_w_m2k=u_val,
            airflow_m3_h=airflow,
            hx_area_m2=settings.tes.hx_area_m2,
        )
        curve_name = GEN0_CURVE_NAME

    loss_pct = args.loss_pct if args.loss_pct is not None else settings.tes.standing_loss_percent_per_day
    aux_kw = args.aux_kw if args.aux_kw is not None else settings.tes.auxiliary_power_kw

    tes = settings.tes.model_copy(
        update={
            "standing_loss_percent_per_day": loss_pct,
            "auxiliary_power_kw": aux_kw,
            "overall_u_w_m2k": u_val,
            "airflow_m3_h": airflow,
            "hx_model_mode": effective_hx_mode,
        }
    )

    heat_profile = get_heat_demand_scenario(args.heat_demand, tz)
    site_profile = ConstantSiteLoad(value_kw=settings.site.other_loads_kw)

    try:
        prices = load_backtest_prices(
            start_date=s_date,
            end_date=e_date,
            provider_name=args.provider,
            tz=tz,
            bidding_zone=settings.bidding_zone,
            csv_file=args.csv_file,
        )
    except Exception as exc:
        print(f"PRICE LOAD ERROR: {exc}", file=sys.stderr)
        return 1

    actual_price_source = prices[0].source if prices else args.provider

    config = BacktestConfig(
        start_date=s_date,
        end_date=e_date,
        mode="realistic_rolling",
        initial_soc_kwh=tes.initial_soc_kwh,
        tes_params=tes,
        site_params=settings.site,
        tariff_params=settings.tariff,
        discharge_limit_curve=curve,
        discharge_curve_name=curve_name,
        discharge_curve_mode=effective_hx_mode,
        auxiliary_power_kw=aux_kw,
        cheapest_n_hours=4,
        bidding_zone=settings.bidding_zone,
        price_source=actual_price_source,
        heat_demand_profile_name=heat_profile.name,
        overall_u_w_m2k=u_val,
        airflow_m3_h=airflow,
        backup_heat_enabled=args.backup_heat,
    )

    runner = BacktestRunner(tz=tz)
    try:
        report = runner.run(config, heat_profile, site_profile, prices)
    except Exception as exc:
        print(f"BACKTEST EXECUTION ERROR: {exc}", file=sys.stderr)
        return 1

    foresight_metrics: BacktestStrategyMetrics | None = None
    if args.mode in ("perfect_foresight", "both"):
        foresight_cfg = BacktestConfig(**{**config.__dict__, "mode": "perfect_foresight"})
        foresight_rep = runner.run(foresight_cfg, heat_profile, site_profile, prices)
        foresight_metrics = foresight_rep.strategies["optimized"]

    # Header and Viability Status per Phase 5.6
    opt_m = report.strategies["optimized"]
    status_label = opt_m.status_label

    print("=" * 122)
    print("VIRTUAL TES GEN0 -- PHYSICAL MODEL VALIDATION & BASELINE BENCHMARK (PHASE 5.6)")
    print(f"Period: {s_date.isoformat()} to {e_date.isoformat()} ({days_count} days | {report.total_intervals} intervals of 15m) | Timezone: {tz}")
    print(f"STATUS: [{status_label}]")
    print("")
    print("PHYSICAL STORAGE")
    print(f"  Temperature span:                 {tes.physical_temperature_min_c:.1f} -> {tes.physical_temperature_max_c:.1f} °C")
    print(f"  Full-span thermal capacity:       {tes.thermal_capacity_full_span_kwh:.2f} kWh")
    print(f"  Equivalent quartz sand mass:      {tes.sand_mass_kg:.1f} kg")
    print("")
    print("DISPATCH WINDOW")
    print(f"  Optimizer SOC range:              {tes.optimizer_soc_min_fraction*100.0:.1f} -> {tes.optimizer_soc_max_fraction*100.0:.1f} %")
    print(f"  Minimum stored energy:             {tes.soc_min_energy_kwh:.2f} kWh")
    print(f"  Maximum stored energy:            {tes.soc_max_energy_kwh:.2f} kWh")
    print(f"  Dispatchable thermal capacity:    {tes.dispatchable_capacity_kwh:.2f} kWh")
    print(f"  Temperature at SOC minimum:       {tes.temperature_at_optimizer_min_soc_c:.2f} °C")
    print("")
    print("CHARGING / GRID")
    print(f"  Configured charge power limit:    {opt_m.configured_charge_power_limit_kw:.2f} kW")
    print(f"  Peak actual charge power:         {opt_m.peak_actual_charge_power_kw:.2f} kW")
    print(f"  Grid connection limit:            {opt_m.grid_connection_limit_kw:.2f} kW")
    print(f"  Peak total grid power:            {opt_m.peak_total_grid_power_kw:.2f} kW")
    print("")
    print("MAJOR MODELING ASSUMPTIONS:")
    print(f"  * Price Source: {args.provider.upper()} ({settings.bidding_zone})")
    print(f"  * Heat Exchanger: {curve_name} (Mode: {effective_hx_mode}, Area={opt_m.hx_area_m2:.2f} m2, U={u_val:.1f} W/m2K, Airflow={airflow:.1f} m3/h)")
    print(f"  * Backup Heater: {'ENABLED (Hypothetical)' if args.backup_heat else 'DISABLED (Default Physical Gen0)'}")
    print(f"  * Standing Loss Assumption: {loss_pct:.1f}%/day [provisional engineering assumption]")
    print(f"  * Provisional Efficiencies: eta_charge={tes.charge_efficiency:.2f}, eta_discharge={tes.discharge_efficiency:.2f} (RTE={tes.round_trip_efficiency*100:.1f}%)")
    print(f"  * Auxiliary Electrical Load: {aux_kw*1000:.0f} W continuous")
    print(f"  * Process Heat Demand: {heat_profile.name} (Dryer reference)")
    if opt_m.viability_caveats:
        print("MODELING CAVEATS:")
        for c in opt_m.viability_caveats:
            print(f"  * {c}")
    print("=" * 122)

    print_comparison_table(report, foresight_metrics)

    # Physical Sensitivity Sweeps (Section 5, 6, 11, 17)
    if args.sensitivity or args.physical_sensitivity:
        print("\n" + "=" * 122)
        print("PHYSICAL SENSITIVITY ANALYSIS (Section 17)")
        print("=" * 122)
        analyzer = SensitivityAnalyzer(tz=tz)

        print("\n1. Standing Loss Sensitivity (Rolling Mode, 1% to 30%/day):")
        print(f"  {'SCENARIO':<46} {'STANDING LOSS':>16} {'NET COST':>14} {'SAVINGS (EUR)':>16} {'SAVINGS (%)':>14} {'CYCLES':>10}")
        print("  " + "-" * 118)
        loss_cases = analyzer.sweep_standing_losses(config, heat_profile, site_profile, prices)
        for c in loss_cases:
            print(f"  {c.scenario_name:<46} {c.standing_losses_kwh:14.2f} kWh {c.total_cost_eur:12.2f} EUR {c.savings_vs_direct_eur:14.2f} EUR {c.savings_vs_direct_percent:13.2f} % {c.equivalent_cycles:9.2f}")

        print("\n2. Round-Trip Efficiency Sensitivity (eta_c * eta_d):")
        print(f"  {'SCENARIO':<48} {'RTE':>8} {'NET COST':>14} {'SAVINGS (EUR)':>16} {'SAVINGS (%)':>14} {'CYCLES':>10}")
        print("  " + "-" * 122)
        eff_cases = analyzer.sweep_efficiencies(config, heat_profile, site_profile, prices)
        for c in eff_cases:
            print(f"  {c.scenario_name:<48} {c.parameter_value*100:6.1f} % {c.total_cost_eur:12.2f} EUR {c.savings_vs_direct_eur:14.2f} EUR {c.savings_vs_direct_percent:13.2f} % {c.equivalent_cycles:9.2f}")

        if args.physical_sensitivity:
            print("\n3. Auxiliary Electrical Load Sensitivity (0 W to 200 W continuous):")
            print(f"  {'SCENARIO':<36} {'ELEC CONS':>16} {'NET COST':>14} {'SAVINGS (EUR)':>16} {'SAVINGS (%)':>14} {'CYCLES':>10}")
            print("  " + "-" * 108)
            aux_cases = analyzer.sweep_auxiliary_power(config, heat_profile, site_profile, prices)
            for c in aux_cases:
                print(f"  {c.scenario_name:<36} {c.total_electricity_kwh:14.2f} kWh {c.total_cost_eur:12.2f} EUR {c.savings_vs_direct_eur:14.2f} EUR {c.savings_vs_direct_percent:13.2f} % {c.equivalent_cycles:9.2f}")

            print("\n4. Process Heat Demand Scenario Sensitivity:")
            print(f"  {'SCENARIO':<44} {'HEAT DEMAND':>14} {'NET COST':>14} {'COST/MWh':>16} {'RELIABILITY':>14} {'SAVINGS (%)':>12}")
            print("  " + "-" * 116)
            hd_cases = analyzer.sweep_heat_demands(config, site_profile, prices)
            for c in hd_cases:
                print(f"  {c.scenario_name:<44} {c.total_useful_heat_kwh:12.2f} kWh {c.total_cost_eur:12.2f} EUR {c.cost_eur_per_mwh_heat:14.2f} EUR {c.heat_supply_reliability_percent:12.2f} % {c.savings_vs_direct_percent:10.2f} %")
        print("=" * 122)

    # Systematic Sizing Sweeps (Section 8, 9, 16)
    if args.sizing:
        print("\n" + "=" * 122)
        print("SYSTEMATIC TES SIZING STUDY MATRIX (Section 8, 9, 16)")
        print("=" * 122)
        sizing_analyzer = SizingStudyAnalyzer(tz=tz)
        sizing_reports = sizing_analyzer.run_sizing_sweep(
            base_config=config,
            heat_profile=heat_profile,
            site_profile=site_profile,
            prices=prices,
            capacities_kwh=[10.0, 15.0, 20.0, 30.0],
            charge_powers_kw=[6.0, 9.0, 12.0],
            sizing_mode=args.sizing_mode,
        )

        for mode_key, s_report in sizing_reports.items():
            mode_name = "Mode A -- Normalized Benchmark" if mode_key == "normalized_benchmark" else "Mode B -- Fixed Gen0 Heat Exchanger (1.5 m2 coil)"
            print_sizing_table(s_report, mode_name)
        print("=" * 122)

    return 0


if __name__ == "__main__":
    sys.exit(main())
