"""ORM table definitions.

Append-only policy: historical tables are protected by SQLite triggers (see
``session.IMMUTABLE_TABLES``) that abort any UPDATE or DELETE. Changing parameters and
re-running produces NEW rows / runs; nothing is overwritten.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UTCDateTime


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- market data
class RawMarketData(Base):
    __tablename__ = "raw_market_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    bidding_zone: Mapped[str] = mapped_column(String(16))
    request_params: Mapped[dict] = mapped_column(JSON, default=dict)
    period_start_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    period_end_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_type: Mapped[str] = mapped_column(String(64), default="application/json")
    payload: Mapped[str] = mapped_column(Text)
    payload_sha256: Mapped[str] = mapped_column(String(64), index=True)


class DayAheadPrice(Base):
    __tablename__ = "day_ahead_prices"
    __table_args__ = (
        UniqueConstraint("bidding_zone", "source", "delivery_start_utc", "resolution_minutes", "version",
                         name="uq_price_interval_version"),
        Index("ix_price_zone_start", "bidding_zone", "delivery_start_utc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bidding_zone: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(32))
    delivery_start_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    delivery_end_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    delivery_start_local: Mapped[str] = mapped_column(String(32))  # ISO-8601 with offset, display aid
    resolution_minutes: Mapped[int] = mapped_column(Integer)
    price_eur_mwh: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3), default="EUR")
    version: Mapped[int] = mapped_column(Integer, default=1)
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    raw_market_data_id: Mapped[int | None] = mapped_column(ForeignKey("raw_market_data.id"), nullable=True)


# --------------------------------------------------------------------------- configuration
class TariffConfig(Base):
    """Versioned tariff configuration. Latest row = active."""

    __tablename__ = "tariff_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    supplier_markup_eur_mwh: Mapped[float] = mapped_column(Float)
    variable_grid_fee_eur_mwh: Mapped[float] = mapped_column(Float)
    variable_tax_eur_mwh: Mapped[float] = mapped_column(Float)
    note: Mapped[str | None] = mapped_column(String(255), nullable=True)


class TESConfig(Base):
    """Versioned TES + grid configuration. Insert-only; latest row = active."""

    __tablename__ = "tes_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    name: Mapped[str] = mapped_column(String(64), default="default")
    capacity_semantics_version: Mapped[str] = mapped_column(String(32), default="full_span_v1")
    physical_temperature_min_c: Mapped[float] = mapped_column(Float, default=80.0)
    physical_temperature_max_c: Mapped[float] = mapped_column(Float, default=300.0)
    thermal_capacity_full_span_kwh: Mapped[float] = mapped_column(Float, default=15.0)
    optimizer_soc_min_fraction: Mapped[float] = mapped_column(Float, default=0.10)
    optimizer_soc_max_fraction: Mapped[float] = mapped_column(Float, default=1.00)
    capacity_kwh: Mapped[float] = mapped_column(Float)
    max_charge_power_kw: Mapped[float] = mapped_column(Float)
    max_discharge_power_kw: Mapped[float] = mapped_column(Float)
    charge_efficiency: Mapped[float] = mapped_column(Float)
    charge_efficiency_is_provisional: Mapped[bool] = mapped_column(Boolean, default=True)
    discharge_efficiency: Mapped[float] = mapped_column(Float)
    discharge_efficiency_is_provisional: Mapped[bool] = mapped_column(Boolean, default=True)
    auxiliary_power_kw: Mapped[float] = mapped_column(Float, default=0.05)
    standing_loss_percent_per_day: Mapped[float] = mapped_column(Float)
    standing_loss_is_provisional: Mapped[bool] = mapped_column(Boolean, default=True)
    standing_loss_model: Mapped[str] = mapped_column(String(32), default="provisional_fractional")
    standing_loss_fixed_kw: Mapped[float] = mapped_column(Float, default=0.0)
    hx_model_mode: Mapped[str] = mapped_column(String(32), default="fixed_gen0_hx")
    hx_area_m2: Mapped[float] = mapped_column(Float, default=1.55)
    overall_u_w_m2k: Mapped[float] = mapped_column(Float, default=9.0)
    airflow_m3_h: Mapped[float] = mapped_column(Float, default=80.0)
    air_inlet_temperature_c: Mapped[float] = mapped_column(Float, default=40.0)
    soc_min_percent: Mapped[float] = mapped_column(Float)
    soc_max_percent: Mapped[float] = mapped_column(Float)
    initial_soc_percent: Mapped[float] = mapped_column(Float)
    grid_connection_limit_kw: Mapped[float] = mapped_column(Float)
    note: Mapped[str | None] = mapped_column(String(255), nullable=True)


# --------------------------------------------------------------------------- live state / profiles
class TESStateRecord(Base):
    __tablename__ = "tes_state"
    __table_args__ = (UniqueConstraint("source", "timestamp_utc", name="uq_tes_state"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), default="virtual")  # virtual | plc
    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    soc_kwh: Mapped[float] = mapped_column(Float)
    soc_percent: Mapped[float] = mapped_column(Float)
    charge_power_kw: Mapped[float] = mapped_column(Float)
    discharge_power_kw: Mapped[float] = mapped_column(Float)
    heat_demand_kw: Mapped[float] = mapped_column(Float)
    storage_loss_kw: Mapped[float] = mapped_column(Float)
    other_loads_kw: Mapped[float] = mapped_column(Float)
    grid_power_kw: Mapped[float] = mapped_column(Float)
    simulation_run_id: Mapped[str | None] = mapped_column(ForeignKey("simulation_runs.id"), nullable=True)


class HeatDemandRecord(Base):
    __tablename__ = "heat_demand"
    __table_args__ = (UniqueConstraint("profile_name", "interval_start_utc", name="uq_heat_demand"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_name: Mapped[str] = mapped_column(String(64))
    interval_start_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    interval_end_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    heat_demand_kw: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class SiteLoadRecord(Base):
    __tablename__ = "site_load"
    __table_args__ = (UniqueConstraint("profile_name", "interval_start_utc", name="uq_site_load"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_name: Mapped[str] = mapped_column(String(64))
    interval_start_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    interval_end_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    site_load_kw: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


# --------------------------------------------------------------------------- optimization (Phase 4)
class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    horizon_start_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    horizon_end_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    bidding_zone: Mapped[str] = mapped_column(String(16))
    config_snapshot: Mapped[dict] = mapped_column(JSON)
    inputs_sha256: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32))
    solver: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    optimization_price_basis: Mapped[str] = mapped_column(String(32), default="EFFECTIVE_VARIABLE_PRICE")
    objective_eur: Mapped[float | None] = mapped_column(Float, nullable=True)
    initial_soc_kwh: Mapped[float] = mapped_column(Float)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    schedule: Mapped[list["OptimizationSchedule"]] = relationship(
        back_populates="run", order_by="OptimizationSchedule.interval_start_utc"
    )


class OptimizationSchedule(Base):
    __tablename__ = "optimization_schedule"
    __table_args__ = (UniqueConstraint("run_id", "interval_start_utc", name="uq_opt_schedule"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("optimization_runs.id"), index=True)
    interval_start_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    interval_end_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    spot_price_eur_mwh: Mapped[float] = mapped_column(Float)
    effective_price_eur_mwh: Mapped[float] = mapped_column(Float)
    charge_power_kw: Mapped[float] = mapped_column(Float)
    discharge_power_kw: Mapped[float] = mapped_column(Float)
    unmet_heat_kw: Mapped[float] = mapped_column(Float, default=0.0)
    predicted_soc_kwh: Mapped[float] = mapped_column(Float)
    predicted_soc_percent: Mapped[float] = mapped_column(Float)
    heat_demand_kw: Mapped[float] = mapped_column(Float)
    other_site_load_kw: Mapped[float] = mapped_column(Float)
    auxiliary_load_kw: Mapped[float] = mapped_column(Float, default=0.0)
    grid_power_kw: Mapped[float] = mapped_column(Float)
    cost_eur: Mapped[float] = mapped_column(Float)

    run: Mapped[OptimizationRun] = relationship(back_populates="schedule")


# --------------------------------------------------------------------------- simulation
class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    run_type: Mapped[str] = mapped_column(String(32))  # single_day | backtest
    strategy: Mapped[str] = mapped_column(String(64))
    bidding_zone: Mapped[str] = mapped_column(String(16))
    price_source: Mapped[str] = mapped_column(String(32))
    period_start_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    period_end_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    config_snapshot: Mapped[dict] = mapped_column(JSON)
    inputs_sha256: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32))
    app_version: Mapped[str] = mapped_column(String(16))
    summary: Mapped[dict] = mapped_column(JSON)
    note: Mapped[str | None] = mapped_column(String(255), nullable=True)

    results: Mapped[list["SimulationResultRecord"]] = relationship(
        back_populates="run", order_by="SimulationResultRecord.interval_start_utc")


class SimulationResultRecord(Base):
    __tablename__ = "simulation_results"
    __table_args__ = (UniqueConstraint("run_id", "interval_start_utc", name="uq_sim_result"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("simulation_runs.id"), index=True)
    interval_start_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    interval_end_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    spot_price_eur_mwh: Mapped[float] = mapped_column(Float)
    effective_price_eur_mwh: Mapped[float] = mapped_column(Float)
    heat_demand_kw: Mapped[float] = mapped_column(Float)
    heat_delivered_kw: Mapped[float] = mapped_column(Float)
    unmet_heat_kw: Mapped[float] = mapped_column(Float)
    charge_power_kw: Mapped[float] = mapped_column(Float)
    discharge_power_kw: Mapped[float] = mapped_column(Float)
    storage_loss_kw: Mapped[float] = mapped_column(Float)
    soc_start_kwh: Mapped[float] = mapped_column(Float)
    soc_end_kwh: Mapped[float] = mapped_column(Float)
    soc_end_percent: Mapped[float] = mapped_column(Float)
    other_loads_kw: Mapped[float] = mapped_column(Float)
    grid_power_kw: Mapped[float] = mapped_column(Float)
    electricity_kwh: Mapped[float] = mapped_column(Float)
    cost_eur: Mapped[float] = mapped_column(Float)
    charge_limited_by: Mapped[str] = mapped_column(String(64), default="")
    discharge_limited_by: Mapped[str] = mapped_column(String(64), default="")
    soc_below_min_due_to_losses: Mapped[bool] = mapped_column(Boolean, default=False)

    run: Mapped[SimulationRun] = relationship(back_populates="results")


class RunStatusEvent(Base):
    """Append-only lifecycle status events for runs (simulations and optimizations).

    Because run tables are immutable (append-only), state transitions (e.g.
    QUEUED -> RUNNING -> COMPLETED / FAILED) are modeled as immutable events
    rather than in-place updates.
    """

    __tablename__ = "run_status_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    run_type: Mapped[str] = mapped_column(String(32))  # "simulation" | "optimization"
    status: Mapped[str] = mapped_column(String(32))  # "QUEUED", "RUNNING", "COMPLETED", "FAILED"
    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    message: Mapped[str | None] = mapped_column(String(255), nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)


# --------------------------------------------------------------------------- saved scenarios (Phase 5.7)
class EngineeringScenario(Base):
    """Saved engineering scenarios for Gen0 parameter exploration (Phase 5.7).

    Append-only; versioned. Protected by immutability triggers.
    """

    __tablename__ = "engineering_scenarios"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_scenario_name_version"),
        Index("ix_scenario_name", "name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    parameters: Mapped[dict] = mapped_column(JSON)
    summary: Mapped[dict] = mapped_column(JSON)


# --------------------------------------------------------------------------- live shadow runtime (Phase 5.8.2)
class ShadowTESSession(Base):
    """Persistent Shadow Digital Twin session for live continuous operation."""

    __tablename__ = "shadow_tes_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    status: Mapped[str] = mapped_column(String(16), default="STOPPED")  # STOPPED | RUNNING | PAUSED | ERROR
    created_at_utc: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    started_at_utc: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    stopped_at_utc: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    model_version: Mapped[str] = mapped_column(String(32), default="gen0-shadow-v1")
    config_snapshot_id: Mapped[int | None] = mapped_column(ForeignKey("tes_config.id"), nullable=True)
    market_source: Mapped[str] = mapped_column(String(32), default="LITGRID")

    initial_soc_fraction: Mapped[float] = mapped_column(Float, default=0.50)
    initial_stored_energy_kwh: Mapped[float] = mapped_column(Float, default=7.50)
    initial_sand_temperature_c: Mapped[float] = mapped_column(Float, default=197.8)

    current_soc_fraction: Mapped[float] = mapped_column(Float, default=0.50)
    current_stored_energy_kwh: Mapped[float] = mapped_column(Float, default=7.50)
    current_sand_temperature_c: Mapped[float] = mapped_column(Float, default=197.8)

    last_state_update_utc: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_executed_interval_start_utc: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    process_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    process_heat_demand_kw: Mapped[float] = mapped_column(Float, default=1.50)

    active_schedule_version: Mapped[int] = mapped_column(Integer, default=1)
    active_optimizer_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    optimization_price_basis: Mapped[str] = mapped_column(String(32), default="EFFECTIVE_VARIABLE_PRICE")
    optimization_price_eur_mwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    latest_action_type: Mapped[str] = mapped_column(String(32), default="IDLE")
    latest_reason_code: Mapped[str] = mapped_column(String(64), default="INITIALIZED")
    error_message: Mapped[str | None] = mapped_column(String(255), nullable=True)
    planned_schedule: Mapped[list] = mapped_column(JSON, default=list)


class ShadowIntervalRecord(Base):
    """Immutable record of an executed shadow interval (strictly append-only)."""

    __tablename__ = "shadow_interval_records"
    __table_args__ = (
        UniqueConstraint("shadow_session_id", "interval_start_utc", name="uq_shadow_interval"),
        Index("ix_shadow_session_start", "shadow_session_id", "interval_start_utc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shadow_session_id: Mapped[str] = mapped_column(ForeignKey("shadow_tes_sessions.id"), index=True)
    interval_start_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    interval_end_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    duration_hours: Mapped[float] = mapped_column(Float)

    spot_price_eur_mwh: Mapped[float] = mapped_column(Float)
    effective_price_eur_mwh: Mapped[float] = mapped_column(Float)
    optimization_price_eur_mwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    optimization_price_basis: Mapped[str] = mapped_column(String(32), default="EFFECTIVE_VARIABLE_PRICE")
    action_type: Mapped[str] = mapped_column(String(32))  # CHARGE, DISCHARGE, CHARGE_AND_DISCHARGE, IDLE

    requested_charge_kw: Mapped[float] = mapped_column(Float)
    actual_charge_kw: Mapped[float] = mapped_column(Float)
    requested_discharge_kw: Mapped[float] = mapped_column(Float)
    actual_discharge_kw: Mapped[float] = mapped_column(Float)

    process_demand_kw: Mapped[float] = mapped_column(Float)
    hx_power_limit_kw: Mapped[float] = mapped_column(Float)

    soc_start_fraction: Mapped[float] = mapped_column(Float)
    soc_end_fraction: Mapped[float] = mapped_column(Float)
    stored_energy_start_kwh: Mapped[float] = mapped_column(Float)
    stored_energy_end_kwh: Mapped[float] = mapped_column(Float)
    sand_temp_start_c: Mapped[float] = mapped_column(Float)
    sand_temp_end_c: Mapped[float] = mapped_column(Float)

    standing_loss_kwh: Mapped[float] = mapped_column(Float)
    grid_power_total_kw: Mapped[float] = mapped_column(Float)
    grid_power_limit_kw: Mapped[float] = mapped_column(Float, default=12.0)

    electricity_consumed_kwh: Mapped[float] = mapped_column(Float)
    interval_cost_eur: Mapped[float] = mapped_column(Float)
    energy_balance_residual_kwh: Mapped[float] = mapped_column(Float)
    reason_code: Mapped[str] = mapped_column(String(64))
    created_at_utc: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class ShadowEventAudit(Base):
    """Immutable audit trail of shadow session state transitions, resets, and triggers."""

    __tablename__ = "shadow_event_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shadow_session_id: Mapped[str] = mapped_column(String(36), index=True)
    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    event_type: Mapped[str] = mapped_column(String(32))  # START, PAUSE, RESUME, STOP, REOPTIMIZE, RESET, DEMAND_CHANGE
    previous_state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    new_state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    reason: Mapped[str] = mapped_column(String(255))
    details_json: Mapped[dict] = mapped_column(JSON, default=dict)


# --------------------------------------------------------------------------- Cloud 24/7 Deployment (Phase 5.9)
class WorkerHeartbeat(Base):
    """Heartbeat record published by the 24/7 background worker every 60 seconds."""

    __tablename__ = "worker_heartbeats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    worker_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    status: Mapped[str] = mapped_column(String(32), default="RUNNING")  # RUNNING | STANDBY | STOPPED
    app_env: Mapped[str] = mapped_column(String(32), default="production")
    market_status: Mapped[str] = mapped_column(String(32), default="LIVE")  # LIVE | STALE | ERROR
    shadow_status: Mapped[str] = mapped_column(String(32), default="RUNNING")  # RUNNING | PAUSED | STOPPED
    last_market_fetch_utc: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_optimization_utc: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_executed_interval_utc: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    next_interval_utc: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    details_json: Mapped[dict] = mapped_column(JSON, default=dict)


class DistributedLock(Base):
    """Distributed coordination lease table for single-active worker election fallback."""

    __tablename__ = "distributed_locks"

    lock_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(64))
    acquired_at_utc: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at_utc: Mapped[datetime] = mapped_column(UTCDateTime, index=True)


# --------------------------------------------------------------------------- Energy Ledger (Settlement & Savings)
class EnergyLedgerInterval(Base):
    """15-minute settlement energy ledger record for Virtual Dryer & Gen0 TES.

    Stores audit-grade financial and physical settlement metrics:
    - Baseline cost (if useful heat were provided by direct electric heating)
    - Actual operating cost (TES charge + backup heater + blower fan + aux)
    - Inventory valuation delta (SOC change valued at effective tariff)
    - Net inventory-adjusted savings
    - Moisture kinetics and provenance
    """

    __tablename__ = "energy_ledger_intervals"
    __table_args__ = (
        UniqueConstraint("site_id", "interval_start_utc", name="uq_ledger_interval"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[str] = mapped_column(String(64), default="gen0-vilnius-demo", index=True)
    interval_start_utc: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    interval_end_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    duration_minutes: Mapped[int] = mapped_column(Integer, default=15)

    # Market Prices (€/MWh)
    spot_price_eur_mwh: Mapped[float] = mapped_column(Float)
    effective_price_eur_mwh: Mapped[float] = mapped_column(Float)

    # Energy Quantities (kWh per 15-min interval)
    tes_charge_energy_kwh: Mapped[float] = mapped_column(Float, default=0.0)
    dryer_thermal_delivered_kwh: Mapped[float] = mapped_column(Float, default=0.0)
    backup_heater_energy_kwh: Mapped[float] = mapped_column(Float, default=0.0)
    blower_electric_kwh: Mapped[float] = mapped_column(Float, default=0.0)
    auxiliary_electric_kwh: Mapped[float] = mapped_column(Float, default=0.0)
    total_grid_import_kwh: Mapped[float] = mapped_column(Float, default=0.0)

    # Power Peaks (kW)
    peak_grid_power_kw: Mapped[float] = mapped_column(Float, default=0.0)
    avg_tes_heat_kw: Mapped[float] = mapped_column(Float, default=0.0)
    avg_dryer_demand_kw: Mapped[float] = mapped_column(Float, default=1.50)

    # Temperatures & Physical State
    avg_sand_temp_c: Mapped[float] = mapped_column(Float)
    avg_dryer_supply_temp_c: Mapped[float] = mapped_column(Float)
    avg_dryer_exhaust_temp_c: Mapped[float] = mapped_column(Float)
    opening_soc_fraction: Mapped[float] = mapped_column(Float)
    closing_soc_fraction: Mapped[float] = mapped_column(Float)

    # Financial Ledger (€ per interval)
    baseline_cost_eur: Mapped[float] = mapped_column(Float)      # Heat delivered * effective_price
    actual_cost_eur: Mapped[float] = mapped_column(Float)        # (Charge + Backup + Blower + Aux) * effective_price
    inventory_delta_eur: Mapped[float] = mapped_column(Float)    # (SOC_end - SOC_start) * capacity * effective_price
    net_savings_eur: Mapped[float] = mapped_column(Float)        # Baseline - Actual + InventoryDelta

    # Timber Batch Progress (Estimated)
    moisture_content_percent: Mapped[float] = mapped_column(Float, default=50.0)
    water_removed_kg: Mapped[float] = mapped_column(Float, default=0.0)

    # Provenance & Audit
    data_provenance: Mapped[str] = mapped_column(String(32), default="SIMULATED")
    operating_mode: Mapped[str] = mapped_column(String(32), default="VIRTUAL")
    created_at_utc: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


