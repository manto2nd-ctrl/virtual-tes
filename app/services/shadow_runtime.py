"""Live Virtual TES Shadow Runtime Service (Phase 5.8.2).

Continuously operates a persistent virtual thermal storage plant against the real
market clock and real Lithuanian electricity prices.

CRITICAL DIGITAL TWIN CONSTRAINTS:
- ZERO hardware/PLC/relay commands: Purely a software simulation twin.
- SQLite persistence: Sessions survive browser refresh, app restarts, and reboot.
- Idempotent interval execution: UNIQUE(shadow_session_id, interval_start_utc).
- Fractional dt handling: Starting mid-interval applies exact remaining duration.
- Simultaneous charge & discharge: Virtual heaters (up to 9 kW) and closed-air HX (1.5 kW)
  operate simultaneously in the same interval.
- Headroom enforcement: P_grid = P_charge + P_site (2 kW) + P_aux (0.05 kW) <= 12 kW.
- Energy balance verification: Evaluates energy residual per step, warning if violated.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.core.timegrid import ensure_utc
from app.database.models import (
    DayAheadPrice,
    ShadowEventAudit,
    ShadowIntervalRecord,
    ShadowTESSession,
    TESConfig,
)
from app.database.repositories import PriceRepository, ShadowRepository
from app.economics.tariff import effective_price_eur_mwh as calc_effective_price_eur_mwh
from app.models.domain import MarketPriceInterval, PricePoint
from app.optimization.domain import (
    OptimizationIntervalInput,
    OptimizationProblemInput,
    OptimizationResult,
)
from app.optimization.lp_optimizer import LPOptimizer
from app.services.market_data_service import MarketDataService
from app.tes.model import HelicalAirHXModel, ThermalStateMapper, compute_step, retention_factor

log = logging.getLogger(__name__)

VILNIUS_TZ = ZoneInfo("Europe/Vilnius")
EPS_ENERGY = 1e-4  # kWh tolerance for energy balance residual check


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_market_interval_bounds(dt: datetime | None = None) -> tuple[datetime, datetime]:
    """Return the 15-minute UTC interval [start_utc, end_utc) containing `dt`."""
    t = ensure_utc(dt) if dt is not None else utcnow()
    minute_bin = (t.minute // 15) * 15
    start_utc = t.replace(minute=minute_bin, second=0, microsecond=0)
    end_utc = start_utc + timedelta(minutes=15)
    return start_utc, end_utc


class ShadowRuntimeService:
    """Manages continuous execution of the Virtual TES Shadow Digital Twin."""

    def __init__(
        self,
        market_service: MarketDataService | None = None,
        tes_params: TESParameters | None = None,
        site_params: SiteParameters | None = None,
        tariff_params: TariffParameters | None = None,
        tz: ZoneInfo = VILNIUS_TZ,
    ) -> None:
        self.tz = tz
        self.market_service = market_service or MarketDataService(tz=tz)
        self.tes_params = tes_params or TESParameters(
            thermal_capacity_full_span_kwh=15.0,
            optimizer_soc_min_fraction=0.10,
            optimizer_soc_max_fraction=1.00,
            max_charge_power_kw=9.0,
            max_discharge_power_kw=3.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.90,
            standing_loss_percent_per_day=2.0,
            auxiliary_power_kw=0.05,
        )
        self.site_params = site_params or SiteParameters(
            grid_connection_limit_kw=12.0,
            process_heat_demand_kw=1.50,
            other_loads_kw=2.0,
        )
        self.tariff_params = tariff_params or TariffParameters(
            supplier_markup_eur_mwh=1.50,
            variable_grid_fee_eur_mwh=25.00,
            variable_tax_eur_mwh=5.00,
        )
        self.thermal_mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)
        self.hx_model = HelicalAirHXModel(
            hx_area_m2=1.55,
            overall_u_w_m2k=9.0,
            airflow_m3_h=80.0,
            air_inlet_temperature_c=40.0,
            thermal_state_mapper=self.thermal_mapper,
        )

    # --------------------------------------------------------------------------- State mapping
    def convert_initial_conditions(
        self,
        initial_soc_percent: float | None = None,
        initial_energy_kwh: float | None = None,
        initial_temp_c: float | None = None,
        initialization_mode: str | None = None,
    ) -> tuple[float, float, float]:
        """Convert single chosen canonical source of truth to (fraction, kWh, °C).

        Backend recalculates all derived values using ThermalStateMapper and TESParameters.
        Contradictory values from other non-canonical inputs are ignored.
        """
        full_cap = self.tes_params.thermal_capacity_full_span_kwh

        if initialization_mode is not None:
            mode = initialization_mode.lower().strip()
        else:
            if initial_temp_c is not None and initial_soc_percent is None and initial_energy_kwh is None:
                mode = "temp"
            elif initial_energy_kwh is not None and initial_soc_percent is None:
                mode = "energy"
            else:
                mode = "soc"

        if mode == "soc":
            val = 50.0 if initial_soc_percent is None else float(initial_soc_percent)
            frac = max(0.0, min(1.0, val / 100.0))
            kwh = frac * full_cap
            temp = self.thermal_mapper.temperature_from_soc_fraction(frac)
        elif mode == "energy":
            val = (0.5 * full_cap) if initial_energy_kwh is None else float(initial_energy_kwh)
            kwh = max(0.0, min(full_cap, val))
            frac = kwh / full_cap if full_cap > 0 else 0.0
            temp = self.thermal_mapper.temperature_from_soc_fraction(frac)
        elif mode == "temp":
            t_min = self.thermal_mapper.t_min_c
            t_max = self.thermal_mapper.t_max_c
            val = 196.77 if initial_temp_c is None else float(initial_temp_c)
            temp = max(t_min, min(t_max, val))
            frac = self.thermal_mapper.soc_fraction_from_temperature(temp)
            kwh = frac * full_cap
        else:
            # Fallback to SOC default (50%)
            frac = 0.50
            kwh = frac * full_cap
            temp = self.thermal_mapper.temperature_from_soc_fraction(frac)

        return round(frac, 4), round(kwh, 4), round(temp, 2)

    # --------------------------------------------------------------------------- Session lifecycle
    def get_or_create_session(
        self,
        db: Session,
        session_id: str | None = None,
    ) -> ShadowTESSession:
        repo = ShadowRepository(db)
        if session_id:
            s = repo.get_session(session_id)
            if s:
                return s
        latest = repo.get_latest_session()
        if latest:
            return latest

        # Create fresh session in STOPPED state
        init_frac, init_kwh, init_temp = self.convert_initial_conditions(initial_soc_percent=50.0, initialization_mode="soc")
        now = utcnow()
        new_sess = ShadowTESSession(
            id=str(uuid.uuid4()),
            status="STOPPED",
            created_at_utc=now,
            model_version="gen0-shadow-v1",
            market_source="LITGRID",
            initial_soc_fraction=init_frac,
            initial_stored_energy_kwh=init_kwh,
            initial_sand_temperature_c=init_temp,
            current_soc_fraction=init_frac,
            current_stored_energy_kwh=init_kwh,
            current_sand_temperature_c=init_temp,
            process_enabled=True,
            process_heat_demand_kw=self.site_params.process_heat_demand_kw,
            active_schedule_version=1,
            planned_schedule=[],
        )
        repo.create_session(new_sess)
        db.commit()
        return new_sess

    def start_session(
        self,
        db: Session,
        initial_soc_percent: float | None = None,
        initial_energy_kwh: float | None = None,
        initial_temp_c: float | None = None,
        initialization_mode: str | None = None,
        process_demand_kw: float | None = None,
        process_enabled: bool = True,
        session_id: str | None = None,
        now_utc: datetime | None = None,
    ) -> ShadowTESSession:
        """Start or reinitialize the live Virtual TES shadow runtime."""
        repo = ShadowRepository(db)
        now = ensure_utc(now_utc) if now_utc is not None else utcnow()

        frac, kwh, temp = self.convert_initial_conditions(
            initial_soc_percent=initial_soc_percent,
            initial_energy_kwh=initial_energy_kwh,
            initial_temp_c=initial_temp_c,
            initialization_mode=initialization_mode,
        )

        p_demand = process_demand_kw if process_demand_kw is not None else self.site_params.process_heat_demand_kw

        # Check existing active session
        session_obj = repo.get_session(session_id) if session_id else repo.get_latest_session()
        if session_obj is None:
            session_obj = ShadowTESSession(
                id=str(uuid.uuid4()),
                status="RUNNING",
                created_at_utc=now,
                started_at_utc=now,
                model_version="gen0-shadow-v1",
                market_source="LITGRID",
                initial_soc_fraction=frac,
                initial_stored_energy_kwh=kwh,
                initial_sand_temperature_c=temp,
                current_soc_fraction=frac,
                current_stored_energy_kwh=kwh,
                current_sand_temperature_c=temp,
                last_state_update_utc=now,
                process_enabled=process_enabled,
                process_heat_demand_kw=p_demand,
                active_schedule_version=1,
                planned_schedule=[],
            )
            repo.create_session(session_obj)
        else:
            # Update existing session
            session_obj.status = "RUNNING"
            session_obj.started_at_utc = now
            session_obj.stopped_at_utc = None
            session_obj.initial_soc_fraction = frac
            session_obj.initial_stored_energy_kwh = kwh
            session_obj.initial_sand_temperature_c = temp
            session_obj.current_soc_fraction = frac
            session_obj.current_stored_energy_kwh = kwh
            session_obj.current_sand_temperature_c = temp
            session_obj.last_state_update_utc = now
            session_obj.process_enabled = process_enabled
            session_obj.process_heat_demand_kw = p_demand
            session_obj.active_schedule_version += 1
            repo.update_session(session_obj)

        audit = ShadowEventAudit(
            shadow_session_id=session_obj.id,
            timestamp_utc=now,
            event_type="START",
            previous_state_json={"status": "STOPPED"},
            new_state_json={
                "status": "RUNNING",
                "soc_fraction": frac,
                "stored_energy_kwh": kwh,
                "sand_temp_c": temp,
            },
            reason="SHADOW_START",
            details_json={"process_demand_kw": p_demand, "process_enabled": process_enabled},
        )
        repo.record_audit(audit)
        db.commit()

        # Run immediate optimization
        self.reoptimize(db, session_obj, trigger_reason="SHADOW_START", now_utc=now)
        return session_obj

    def pause_session(self, db: Session, session_id: str | None = None, now_utc: datetime | None = None) -> ShadowTESSession:
        repo = ShadowRepository(db)
        sess = self.get_or_create_session(db, session_id)
        now = ensure_utc(now_utc) if now_utc is not None else utcnow()

        old_status = sess.status
        sess.status = "PAUSED"
        sess.last_state_update_utc = now
        repo.update_session(sess)

        audit = ShadowEventAudit(
            shadow_session_id=sess.id,
            timestamp_utc=now,
            event_type="PAUSE",
            previous_state_json={"status": old_status},
            new_state_json={"status": "PAUSED"},
            reason="USER_PAUSE",
        )
        repo.record_audit(audit)
        db.commit()
        return sess

    def resume_session(self, db: Session, session_id: str | None = None, now_utc: datetime | None = None) -> ShadowTESSession:
        repo = ShadowRepository(db)
        sess = self.get_or_create_session(db, session_id)
        now = ensure_utc(now_utc) if now_utc is not None else utcnow()

        old_status = sess.status
        sess.status = "RUNNING"
        sess.last_state_update_utc = now
        repo.update_session(sess)

        audit = ShadowEventAudit(
            shadow_session_id=sess.id,
            timestamp_utc=now,
            event_type="RESUME",
            previous_state_json={"status": old_status},
            new_state_json={"status": "RUNNING"},
            reason="USER_RESUME",
        )
        repo.record_audit(audit)
        db.commit()

        self.reoptimize(db, sess, trigger_reason="RESUME", now_utc=now)
        return sess

    def stop_session(self, db: Session, session_id: str | None = None, now_utc: datetime | None = None) -> ShadowTESSession:
        repo = ShadowRepository(db)
        sess = self.get_or_create_session(db, session_id)
        now = ensure_utc(now_utc) if now_utc is not None else utcnow()

        old_status = sess.status
        sess.status = "STOPPED"
        sess.stopped_at_utc = now
        sess.last_state_update_utc = now
        repo.update_session(sess)

        audit = ShadowEventAudit(
            shadow_session_id=sess.id,
            timestamp_utc=now,
            event_type="STOP",
            previous_state_json={"status": old_status},
            new_state_json={"status": "STOPPED"},
            reason="USER_STOP",
        )
        repo.record_audit(audit)
        db.commit()
        return sess

    def reset_state(
        self,
        db: Session,
        target_soc_percent: float,
        reason: str = "MANUAL_RESET",
        session_id: str | None = None,
        now_utc: datetime | None = None,
    ) -> ShadowTESSession:
        """Reset virtual physical state with mandatory audit trail (preserves interval history)."""
        repo = ShadowRepository(db)
        sess = self.get_or_create_session(db, session_id)
        now = ensure_utc(now_utc) if now_utc is not None else utcnow()

        prev_state = {
            "soc_fraction": sess.current_soc_fraction,
            "stored_energy_kwh": sess.current_stored_energy_kwh,
            "sand_temp_c": sess.current_sand_temperature_c,
            "status": sess.status,
        }

        frac, kwh, temp = self.convert_initial_conditions(initial_soc_percent=target_soc_percent)
        sess.current_soc_fraction = frac
        sess.current_stored_energy_kwh = kwh
        sess.current_sand_temperature_c = temp
        sess.last_state_update_utc = now
        sess.active_schedule_version += 1
        repo.update_session(sess)

        new_state = {
            "soc_fraction": frac,
            "stored_energy_kwh": kwh,
            "sand_temp_c": temp,
            "status": sess.status,
        }

        audit = ShadowEventAudit(
            shadow_session_id=sess.id,
            timestamp_utc=now,
            event_type="RESET",
            previous_state_json=prev_state,
            new_state_json=new_state,
            reason=reason,
            details_json={"target_soc_percent": target_soc_percent},
        )
        repo.record_audit(audit)
        db.commit()

        if sess.status == "RUNNING":
            self.reoptimize(db, sess, trigger_reason="STATE_DEVIATION", now_utc=now)
        return sess

    def update_process_demand(
        self,
        db: Session,
        demand_kw: float,
        enabled: bool = True,
        session_id: str | None = None,
        now_utc: datetime | None = None,
    ) -> ShadowTESSession:
        """Update virtual process heat demand without modifying market price dependencies."""
        repo = ShadowRepository(db)
        sess = self.get_or_create_session(db, session_id)
        now = ensure_utc(now_utc) if now_utc is not None else utcnow()

        old_demand = sess.process_heat_demand_kw
        old_enabled = sess.process_enabled

        sess.process_heat_demand_kw = float(demand_kw)
        sess.process_enabled = bool(enabled)
        repo.update_session(sess)

        audit = ShadowEventAudit(
            shadow_session_id=sess.id,
            timestamp_utc=now,
            event_type="DEMAND_CHANGE",
            previous_state_json={"process_demand_kw": old_demand, "process_enabled": old_enabled},
            new_state_json={"process_demand_kw": demand_kw, "process_enabled": enabled},
            reason="PROCESS_DEMAND_CHANGE",
        )
        repo.record_audit(audit)
        db.commit()

        if sess.status == "RUNNING":
            self.reoptimize(db, sess, trigger_reason="PROCESS_DEMAND_CHANGE", now_utc=now)
        return sess

    # --------------------------------------------------------------------------- Live Optimization
    def get_known_market_intervals(
        self,
        db: Session,
        start_from_utc: datetime,
        limit_hours: int = 48,
    ) -> list[MarketPriceInterval]:
        """Fetch all currently known and published market intervals starting from `start_from_utc`."""
        start_utc = ensure_utc(start_from_utc)
        end_utc = start_utc + timedelta(hours=limit_hours)

        try:
            fetch_res = self.market_service.fetch_market_data(
                bidding_zone="LT",
                start_utc=start_utc,
                end_utc=end_utc,
                target_resolution_minutes=15,
                session=db,
            )
            return [iv for iv in fetch_res.intervals if iv.end_utc > start_utc]
        except Exception as exc:
            log.warning("Market service fetch encountered exception: %s. Falling back to DB cache.", exc)
            repo = PriceRepository(db, self.tz)
            db_prices = repo.get_prices("LT", start_utc, end_utc, latest_only=True)
            intervals = [
                MarketPriceInterval(
                    start_utc=p.delivery_start_utc,
                    end_utc=p.delivery_end_utc,
                    price_eur_mwh=p.price_eur_mwh,
                    bidding_zone=p.bidding_zone,
                    source=p.source,
                    original_resolution_minutes=p.resolution_minutes,
                    fetched_at_utc=p.fetched_at,
                    source_record_id=f"db-{p.id}",
                )
                for p in db_prices
            ]
            return [iv for iv in intervals if iv.end_utc > start_utc]

    def reoptimize(
        self,
        db: Session,
        shadow_session: ShadowTESSession,
        trigger_reason: str = "MANUAL_REOPTIMIZE",
        now_utc: datetime | None = None,
    ) -> ShadowTESSession:
        """Run receding-horizon LP optimization from current time across all known market prices."""
        now = ensure_utc(now_utc) if now_utc is not None else utcnow()
        curr_start_utc, curr_end_utc = get_market_interval_bounds(now)

        known_intervals = self.get_known_market_intervals(db, curr_start_utc)
        if not known_intervals:
            log.warning("No future market intervals found for re-optimization.")
            return shadow_session

        # Determine if mid-interval start
        dt_remaining_sec = max(30.0, (curr_end_utc - now).total_seconds())
        dt_first_h = min(0.25, dt_remaining_sec / 3600.0)

        # Build optimization problem input
        opt_inputs: list[OptimizationIntervalInput] = []
        p_demand = shadow_session.process_heat_demand_kw if shadow_session.process_enabled else 0.0

        for idx, iv in enumerate(known_intervals):
            eff = calc_effective_price_eur_mwh(iv.price_eur_mwh, self.tariff_params)
            # If interval 0 is currently in progress, adjust start and duration
            if iv.start_utc == curr_start_utc:
                # Fractional interval from now to curr_end_utc
                start_point = now if dt_first_h < 0.24 else curr_start_utc
                end_point = curr_end_utc
            else:
                start_point = iv.start_utc
                end_point = iv.end_utc

            opt_inputs.append(
                OptimizationIntervalInput(
                    start_utc=start_point,
                    end_utc=end_point,
                    spot_price_eur_mwh=iv.price_eur_mwh,
                    effective_price_eur_mwh=eff,
                    heat_demand_kw=p_demand,
                    other_site_load_kw=self.site_params.other_loads_kw,
                    auxiliary_load_kw=self.tes_params.auxiliary_power_kw,
                )
            )

        # Headroom and capacity bounds
        initial_kwh = shadow_session.current_stored_energy_kwh

        problem = OptimizationProblemInput(
            intervals=opt_inputs,
            tes_params=self.tes_params,
            grid_connection_limit_kw=self.site_params.grid_connection_limit_kw,
            initial_soc_kwh=initial_kwh,
            target_terminal_soc_kwh=self.tes_params.soc_min_kwh,
            terminal_soc_condition="min",
            discharge_limit_curve=self.hx_model,
            lexicographic=True,
        )

        try:
            optimizer = LPOptimizer()
            res = optimizer.optimize(problem)

            schedule_list = []
            for item in res.intervals:
                schedule_list.append({
                    "start_utc": item.start_utc.isoformat(),
                    "end_utc": item.end_utc.isoformat(),
                    "start_vilnius": item.start_utc.astimezone(self.tz).strftime("%H:%M"),
                    "end_vilnius": item.end_utc.astimezone(self.tz).strftime("%H:%M"),
                    "spot_price_eur_mwh": round(item.spot_price_eur_mwh, 2),
                    "effective_price_eur_mwh": round(item.effective_price_eur_mwh, 2),
                    "charge_kw": round(item.charge_power_kw, 2),
                    "discharge_kw": round(item.discharge_power_kw, 2),
                    "predicted_soc_kwh": round(item.soc_kwh, 2),
                    "predicted_soc_percent": round(item.soc_percent, 1),
                })

            shadow_session.planned_schedule = schedule_list
            shadow_session.active_schedule_version += 1

            # Determine immediate action & reason code
            if res.intervals:
                first = res.intervals[0]
                chg = first.charge_power_kw
                dis = first.discharge_power_kw

                if chg > 0.05 and dis > 0.05:
                    action_type = "CHARGE + SUPPLY PROCESS"
                    reason = "CHEAP_RELATIVE_TO_KNOWN_FUTURE_PRICES"
                elif chg > 0.05:
                    action_type = "CHARGE"
                    reason = (
                        "NEGATIVE_OR_ZERO_SPOT_PRICE"
                        if first.spot_price_eur_mwh <= 0.0
                        else "CHEAP_RELATIVE_TO_KNOWN_FUTURE_PRICES"
                    )
                elif dis > 0.05:
                    action_type = "SUPPLY PROCESS"
                    reason = "PROCESS_HEAT_DELIVERY"
                else:
                    action_type = "IDLE"
                    reason = "EXPENSIVE_PRESERVE_CHARGE"

                shadow_session.latest_action_type = action_type
                shadow_session.latest_reason_code = reason
                shadow_session.optimization_price_eur_mwh = round(first.effective_price_eur_mwh, 2)
                shadow_session.optimization_price_basis = "EFFECTIVE_VARIABLE_PRICE"

            repo = ShadowRepository(db)
            repo.update_session(shadow_session)
            db.commit()
            log.info("Shadow re-optimization completed (reason: %s, intervals: %d)", trigger_reason, len(res.intervals))
        except Exception as opt_err:
            log.error("Shadow re-optimization failed: %s", opt_err)
            shadow_session.error_message = f"Optimization failed: {opt_err}"
            db.commit()

        return shadow_session

    # --------------------------------------------------------------------------- Physical interval execution
    def execute_interval_step(
        self,
        db: Session,
        shadow_session: ShadowTESSession,
        interval_start_utc: datetime,
        interval_end_utc: datetime,
        spot_price_eur_mwh: float,
        effective_price_eur_mwh: float | None = None,
        requested_charge_kw: float | None = None,
        requested_discharge_kw: float | None = None,
        reason_code: str | None = None,
        now_utc: datetime | None = None,
    ) -> ShadowIntervalRecord:
        """Execute exactly one physical interval step idempotently.
        
        Enforces grid connection limit, heater maximum, HX power derating,
        efficiency losses, standing losses, and energy balance verification.
        """
        repo = ShadowRepository(db)
        start_utc = ensure_utc(interval_start_utc)
        end_utc = ensure_utc(interval_end_utc)

        # Idempotency check: never execute twice
        existing = db.execute(
            select(ShadowIntervalRecord).where(
                ShadowIntervalRecord.shadow_session_id == shadow_session.id,
                ShadowIntervalRecord.interval_start_utc == start_utc,
            )
        ).scalar_one_or_none()
        if existing is not None:
            log.info("Interval %s already executed (idempotent no-op)", start_utc.isoformat())
            return existing

        dt_h = max(0.001, (end_utc - start_utc).total_seconds() / 3600.0)

        # Process demand
        p_demand = shadow_session.process_heat_demand_kw if shadow_session.process_enabled else 0.0

        # Current state before interval
        soc_start_kwh = shadow_session.current_stored_energy_kwh
        soc_start_fraction = shadow_session.current_soc_fraction
        sand_temp_start_c = shadow_session.current_sand_temperature_c

        # HX power limit at current sand temperature
        hx_limit_kw = self.hx_model.p_max_at_temperature_kw(sand_temp_start_c)

        # Requested discharge power (bounded by process demand and HX limit)
        if requested_discharge_kw is not None:
            req_dis = min(float(requested_discharge_kw), hx_limit_kw)
        else:
            req_dis = min(p_demand, hx_limit_kw)

        # Requested charge power from active optimizer schedule
        if requested_charge_kw is not None:
            req_chg = float(requested_charge_kw)
        else:
            req_chg = 0.0
            if shadow_session.planned_schedule:
                # Find matching or nearest planned interval
                for item in shadow_session.planned_schedule:
                    p_start = datetime.fromisoformat(item["start_utc"])
                    if abs((p_start - start_utc).total_seconds()) < 600:
                        req_chg = item.get("charge_kw", 0.0)
                        break

        # Grid headroom constraint
        p_site = self.site_params.other_loads_kw
        p_aux = self.tes_params.auxiliary_power_kw
        p_grid_limit = self.site_params.grid_connection_limit_kw
        headroom = max(0.0, p_grid_limit - p_site - p_aux)

        # Prices & costs
        eff_price = (
            effective_price_eur_mwh
            if effective_price_eur_mwh is not None
            else calc_effective_price_eur_mwh(spot_price_eur_mwh, self.tariff_params)
        )

        is_running = (shadow_session.status == "RUNNING")
        if not is_running:
            # Physical state does not evolve when paused or stopped (Requirement 4 & 9)
            chg_act = 0.0
            dis_act = 0.0
            elec_consumed_kwh = 0.0
            interval_cost = 0.0
            grid_total_kw = p_site + p_aux
            soc_end_kwh = soc_start_kwh
            soc_end_fraction = soc_start_fraction
            sand_temp_end_c = sand_temp_start_c
            standing_loss_kwh = 0.0
            residual = 0.0
            action_type = shadow_session.status
            r_code = reason_code or f"{shadow_session.status}_NO_ACTUATION"
        else:
            # Physical step evaluation (using standard compute_step)
            step_res = compute_step(
                params=self.tes_params,
                soc_kwh=soc_start_kwh,
                dt_h=dt_h,
                requested_charge_kw=req_chg,
                requested_discharge_kw=req_dis,
                external_charge_limit_kw=headroom,
                interval_start_utc=start_utc,
                discharge_limit_curve=self.hx_model,
            )

            # Energy balance verification
            expected_delta = step_res.energy_stored_kwh - step_res.energy_withdrawn_kwh - step_res.storage_loss_kwh
            actual_delta = step_res.soc_end_kwh - soc_start_kwh
            residual = abs(actual_delta - expected_delta)
            if residual > EPS_ENERGY:
                log.error("SHADOW ENERGY BALANCE ERROR: residual=%.6f kWh > tolerance=%.6f", residual, EPS_ENERGY)
                shadow_session.error_message = f"SHADOW ENERGY BALANCE ERROR (res={residual:.6f} kWh)"

            elec_consumed_kwh = step_res.charge_power_kw * dt_h
            interval_cost = elec_consumed_kwh * (eff_price / 1000.0)
            grid_total_kw = step_res.charge_power_kw + p_site + p_aux

            soc_end_kwh = step_res.soc_end_kwh
            soc_end_fraction = soc_end_kwh / self.tes_params.thermal_capacity_full_span_kwh
            sand_temp_end_c = self.thermal_mapper.temperature_from_soc_fraction(soc_end_fraction)
            standing_loss_kwh = step_res.storage_loss_kwh

            chg_act = step_res.charge_power_kw
            dis_act = step_res.discharge_power_kw

            if chg_act > 0.05 and dis_act > 0.05:
                action_type = "CHARGE + SUPPLY PROCESS"
            elif chg_act > 0.05:
                action_type = "CHARGE"
            elif dis_act > 0.05:
                action_type = "SUPPLY PROCESS"
            else:
                action_type = "IDLE"

            r_code = reason_code or shadow_session.latest_reason_code or "OPTIMIZED_DISPATCH"

        record = ShadowIntervalRecord(
            shadow_session_id=shadow_session.id,
            interval_start_utc=start_utc,
            interval_end_utc=end_utc,
            duration_hours=dt_h,
            spot_price_eur_mwh=round(spot_price_eur_mwh, 2),
            effective_price_eur_mwh=round(eff_price, 2),
            optimization_price_eur_mwh=round(eff_price, 2),
            optimization_price_basis="EFFECTIVE_VARIABLE_PRICE",
            action_type=action_type,
            requested_charge_kw=round(req_chg, 2),
            actual_charge_kw=round(chg_act, 2),
            requested_discharge_kw=round(req_dis, 2),
            actual_discharge_kw=round(dis_act, 2),
            process_demand_kw=round(p_demand, 2),
            hx_power_limit_kw=round(hx_limit_kw, 2),
            soc_start_fraction=round(soc_start_fraction, 4),
            soc_end_fraction=round(soc_end_fraction, 4),
            stored_energy_start_kwh=round(soc_start_kwh, 3),
            stored_energy_end_kwh=round(soc_end_kwh, 3),
            sand_temp_start_c=round(sand_temp_start_c, 2),
            sand_temp_end_c=round(sand_temp_end_c, 2),
            standing_loss_kwh=round(standing_loss_kwh, 4),
            grid_power_total_kw=round(grid_total_kw, 2),
            grid_power_limit_kw=round(p_grid_limit, 2),
            electricity_consumed_kwh=round(elec_consumed_kwh, 3),
            interval_cost_eur=round(interval_cost, 4),
            energy_balance_residual_kwh=round(residual, 7),
            reason_code=r_code,
            created_at_utc=utcnow(),
        )
        repo.record_interval(record)

        # Update session active state
        shadow_session.current_stored_energy_kwh = soc_end_kwh
        shadow_session.current_soc_fraction = soc_end_fraction
        shadow_session.current_sand_temperature_c = round(sand_temp_end_c, 2)
        shadow_session.last_executed_interval_start_utc = start_utc
        shadow_session.last_state_update_utc = end_utc
        shadow_session.latest_action_type = action_type
        shadow_session.optimization_price_eur_mwh = round(eff_price, 2)
        shadow_session.optimization_price_basis = "EFFECTIVE_VARIABLE_PRICE"
        repo.update_session(shadow_session)
        db.commit()

        log.info(
            "Executed shadow interval %s: chg=%.2f kW, dis=%.2f kW, soc_end=%.1f%%, res=%.6f",
            start_utc.strftime("%H:%M"),
            chg_act,
            dis_act,
            soc_end_fraction * 100.0,
            residual,
        )
        return record

    # --------------------------------------------------------------------------- Continuous Background Tick & Catch-up
    def catch_up_missed_intervals(
        self,
        db: Session,
        shadow_session: ShadowTESSession,
        now_utc: datetime | None = None,
    ) -> int:
        """Sequentially replay intervals missed during application shutdown or pauses."""
        now = ensure_utc(now_utc) if now_utc is not None else utcnow()
        curr_start_utc, _ = get_market_interval_bounds(now)

        last_exec = shadow_session.last_executed_interval_start_utc
        if last_exec is None:
            if shadow_session.started_at_utc is not None:
                # If never executed an interval yet, start from the interval containing started_at_utc
                start_bound, _ = get_market_interval_bounds(shadow_session.started_at_utc)
                last_exec = start_bound - timedelta(minutes=15)
            else:
                return 0

        last_exec = ensure_utc(last_exec)
        # Next expected interval start is last_exec + 15 min
        next_iv_start = last_exec + timedelta(minutes=15)
        count = 0


        while next_iv_start < curr_start_utc:
            next_iv_end = next_iv_start + timedelta(minutes=15)
            # Find price for this interval
            repo_price = PriceRepository(db, self.tz)
            prices = repo_price.get_prices("LT", next_iv_start, next_iv_end, latest_only=True)
            if not prices:
                log.warning("SHADOW DATA GAP: Missing price for %s during catch-up", next_iv_start.isoformat())
                shadow_session.error_message = "SHADOW DATA GAP: Missing historical market prices during catch-up"
                db.commit()
                break

            spot_price = prices[0].price_eur_mwh
            self.execute_interval_step(
                db=db,
                shadow_session=shadow_session,
                interval_start_utc=next_iv_start,
                interval_end_utc=next_iv_end,
                spot_price_eur_mwh=spot_price,
                reason_code="REPLAY_CATCHUP",
            )
            count += 1
            next_iv_start += timedelta(minutes=15)

        if count > 0:
            log.info("Caught up %d missed shadow intervals", count)
            # Re-optimize with new current state
            self.reoptimize(db, shadow_session, trigger_reason="STATE_DEVIATION", now_utc=now)

        return count

    def step_continuous_tick(
        self,
        db: Session,
        now_utc: datetime | None = None,
    ) -> None:
        """Periodic background tick (called every 2-5 seconds).
        
        1. Checks for interval boundary crossing and executes completed interval step.
        2. Smoothly progresses continuous intra-interval physical state for live UI view.
        """
        repo = ShadowRepository(db)
        session_obj = repo.get_latest_session()
        if session_obj is None or session_obj.status != "RUNNING":
            return

        now = ensure_utc(now_utc) if now_utc is not None else utcnow()
        curr_start_utc, curr_end_utc = get_market_interval_bounds(now)

        # Check if previous interval finished and needs formal recording
        last_exec = session_obj.last_executed_interval_start_utc
        if last_exec is None:
            # First tick of session: record boundary as curr_start_utc
            session_obj.last_executed_interval_start_utc = curr_start_utc
            repo.update_session(session_obj)
            db.commit()
        elif last_exec < curr_start_utc:
            # Boundary crossed! Catch-up or execute the completed interval
            prev_end = last_exec + timedelta(minutes=15)
            # Look up price for completed interval
            repo_price = PriceRepository(db, self.tz)
            prices = repo_price.get_prices("LT", last_exec, prev_end, latest_only=True)
            spot_price = prices[0].price_eur_mwh if prices else 50.0  # default if not fetched

            self.execute_interval_step(
                db=db,
                shadow_session=session_obj,
                interval_start_utc=last_exec,
                interval_end_utc=curr_start_utc,
                spot_price_eur_mwh=spot_price,
            )
            # Re-optimize after step
            self.reoptimize(db, session_obj, trigger_reason="INTERVAL_STEP_COMPLETE", now_utc=now)

        # Intra-interval continuous state evolution for display
        # Calculate active powers in this interval
        p_chg = 0.0
        p_dis = 0.0
        if session_obj.planned_schedule:
            for item in session_obj.planned_schedule:
                p_start = datetime.fromisoformat(item["start_utc"])
                if abs((p_start - curr_start_utc).total_seconds()) < 600:
                    p_chg = item.get("charge_kw", 0.0)
                    p_dis = item.get("discharge_kw", 0.0)
                    break

        # Compute elapsed time since last update (dt_tick)
        last_update = session_obj.last_state_update_utc or now
        last_update = ensure_utc(last_update)
        elapsed_sec = (now - last_update).total_seconds()

        if 0.5 <= elapsed_sec <= 60.0:
            dt_tick_h = elapsed_sec / 3600.0
            # Instantaneous power derivative
            eta_c = self.tes_params.charge_efficiency
            eta_d = self.tes_params.discharge_efficiency
            standing_loss_kwh = (self.tes_params.standing_loss_percent_per_day / 100.0 / 24.0) * session_obj.current_stored_energy_kwh * dt_tick_h

            dE = (eta_c * p_chg - (p_dis / eta_d) if p_dis > 0 else eta_c * p_chg) * dt_tick_h - standing_loss_kwh
            new_kwh = max(
                self.tes_params.soc_min_kwh,
                min(self.tes_params.thermal_capacity_full_span_kwh, session_obj.current_stored_energy_kwh + dE),
            )
            new_frac = new_kwh / self.tes_params.thermal_capacity_full_span_kwh
            new_temp = self.thermal_mapper.temperature_from_soc_fraction(new_frac)

            session_obj.current_stored_energy_kwh = round(new_kwh, 4)
            session_obj.current_soc_fraction = round(new_frac, 4)
            session_obj.current_sand_temperature_c = round(new_temp, 2)
            session_obj.last_state_update_utc = now
            repo.update_session(session_obj)
            db.commit()

    # --------------------------------------------------------------------------- Dashboard View Model
    def get_live_dashboard_state(
        self,
        db: Session,
        now_utc: datetime | None = None,
    ) -> dict[str, Any]:
        """Aggregate all real-time metrics, planned actions, and interval history for UI display."""
        now = ensure_utc(now_utc) if now_utc is not None else utcnow()
        now_local = now.astimezone(self.tz)
        curr_start_utc, curr_end_utc = get_market_interval_bounds(now)

        repo = ShadowRepository(db)
        sess = self.get_or_create_session(db)

        # Real market prices
        repo_price = PriceRepository(db, self.tz)
        prices = repo_price.get_prices("LT", curr_start_utc, curr_end_utc, latest_only=True)
        if prices:
            spot_price = prices[0].price_eur_mwh
            active_source = prices[0].source
        else:
            # Try fetching from market data service
            try:
                fetch_res = self.market_service.fetch_market_data("LT", curr_start_utc, curr_end_utc, target_resolution_minutes=15, session=db)
                spot_price = fetch_res.intervals[0].price_eur_mwh if fetch_res.intervals else 45.0
                active_source = fetch_res.active_source
            except Exception:
                spot_price = 45.0
                active_source = "CACHE"

        eff_price = calc_effective_price_eur_mwh(spot_price, self.tariff_params)

        # Boundary countdown
        rem_sec = max(0, int((curr_end_utc - now).total_seconds()))
        countdown_str = f"{rem_sec // 60:02d}:{rem_sec % 60:02d}"
        next_boundary_str = curr_end_utc.astimezone(self.tz).strftime("%H:%M")

        # Current thermal & electrical parameters
        full_cap = self.tes_params.thermal_capacity_full_span_kwh
        min_reserve_kwh = self.tes_params.soc_min_kwh
        disp_remaining = max(0.0, sess.current_stored_energy_kwh - min_reserve_kwh)
        soc_pct = sess.current_soc_fraction * 100.0

        hx_limit = self.hx_model.p_max_at_temperature_kw(sess.current_sand_temperature_c)
        p_demand = sess.process_heat_demand_kw if sess.process_enabled else 0.0

        # Requested / actual powers
        req_chg = 0.0
        req_dis = min(p_demand, hx_limit)
        if sess.planned_schedule:
            for item in sess.planned_schedule:
                p_start = ensure_utc(datetime.fromisoformat(item["start_utc"]))
                if curr_start_utc <= p_start < curr_end_utc or abs((p_start - curr_start_utc).total_seconds()) < 900:
                    req_chg = item.get("charge_kw", 0.0)
                    req_dis = item.get("discharge_kw", req_dis)
                    break

        p_site = self.site_params.other_loads_kw
        p_aux = self.tes_params.auxiliary_power_kw
        p_grid_limit = self.site_params.grid_connection_limit_kw
        available_headroom = max(0.0, p_grid_limit - p_site - p_aux)  # 12.0 - 2.0 - 0.05 = 9.95 kW
        margin_at_full_charge = max(0.0, p_grid_limit - (self.tes_params.max_charge_power_kw + p_site + p_aux))  # 12.0 - 11.05 = 0.95 kW

        is_running = (sess.status == "RUNNING")
        if is_running:
            actual_chg = min(req_chg, self.tes_params.max_charge_power_kw, available_headroom)
            actual_dis = min(req_dis, hx_limit)
            execution_state = "RUNNING"
            state_evolution = "RUNNING"
            if actual_chg > 0.05 and actual_dis > 0.05:
                planned_desc = f"CHARGE ({actual_chg:.2f} kW) + SUPPLY PROCESS ({actual_dis:.2f} kW)"
            elif actual_chg > 0.05:
                planned_desc = f"CHARGE ({actual_chg:.2f} kW)"
            elif actual_dis > 0.05:
                planned_desc = f"SUPPLY PROCESS ({actual_dis:.2f} kW)"
            else:
                planned_desc = "IDLE"
            action_now = f"EXECUTING: {planned_desc}"
            planned_action = planned_desc
        elif sess.status == "PAUSED":
            actual_chg = 0.0
            actual_dis = 0.0
            execution_state = "PAUSED"
            state_evolution = "PAUSED"
            if req_chg > 0.05 and req_dis > 0.05:
                planned_desc = f"CHARGE ({req_chg:.2f} kW) + SUPPLY PROCESS ({req_dis:.2f} kW)"
            elif req_chg > 0.05:
                planned_desc = f"CHARGE ({req_chg:.2f} kW)"
            elif req_dis > 0.05:
                planned_desc = f"SUPPLY PROCESS ({req_dis:.2f} kW)"
            else:
                planned_desc = "IDLE"
            action_now = f"PAUSED (Planned: {planned_desc})"
            planned_action = planned_desc
        else:  # STOPPED
            actual_chg = 0.0
            actual_dis = 0.0
            execution_state = "STOPPED"
            state_evolution = "STOPPED"
            planned_action = "STOPPED"
            action_now = "STOPPED"

        total_grid = actual_chg + p_site + p_aux
        current_unused_margin = max(0.0, p_grid_limit - total_grid)

        # Actuation & Staging Model (Requirement 6)
        actuation_model = "CONTINUOUS IDEALIZED"
        hardware_staging = "NOT YET IMPLEMENTED"
        staging_warning = "Continuous optimizer request — physical heater staging not yet modeled."

        # Intervals history
        interval_records = repo.get_intervals(sess.id, limit=100, ascending=False)
        history_rows = []
        for r in interval_records:
            r_start_loc = r.interval_start_utc.astimezone(self.tz).strftime("%H:%M")
            r_end_loc = r.interval_end_utc.astimezone(self.tz).strftime("%H:%M")
            opt_p = r.optimization_price_eur_mwh if r.optimization_price_eur_mwh is not None else r.effective_price_eur_mwh
            history_rows.append({
                "time": f"{r_start_loc} -> {r_end_loc}",
                "spot_price": f"{r.spot_price_eur_mwh:.2f}",
                "effective_price": f"{r.effective_price_eur_mwh:.2f}",
                "optimization_price": f"{opt_p:.2f}",
                "price_basis": r.optimization_price_basis or "EFFECTIVE_VARIABLE_PRICE",
                "action": r.action_type,
                "charge_requested": f"{r.requested_charge_kw:.2f}",
                "charge_actual": f"{r.actual_charge_kw:.2f}",
                "discharge_actual": f"{r.actual_discharge_kw:.2f}",
                "process_demand": f"{r.process_demand_kw:.2f}",
                "hx_limit": f"{r.hx_power_limit_kw:.2f}",
                "soc_start": f"{r.soc_start_fraction * 100.0:.1f}%",
                "soc_end": f"{r.soc_end_fraction * 100.0:.1f}%",
                "temp_start": f"{r.sand_temp_start_c:.1f}°C",
                "temp_end": f"{r.sand_temp_end_c:.1f}°C",
                "grid_power": f"{r.grid_power_total_kw:.2f}",
                "electricity_kwh": f"{r.electricity_consumed_kwh:.3f}",
                "cost_eur": f"€{r.interval_cost_eur:.4f}",
                "residual_kwh": f"{r.energy_balance_residual_kwh:.6f}",
                "reason": r.reason_code,
            })

        # Planned next actions (from schedule)
        planned_actions = []
        if sess.planned_schedule:
            for item in sess.planned_schedule[:12]:
                planned_actions.append({
                    "interval": f"{item.get('start_vilnius', '')}–{item.get('end_vilnius', '')}",
                    "charge_kw": item.get("charge_kw", 0.0),
                    "discharge_kw": item.get("discharge_kw", 0.0),
                    "predicted_soc_pct": item.get("predicted_soc_percent", 0.0),
                    "spot_price": item.get("spot_price_eur_mwh", 0.0),
                })

        # Check for energy balance warning
        has_energy_balance_error = any(
            r.energy_balance_residual_kwh > EPS_ENERGY for r in interval_records[:5]
        ) or (sess.error_message and "ENERGY BALANCE ERROR" in sess.error_message)

        return {
            "session_id": sess.id,
            "status": sess.status,
            "execution_state": execution_state,
            "state_evolution": state_evolution,
            "planned_action": planned_action,
            "local_time_vilnius": now_local.strftime("%H:%M:%S %Z"),
            "current_interval_vilnius": f"{curr_start_utc.astimezone(self.tz).strftime('%H:%M')} -> {curr_end_utc.astimezone(self.tz).strftime('%H:%M')}",
            "next_market_boundary": next_boundary_str,
            "countdown": countdown_str,
            "last_state_update": (sess.last_state_update_utc.astimezone(self.tz).strftime("%H:%M:%S") if sess.last_state_update_utc else "None"),
            "spot_price_eur_mwh": round(spot_price, 2),
            "effective_price_eur_mwh": round(eff_price, 2),
            "optimization_price_eur_mwh": round(eff_price, 2),
            "optimization_price_basis": sess.optimization_price_basis or "EFFECTIVE_VARIABLE_PRICE",
            "market_source": active_source,
            "soc_percent": round(soc_pct, 1),
            "stored_energy_kwh": round(sess.current_stored_energy_kwh, 2),
            "full_span_capacity_kwh": round(full_cap, 2),
            "dispatchable_remaining_kwh": round(disp_remaining, 2),
            "sand_temperature_c": round(sess.current_sand_temperature_c, 1),
            "action_now": action_now,
            "reason_code": sess.latest_reason_code or "OPTIMIZED_DISPATCH",
            "heater_requested_kw": round(req_chg, 2),
            "heater_actual_kw": round(actual_chg, 2),
            "tes_discharge_kw": round(actual_dis, 2),
            "process_heat_demand_kw": round(p_demand, 2),
            "process_enabled": sess.process_enabled,
            "hx_max_available_kw": round(hx_limit, 2),
            "grid_total_kw": round(total_grid, 2),
            "grid_limit_kw": round(p_grid_limit, 2),
            "other_loads_kw": round(p_site, 2),
            "aux_load_kw": round(p_aux, 3),
            "available_tes_charging_headroom_kw": round(available_headroom, 2),
            "current_unused_grid_margin_kw": round(current_unused_margin, 2),
            "grid_margin_at_full_charge_kw": round(margin_at_full_charge, 2),
            "actuation_model": actuation_model,
            "hardware_staging": hardware_staging,
            "staging_warning": staging_warning,
            "has_energy_balance_error": bool(has_energy_balance_error),
            "error_message": sess.error_message,
            "planned_actions": planned_actions,
            "history_rows": history_rows,
        }
