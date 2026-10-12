"""15-Minute Settlement Energy Ledger Service (Phase 5.10).

Calculates audit-grade industrial accounting metrics:
- Direct Electric Baseline Cost (cost if heat were supplied by raw electric elements)
- Actual Operating Cost (TES charging + backup duct heater + blower fan + auxiliaries)
- Inventory Valuation Delta (SOC change priced at interval's effective tariff)
- Net Inventory-Adjusted Savings
- Period aggregation (Today, 7d, 30d, All) and CSV audit export.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session

from app.core.timegrid import ensure_utc
from app.database.models import EnergyLedgerInterval
from app.models.hardware_interfaces import DataProvenance, OperatingMode

VILNIUS_TZ = ZoneInfo("Europe/Vilnius")


def utcnow() -> datetime:
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)



class EnergyLedgerService:
    """Service for recording, querying, and exporting 15-minute settlement ledger entries."""

    def __init__(self, tz: ZoneInfo = VILNIUS_TZ, full_capacity_kwh: float = 15.0) -> None:
        self.tz = tz
        self.full_capacity_kwh = full_capacity_kwh

    def calculate_interval_finances(
        self,
        heat_delivered_kwh: float,
        tes_charge_kwh: float,
        backup_heat_kwh: float,
        blower_kwh: float,
        aux_kwh: float,
        opening_soc_fraction: float,
        closing_soc_fraction: float,
        effective_price_eur_mwh: float,
    ) -> dict[str, float]:
        """Compute exact settlement economics for one interval."""
        # Convert €/MWh to €/kWh (/ 1000)
        tariff_eur_kwh = effective_price_eur_mwh / 1000.0

        baseline_cost_eur = heat_delivered_kwh * tariff_eur_kwh
        total_grid_kwh = tes_charge_kwh + backup_heat_kwh + blower_kwh + aux_kwh
        actual_cost_eur = total_grid_kwh * tariff_eur_kwh

        # Inventory delta: change in stored energy valued at the current tariff
        delta_soc = closing_soc_fraction - opening_soc_fraction
        inventory_delta_kwh = delta_soc * self.full_capacity_kwh
        inventory_delta_eur = inventory_delta_kwh * tariff_eur_kwh

        # Net inventory-adjusted savings
        net_savings_eur = baseline_cost_eur - actual_cost_eur + inventory_delta_eur

        return {
            "baseline_cost_eur": round(baseline_cost_eur, 4),
            "actual_cost_eur": round(actual_cost_eur, 4),
            "total_grid_import_kwh": round(total_grid_kwh, 4),
            "inventory_delta_eur": round(inventory_delta_eur, 4),
            "net_savings_eur": round(net_savings_eur, 4),
        }

    def record_interval(
        self,
        db: Session,
        interval_start_utc: datetime,
        interval_end_utc: datetime,
        spot_price_eur_mwh: float,
        effective_price_eur_mwh: float,
        tes_charge_energy_kwh: float,
        dryer_thermal_delivered_kwh: float,
        backup_heater_energy_kwh: float,
        blower_electric_kwh: float,
        auxiliary_electric_kwh: float,
        peak_grid_power_kw: float,
        avg_tes_heat_kw: float,
        avg_dryer_demand_kw: float,
        avg_sand_temp_c: float,
        avg_dryer_supply_temp_c: float,
        avg_dryer_exhaust_temp_c: float,
        opening_soc_fraction: float,
        closing_soc_fraction: float,
        moisture_content_percent: float | None = None,
        water_removed_kg: float = 0.0,
        site_id: str = "gen0-vilnius-demo",
        data_provenance: str = DataProvenance.SIMULATED.value,
        operating_mode: str = OperatingMode.VIRTUAL.value,
        shadow_session_id: str | None = None,
        stored_energy_start_kwh: float | None = None,
        stored_energy_end_kwh: float | None = None,
        standing_loss_kwh: float = 0.0,
        dryer_heat_demand_kw: float | None = None,
        unmet_heat_kw: float = 0.0,
        optimization_price_basis: str = "EFFECTIVE_VARIABLE_PRICE",
        market_source: str = "LITGRID",
        model_version: str = "gen0-shadow-v1",
        commit: bool = True,
    ) -> EnergyLedgerInterval:
        """Persist or update an immutable settlement ledger interval."""
        start_utc = ensure_utc(interval_start_utc)
        end_utc = ensure_utc(interval_end_utc)
        duration_min = int(round((end_utc - start_utc).total_seconds() / 60.0))

        fin = self.calculate_interval_finances(
            heat_delivered_kwh=dryer_thermal_delivered_kwh,
            tes_charge_kwh=tes_charge_energy_kwh,
            backup_heat_kwh=backup_heater_energy_kwh,
            blower_kwh=blower_electric_kwh,
            aux_kwh=auxiliary_electric_kwh,
            opening_soc_fraction=opening_soc_fraction,
            closing_soc_fraction=closing_soc_fraction,
            effective_price_eur_mwh=effective_price_eur_mwh,
        )

        moisture_val = round(float(moisture_content_percent), 1) if moisture_content_percent is not None else 0.0

        stored_start = stored_energy_start_kwh if stored_energy_start_kwh is not None else round(opening_soc_fraction * self.full_capacity_kwh, 4)
        stored_end = stored_energy_end_kwh if stored_energy_end_kwh is not None else round(closing_soc_fraction * self.full_capacity_kwh, 4)
        demand_kw = dryer_heat_demand_kw if dryer_heat_demand_kw is not None else avg_dryer_demand_kw

        existing = db.execute(
            select(EnergyLedgerInterval).where(
                EnergyLedgerInterval.site_id == site_id,
                EnergyLedgerInterval.interval_start_utc == start_utc,
            )
        ).scalar_one_or_none()

        if existing:
            # Update in place idempotently
            existing.interval_end_utc = end_utc
            existing.duration_minutes = duration_min
            existing.spot_price_eur_mwh = spot_price_eur_mwh
            existing.effective_price_eur_mwh = effective_price_eur_mwh
            existing.optimization_price_basis = optimization_price_basis
            existing.market_source = market_source
            existing.model_version = model_version
            if shadow_session_id and not existing.shadow_session_id:
                existing.shadow_session_id = shadow_session_id
            existing.tes_charge_energy_kwh = tes_charge_energy_kwh
            existing.dryer_thermal_delivered_kwh = dryer_thermal_delivered_kwh
            existing.backup_heater_energy_kwh = backup_heater_energy_kwh
            existing.blower_electric_kwh = blower_electric_kwh
            existing.auxiliary_electric_kwh = auxiliary_electric_kwh
            existing.total_grid_import_kwh = fin["total_grid_import_kwh"]
            existing.standing_loss_kwh = standing_loss_kwh
            existing.peak_grid_power_kw = peak_grid_power_kw
            existing.avg_tes_heat_kw = avg_tes_heat_kw
            existing.avg_dryer_demand_kw = avg_dryer_demand_kw
            existing.dryer_heat_demand_kw = demand_kw
            existing.unmet_heat_kw = unmet_heat_kw
            existing.avg_sand_temp_c = avg_sand_temp_c
            existing.avg_dryer_supply_temp_c = avg_dryer_supply_temp_c
            existing.avg_dryer_exhaust_temp_c = avg_dryer_exhaust_temp_c
            existing.opening_soc_fraction = opening_soc_fraction
            existing.closing_soc_fraction = closing_soc_fraction
            existing.stored_energy_start_kwh = stored_start
            existing.stored_energy_end_kwh = stored_end
            existing.baseline_cost_eur = fin["baseline_cost_eur"]
            existing.actual_cost_eur = fin["actual_cost_eur"]
            existing.inventory_delta_eur = fin["inventory_delta_eur"]
            existing.net_savings_eur = fin["net_savings_eur"]
            existing.moisture_content_percent = moisture_val
            existing.water_removed_kg = water_removed_kg
            existing.data_provenance = data_provenance
            existing.operating_mode = operating_mode
            record = existing
        else:
            record = EnergyLedgerInterval(
                site_id=site_id,
                shadow_session_id=shadow_session_id,
                interval_start_utc=start_utc,
                interval_end_utc=end_utc,
                duration_minutes=duration_min,
                spot_price_eur_mwh=spot_price_eur_mwh,
                effective_price_eur_mwh=effective_price_eur_mwh,
                optimization_price_basis=optimization_price_basis,
                market_source=market_source,
                model_version=model_version,
                tes_charge_energy_kwh=tes_charge_energy_kwh,
                dryer_thermal_delivered_kwh=dryer_thermal_delivered_kwh,
                backup_heater_energy_kwh=backup_heater_energy_kwh,
                blower_electric_kwh=blower_electric_kwh,
                auxiliary_electric_kwh=auxiliary_electric_kwh,
                total_grid_import_kwh=fin["total_grid_import_kwh"],
                standing_loss_kwh=standing_loss_kwh,
                peak_grid_power_kw=peak_grid_power_kw,
                avg_tes_heat_kw=avg_tes_heat_kw,
                avg_dryer_demand_kw=avg_dryer_demand_kw,
                dryer_heat_demand_kw=demand_kw,
                unmet_heat_kw=unmet_heat_kw,
                avg_sand_temp_c=avg_sand_temp_c,
                avg_dryer_supply_temp_c=avg_dryer_supply_temp_c,
                avg_dryer_exhaust_temp_c=avg_dryer_exhaust_temp_c,
                opening_soc_fraction=opening_soc_fraction,
                closing_soc_fraction=closing_soc_fraction,
                stored_energy_start_kwh=stored_start,
                stored_energy_end_kwh=stored_end,
                baseline_cost_eur=fin["baseline_cost_eur"],
                actual_cost_eur=fin["actual_cost_eur"],
                inventory_delta_eur=fin["inventory_delta_eur"],
                net_savings_eur=fin["net_savings_eur"],
                moisture_content_percent=moisture_val,
                water_removed_kg=water_removed_kg,
                data_provenance=data_provenance,
                operating_mode=operating_mode,
            )
            db.add(record)

        if commit:
            db.commit()
        else:
            db.flush()
        return record

    def sync_from_shadow_intervals(self, db: Session) -> int:
        """Backfill/recover any executed ShadowIntervalRecord entries missing from EnergyLedgerInterval.

        Unifies durable execution and reporting paths so that past shadow runs appear in History.
        """
        from app.database.models import ShadowIntervalRecord

        existing_starts = set(
            db.execute(select(EnergyLedgerInterval.interval_start_utc)).scalars().all()
        )
        if existing_starts:
            missing_shadow = list(
                db.execute(
                    select(ShadowIntervalRecord)
                    .where(ShadowIntervalRecord.interval_start_utc.not_in(existing_starts))
                    .order_by(ShadowIntervalRecord.interval_start_utc.asc())
                ).scalars()
            )
        else:
            missing_shadow = list(
                db.execute(
                    select(ShadowIntervalRecord).order_by(ShadowIntervalRecord.interval_start_utc.asc())
                ).scalars()
            )

        if not missing_shadow:
            return 0

        added = 0
        for sir in missing_shadow:
            dur_hours = sir.duration_hours or 0.25
            heat_del_kwh = (sir.actual_discharge_kw or 0.0) * dur_hours
            chg_kwh = (sir.actual_charge_kw or 0.0) * dur_hours
            unmet_kw = max(0.0, (sir.process_demand_kw or 0.0) - (sir.actual_discharge_kw or 0.0))
            backup_kwh = unmet_kw * dur_hours
            blower_kwh = 0.05 * dur_hours
            aux_kwh = 0.05 * dur_hours

            self.record_interval(
                db=db,
                interval_start_utc=sir.interval_start_utc,
                interval_end_utc=sir.interval_end_utc,
                spot_price_eur_mwh=sir.spot_price_eur_mwh,
                effective_price_eur_mwh=sir.effective_price_eur_mwh,
                tes_charge_energy_kwh=chg_kwh,
                dryer_thermal_delivered_kwh=heat_del_kwh,
                backup_heater_energy_kwh=backup_kwh,
                blower_electric_kwh=blower_kwh,
                auxiliary_electric_kwh=aux_kwh,
                peak_grid_power_kw=sir.grid_power_total_kw or 0.0,
                avg_tes_heat_kw=sir.actual_discharge_kw or 0.0,
                avg_dryer_demand_kw=sir.process_demand_kw or 1.5,
                avg_sand_temp_c=sir.sand_temp_end_c or 197.8,
                avg_dryer_supply_temp_c=70.0,
                avg_dryer_exhaust_temp_c=40.0,
                opening_soc_fraction=sir.soc_start_fraction or 0.5,
                closing_soc_fraction=sir.soc_end_fraction or 0.5,
                moisture_content_percent=None,
                water_removed_kg=0.0,
                site_id="gen0-vilnius-demo",
                data_provenance=DataProvenance.SIMULATED.value,
                operating_mode=OperatingMode.VIRTUAL.value,
                shadow_session_id=sir.shadow_session_id,
                stored_energy_start_kwh=sir.stored_energy_start_kwh,
                stored_energy_end_kwh=sir.stored_energy_end_kwh,
                standing_loss_kwh=sir.standing_loss_kwh or 0.0,
                dryer_heat_demand_kw=sir.process_demand_kw or 1.5,
                unmet_heat_kw=unmet_kw,
                optimization_price_basis=sir.optimization_price_basis or "EFFECTIVE_VARIABLE_PRICE",
                commit=False,
            )
            added += 1

        if added > 0:
            db.commit()
        return added

    def get_ledger_history(
        self,
        db: Session,
        period: str = "today",
        limit: int = 200,
    ) -> dict[str, Any]:
        """Query intervals and summary KPIs for the History & Savings UI."""
        # Unify durable execution and reporting: recover any unsynced shadow intervals
        try:
            self.sync_from_shadow_intervals(db)
        except Exception:
            pass

        now_utc = utcnow()
        now_local = now_utc.astimezone(self.tz)

        if period == "today":
            start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_utc = start_local.astimezone(timezone.utc)
        elif period == "7d":
            cutoff_utc = now_utc - timedelta(days=7)
        elif period == "30d":
            cutoff_utc = now_utc - timedelta(days=30)
        else:
            cutoff_utc = now_utc - timedelta(days=365)

        records = db.execute(
            select(EnergyLedgerInterval)
            .where(EnergyLedgerInterval.interval_start_utc >= cutoff_utc)
            .order_by(desc(EnergyLedgerInterval.interval_start_utc))
            .limit(limit)
        ).scalars().all()

        rows: list[dict[str, Any]] = []
        tot_baseline = 0.0
        tot_actual = 0.0
        tot_savings = 0.0
        tot_heat_kwh = 0.0
        tot_grid_kwh = 0.0
        tot_tes_heat_kwh = 0.0
        tot_backup_kwh = 0.0
        tot_water_kg = 0.0
        tot_tes_charge_kwh = 0.0

        for r in records:
            dt_loc = r.interval_start_utc.astimezone(self.tz)
            time_str = dt_loc.strftime("%Y-%m-%d %H:%M")

            tot_baseline += r.baseline_cost_eur
            tot_actual += r.actual_cost_eur
            tot_savings += r.net_savings_eur
            tot_heat_kwh += r.dryer_thermal_delivered_kwh
            tot_grid_kwh += r.total_grid_import_kwh
            tot_tes_charge_kwh += (r.tes_charge_energy_kwh or 0.0)
            tot_backup_kwh += r.backup_heater_energy_kwh
            # TES heat portion = total delivered - backup
            tes_portion = max(0.0, r.dryer_thermal_delivered_kwh - r.backup_heater_energy_kwh)
            tot_tes_heat_kwh += tes_portion
            tot_water_kg += (r.water_removed_kg or 0.0)

            dur_hours = (r.duration_minutes / 60.0) if (r.duration_minutes and r.duration_minutes > 0) else 0.25
            tes_chg_kw = round((r.tes_charge_energy_kwh or 0.0) / dur_hours, 2)
            heat_del_kw = round((r.dryer_thermal_delivered_kwh or 0.0) / dur_hours, 2)
            backup_kw = round((r.backup_heater_energy_kwh or 0.0) / dur_hours, 2)

            stored_start = (
                round(r.stored_energy_start_kwh, 2)
                if r.stored_energy_start_kwh is not None
                else round(r.opening_soc_fraction * self.full_capacity_kwh, 2)
            )
            stored_end = (
                round(r.stored_energy_end_kwh, 2)
                if r.stored_energy_end_kwh is not None
                else round(r.closing_soc_fraction * self.full_capacity_kwh, 2)
            )

            rows.append({
                "id": r.id,
                "run_id": r.shadow_session_id or "-",
                "shadow_session_id": r.shadow_session_id or "-",
                "time_vilnius": time_str,
                "interval_start_local": time_str,
                "interval_start_utc": r.interval_start_utc.isoformat(),
                "interval_end_utc": r.interval_end_utc.isoformat(),
                "duration_minutes": r.duration_minutes,
                "spot_price_eur_mwh": round(r.spot_price_eur_mwh, 2),
                "effective_price_eur_mwh": round(r.effective_price_eur_mwh, 2),
                "effective_price_eur_kwh": round(r.effective_price_eur_mwh / 1000.0, 4),
                "optimization_price_basis": r.optimization_price_basis or "EFFECTIVE_VARIABLE_PRICE",
                "market_source": r.market_source or "LITGRID",
                "model_version": r.model_version or "gen0-shadow-v1",
                "heat_delivered_kwh": round(r.dryer_thermal_delivered_kwh, 2),
                "heat_delivered_kw": heat_del_kw,
                "dryer_thermal_delivered_kwh": round(r.dryer_thermal_delivered_kwh, 3),
                "dryer_heat_demand_kw": round(r.dryer_heat_demand_kw if r.dryer_heat_demand_kw is not None else (r.avg_dryer_demand_kw or 1.5), 2),
                "unmet_heat_kw": round(r.unmet_heat_kw or 0.0, 2),
                "tes_charge_energy_kwh": round(r.tes_charge_energy_kwh or 0.0, 2),
                "tes_charge_kw": tes_chg_kw,
                "backup_heater_energy_kwh": round(r.backup_heater_energy_kwh or 0.0, 3),
                "backup_heater_kw": backup_kw,
                "blower_electric_kwh": round(r.blower_electric_kwh or 0.0, 3),
                "auxiliary_electric_kwh": round(r.auxiliary_electric_kwh or 0.0, 3),
                "standing_loss_kwh": round(r.standing_loss_kwh or 0.0, 4),
                "grid_import_kwh": round(r.total_grid_import_kwh, 2),
                "soc_start_fraction": round(r.opening_soc_fraction, 3),
                "soc_end_fraction": round(r.closing_soc_fraction, 3),
                "stored_energy_start_kwh": stored_start,
                "stored_energy_end_kwh": stored_end,
                "sand_temp_end_c": round(r.avg_sand_temp_c, 1),
                "sand_temp_c": round(r.avg_sand_temp_c, 1),
                "supply_temp_c": round(r.avg_dryer_supply_temp_c, 1),
                "exhaust_temp_c": round(r.avg_dryer_exhaust_temp_c, 1),
                "baseline_direct_electric_cost_eur": round(r.baseline_cost_eur, 3),
                "baseline_cost_eur": round(r.baseline_cost_eur, 3),
                "actual_electricity_cost_eur": round(r.actual_cost_eur, 3),
                "actual_cost_eur": round(r.actual_cost_eur, 3),
                "inventory_value_delta_eur": round(r.inventory_delta_eur, 3),
                "inventory_delta_eur": round(r.inventory_delta_eur, 3),
                "net_savings_eur": round(r.net_savings_eur, 3),
                "moisture_content_percent": (
                    round(r.moisture_content_percent, 1)
                    if (r.moisture_content_percent is not None and r.moisture_content_percent > 0.0)
                    else None
                ),

                "water_removed_kg": round(r.water_removed_kg or 0.0, 2),
                "provenance": r.data_provenance,
                "operating_mode": r.operating_mode,
            })

        thermal_coverage_pct = (
            round((tot_tes_heat_kwh / tot_heat_kwh) * 100.0, 1) if tot_heat_kwh > 0 else 100.0
        )
        specific_cost_eur_kg = (
            round(tot_actual / tot_water_kg, 4) if tot_water_kg > 0.1 else 0.0
        )
        tot_inv_delta = sum(r.inventory_delta_eur for r in records)

        summary = {
            "electricity_cost_today_eur": round(tot_actual, 2),
            "today_actual_cost_eur": round(tot_actual, 2),
            "total_actual_cost_eur": round(tot_actual, 2),
            "heat_delivered_today_kwh": round(tot_heat_kwh, 2),
            "today_heat_delivered_kwh": round(tot_heat_kwh, 2),
            "total_heat_delivered_kwh": round(tot_heat_kwh, 2),
            "tes_electricity_used_kwh": round(tot_tes_charge_kwh, 2),
            "tes_charge_kwh": round(tot_tes_charge_kwh, 2),
            "total_grid_import_kwh": round(tot_grid_kwh, 2),
            "total_baseline_cost_eur": round(tot_baseline, 2),
            "total_inventory_delta_eur": round(tot_inv_delta, 2),
            "water_removed_est_kg": round(tot_water_kg, 1),
            "net_savings_today_eur": round(tot_savings, 2),
            "today_net_savings_eur": round(tot_savings, 2),
            "total_net_savings_eur": round(tot_savings, 2),
            "thermal_coverage_percent": thermal_coverage_pct,
            "specific_drying_cost_eur_kg": specific_cost_eur_kg,
        }

        return {
            "period": period,
            "record_count": len(rows),
            "total_intervals": len(rows),
            "summary": summary,
            "rows": rows,
            "intervals": rows,
        }

    def export_csv(self, db: Session, period: str = "today") -> str:
        """Export settlement ledger rows to comprehensive audit-compliant CSV format."""
        history = self.get_ledger_history(db=db, period=period, limit=2000)
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Run ID",
            "Timestamp (Europe/Vilnius)",
            "Interval Start (UTC)",
            "Interval End (UTC)",
            "Spot Price (EUR/MWh)",
            "Effective Price (EUR/MWh)",
            "Price Basis",
            "SOC Start (%)",
            "SOC End (%)",
            "Stored Energy Start (kWh)",
            "Stored Energy End (kWh)",
            "TES Charge Electricity (kWh)",
            "Dryer Heat Delivered (kWh)",
            "Backup Heater Electricity (kWh)",
            "Blower Electricity (kWh)",
            "Auxiliary Electricity (kWh)",
            "Total Grid Import (kWh)",
            "TES Standing Loss (kWh)",
            "Dryer Heat Demand (kW)",
            "Unmet Heat (kW)",
            "Timber Moisture (%)",
            "Water Removed (kg)",
            "Baseline Cost (EUR)",
            "Actual Cost (EUR)",
            "Inventory Valuation Delta (EUR)",
            "Net Savings (EUR)",
            "Sand Temp (C)",
            "Supply Temp (C)",
            "Exhaust Temp (C)",
            "Market Source",
            "Model Version",
            "Provenance",
        ])

        for r in history["rows"]:
            soc_s_pct = round(r["soc_start_fraction"] * 100.0, 1) if r.get("soc_start_fraction") is not None else ""
            soc_e_pct = round(r["soc_end_fraction"] * 100.0, 1) if r.get("soc_end_fraction") is not None else ""
            moist_str = f"{r['moisture_content_percent']:.1f}" if r.get("moisture_content_percent") is not None else "N/A"

            writer.writerow([
                r.get("run_id", "-"),
                r["time_vilnius"],
                r["interval_start_utc"],
                r["interval_end_utc"],
                r["spot_price_eur_mwh"],
                r["effective_price_eur_mwh"],
                r.get("optimization_price_basis", "EFFECTIVE_VARIABLE_PRICE"),
                soc_s_pct,
                soc_e_pct,
                r.get("stored_energy_start_kwh", ""),
                r.get("stored_energy_end_kwh", ""),
                r["tes_charge_energy_kwh"],
                r["heat_delivered_kwh"],
                r.get("backup_heater_energy_kwh", 0.0),
                r.get("blower_electric_kwh", 0.0),
                r.get("auxiliary_electric_kwh", 0.0),
                r["grid_import_kwh"],
                r.get("standing_loss_kwh", 0.0),
                r.get("dryer_heat_demand_kw", 1.5),
                r.get("unmet_heat_kw", 0.0),
                moist_str,
                r.get("water_removed_kg", 0.0),
                r["baseline_cost_eur"],
                r["actual_cost_eur"],
                r["inventory_delta_eur"],
                r["net_savings_eur"],
                r["sand_temp_c"],
                r["supply_temp_c"],
                r["exhaust_temp_c"],
                r.get("market_source", "LITGRID"),
                r.get("model_version", "gen0-shadow-v1"),
                r["provenance"],
            ])

        return output.getvalue()
