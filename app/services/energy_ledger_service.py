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
        moisture_content_percent: float = 32.0,
        water_removed_kg: float = 0.0,
        site_id: str = "gen0-vilnius-demo",
        data_provenance: str = DataProvenance.SIMULATED.value,
        operating_mode: str = OperatingMode.VIRTUAL.value,
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

        existing = db.execute(
            select(EnergyLedgerInterval).where(
                EnergyLedgerInterval.site_id == site_id,
                EnergyLedgerInterval.interval_start_utc == start_utc,
            )
        ).scalar_one_or_none()

        if existing:
            # Update in place
            existing.interval_end_utc = end_utc
            existing.duration_minutes = duration_min
            existing.spot_price_eur_mwh = spot_price_eur_mwh
            existing.effective_price_eur_mwh = effective_price_eur_mwh
            existing.tes_charge_energy_kwh = tes_charge_energy_kwh
            existing.dryer_thermal_delivered_kwh = dryer_thermal_delivered_kwh
            existing.backup_heater_energy_kwh = backup_heater_energy_kwh
            existing.blower_electric_kwh = blower_electric_kwh
            existing.auxiliary_electric_kwh = auxiliary_electric_kwh
            existing.total_grid_import_kwh = fin["total_grid_import_kwh"]
            existing.peak_grid_power_kw = peak_grid_power_kw
            existing.avg_tes_heat_kw = avg_tes_heat_kw
            existing.avg_dryer_demand_kw = avg_dryer_demand_kw
            existing.avg_sand_temp_c = avg_sand_temp_c
            existing.avg_dryer_supply_temp_c = avg_dryer_supply_temp_c
            existing.avg_dryer_exhaust_temp_c = avg_dryer_exhaust_temp_c
            existing.opening_soc_fraction = opening_soc_fraction
            existing.closing_soc_fraction = closing_soc_fraction
            existing.baseline_cost_eur = fin["baseline_cost_eur"]
            existing.actual_cost_eur = fin["actual_cost_eur"]
            existing.inventory_delta_eur = fin["inventory_delta_eur"]
            existing.net_savings_eur = fin["net_savings_eur"]
            existing.moisture_content_percent = moisture_content_percent
            existing.water_removed_kg = water_removed_kg
            existing.data_provenance = data_provenance
            existing.operating_mode = operating_mode
            record = existing
        else:
            record = EnergyLedgerInterval(
                site_id=site_id,
                interval_start_utc=start_utc,
                interval_end_utc=end_utc,
                duration_minutes=duration_min,
                spot_price_eur_mwh=spot_price_eur_mwh,
                effective_price_eur_mwh=effective_price_eur_mwh,
                tes_charge_energy_kwh=tes_charge_energy_kwh,
                dryer_thermal_delivered_kwh=dryer_thermal_delivered_kwh,
                backup_heater_energy_kwh=backup_heater_energy_kwh,
                blower_electric_kwh=blower_electric_kwh,
                auxiliary_electric_kwh=auxiliary_electric_kwh,
                total_grid_import_kwh=fin["total_grid_import_kwh"],
                peak_grid_power_kw=peak_grid_power_kw,
                avg_tes_heat_kw=avg_tes_heat_kw,
                avg_dryer_demand_kw=avg_dryer_demand_kw,
                avg_sand_temp_c=avg_sand_temp_c,
                avg_dryer_supply_temp_c=avg_dryer_supply_temp_c,
                avg_dryer_exhaust_temp_c=avg_dryer_exhaust_temp_c,
                opening_soc_fraction=opening_soc_fraction,
                closing_soc_fraction=closing_soc_fraction,
                baseline_cost_eur=fin["baseline_cost_eur"],
                actual_cost_eur=fin["actual_cost_eur"],
                inventory_delta_eur=fin["inventory_delta_eur"],
                net_savings_eur=fin["net_savings_eur"],
                moisture_content_percent=moisture_content_percent,
                water_removed_kg=water_removed_kg,
                data_provenance=data_provenance,
                operating_mode=operating_mode,
            )
            db.add(record)

        db.commit()
        return record

    def get_ledger_history(
        self,
        db: Session,
        period: str = "today",
        limit: int = 200,
    ) -> dict[str, Any]:
        """Query intervals and summary KPIs for the History & Savings UI."""
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

        for r in records:
            dt_loc = r.interval_start_utc.astimezone(self.tz)
            time_str = dt_loc.strftime("%Y-%m-%d %H:%M")

            tot_baseline += r.baseline_cost_eur
            tot_actual += r.actual_cost_eur
            tot_savings += r.net_savings_eur
            tot_heat_kwh += r.dryer_thermal_delivered_kwh
            tot_grid_kwh += r.total_grid_import_kwh
            tot_backup_kwh += r.backup_heater_energy_kwh
            # TES heat portion = total delivered - backup
            tes_portion = max(0.0, r.dryer_thermal_delivered_kwh - r.backup_heater_energy_kwh)
            tot_tes_heat_kwh += tes_portion
            tot_water_kg += r.water_removed_kg

            rows.append({
                "time_vilnius": time_str,
                "interval_start_utc": r.interval_start_utc.isoformat(),
                "spot_price_eur_mwh": round(r.spot_price_eur_mwh, 2),
                "effective_price_eur_mwh": round(r.effective_price_eur_mwh, 2),
                "heat_delivered_kwh": round(r.dryer_thermal_delivered_kwh, 2),
                "grid_import_kwh": round(r.total_grid_import_kwh, 2),
                "baseline_cost_eur": round(r.baseline_cost_eur, 3),
                "actual_cost_eur": round(r.actual_cost_eur, 3),
                "inventory_delta_eur": round(r.inventory_delta_eur, 3),
                "net_savings_eur": round(r.net_savings_eur, 3),
                "sand_temp_c": round(r.avg_sand_temp_c, 1),
                "supply_temp_c": round(r.avg_dryer_supply_temp_c, 1),
                "exhaust_temp_c": round(r.avg_dryer_exhaust_temp_c, 1),
                "provenance": r.data_provenance,
            })

        thermal_coverage_pct = (
            round((tot_tes_heat_kwh / tot_heat_kwh) * 100.0, 1) if tot_heat_kwh > 0 else 100.0
        )
        specific_cost_eur_kg = (
            round(tot_actual / tot_water_kg, 4) if tot_water_kg > 0.1 else 0.0
        )

        return {
            "period": period,
            "record_count": len(rows),
            "summary": {
                "electricity_cost_today_eur": round(tot_actual, 2),
                "heat_delivered_today_kwh": round(tot_heat_kwh, 2),
                "water_removed_est_kg": round(tot_water_kg, 1),
                "net_savings_today_eur": round(tot_savings, 2),
                "thermal_coverage_percent": thermal_coverage_pct,
                "specific_drying_cost_eur_kg": specific_cost_eur_kg,
            },
            "rows": rows,
        }

    def export_csv(self, db: Session, period: str = "today") -> str:
        """Export settlement ledger rows to audit-compliant CSV format."""
        history = self.get_ledger_history(db=db, period=period, limit=2000)
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Timestamp (Europe/Vilnius)",
            "Interval Start (UTC)",
            "Spot Price (EUR/MWh)",
            "Effective Price (EUR/MWh)",
            "Dryer Heat Delivered (kWh)",
            "Grid Import (kWh)",
            "Baseline Cost (EUR)",
            "Actual Cost (EUR)",
            "Inventory Delta (EUR)",
            "Net Savings (EUR)",
            "Sand Temp (C)",
            "Supply Temp (C)",
            "Exhaust Temp (C)",
            "Provenance",
        ])

        for r in history["rows"]:
            writer.writerow([
                r["time_vilnius"],
                r["interval_start_utc"],
                r["spot_price_eur_mwh"],
                r["effective_price_eur_mwh"],
                r["heat_delivered_kwh"],
                r["grid_import_kwh"],
                r["baseline_cost_eur"],
                r["actual_cost_eur"],
                r["inventory_delta_eur"],
                r["net_savings_eur"],
                r["sand_temp_c"],
                r["supply_temp_c"],
                r["exhaust_temp_c"],
                r["provenance"],
            ])

        return output.getvalue()
