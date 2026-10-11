"""Phase 5.11 - Stage 3 Validation Test Suite.

Validates:
  A. SOC and temperature:
     - Canonical enthalpy model with Cp(T_K) = T_K + 427 J/(kg*K) in Kelvin.
     - 0% physical SOC ≈ 80°C.
     - 50% physical SOC ≈ 196.8°C.
     - 100% physical SOC ≈ 300°C.
     - Sand mass ≈ 275.7 kg for 15.0 kWh full span.
     - Zero duplicate frontend thermodynamic calculations via canonical API conversion.

  B. Thermal power availability:
     - P_HX_available = f(T_sand, airflow, U, HX_area, T_inlet).
     - P_HX_available at 300°C ≈ 2.79 kW; at 196.8°C ≈ 1.68 kW.
     - P_HX_delivered <= P_HX_available.
     - P_HX_delivered <= available stored thermal energy / step.
     - Clear differentiation between maximum available and delivered process heat.

  C. Energy conservation:
     - E_end = E_start + E_added - E_extracted - E_loss.
     - Residual < 1e-6 kWh across charging, discharging, and idle steps.
     - Strict clamping at physical capacity boundaries [0.0, 15.0] kWh.
     - When at 100% SOC, additional charging cannot increase stored energy.
     - Curtailed charging requests are stored separately.

  D. Grid constraints:
     - Total grid import = P_heater + P_backup + P_blower + P_aux + P_site <= 12.0 kW.
     - Grid headroom calculation dynamically accounts for all electrical loads.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
import pytest
from starlette.testclient import TestClient

from app.config.parameters import SiteParameters, TariffParameters, TESParameters
from app.process.dryer import (
    DryerControlMode,
    DryerOperatingConfig,
    DryingRecipePreset,
    VirtualDryerModel,
)
from app.services.energy_system_state import get_energy_system_state
from app.services.shadow_runtime import ShadowRuntimeService
from app.tes.model import (
    ConstantDischargeLimit,
    TESStepResult,
    compute_step,
)
from app.tes.thermal import HelicalAirHXModel, ThermalStateMapper
from app.web.main import app


# ==============================================================================
# CHECK A: SOC and Temperature Canonical Enthalpy Model
# ==============================================================================

class TestCheckA_SOCAndTemperature:
    """Validate canonical quartz sand enthalpy thermodynamics."""

    def test_canonical_enthalpy_reference_points(self) -> None:
        """Verify: 0% SOC ≈ 80°C, 50% SOC ≈ 196.8°C, 100% SOC ≈ 300°C."""
        mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)

        # 0% SOC must be exactly 80.0 °C
        t_0 = mapper.temperature_from_soc_fraction(0.0)
        assert abs(t_0 - 80.0) < 1e-4, f"Expected 80.0°C at 0% SOC, got {t_0}"

        # 100% SOC must be exactly 300.0 °C
        t_100 = mapper.temperature_from_soc_fraction(1.0)
        assert abs(t_100 - 300.0) < 1e-4, f"Expected 300.0°C at 100% SOC, got {t_100}"

        # 50% SOC must be approximately 196.8 °C (exact analytical: ~196.77 °C)
        t_50 = mapper.temperature_from_soc_fraction(0.50)
        assert abs(t_50 - 196.8) < 0.15, f"Expected ~196.8°C at 50% SOC, got {t_50}"

    def test_sand_mass_and_specific_capacity(self) -> None:
        """Verify equivalent quartz sand mass is ~275.7 kg for 15.0 kWh full span."""
        mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)

        sand_mass = mapper.equivalent_sand_mass_for_capacity(15.0)
        assert abs(sand_mass - 275.7) < 0.2, f"Expected ~275.7 kg, got {sand_mass}"

        specific_cap = mapper.specific_stored_energy_kwh_per_kg()
        assert abs(specific_cap - 0.0544) < 0.001, f"Expected ~0.0544 kWh/kg, got {specific_cap}"

    def test_monotonicity_and_round_trip(self) -> None:
        """Verify strict monotonicity and exact round-trip inverse mapping."""
        mapper = ThermalStateMapper(t_min_c=80.0, t_max_c=300.0)

        # Monotonicity
        temps = [mapper.temperature_from_soc_fraction(s / 100.0) for s in range(0, 101, 5)]
        for i in range(len(temps) - 1):
            assert temps[i] < temps[i + 1], f"Temperature not strictly increasing at index {i}"

        # Round-trip: T -> SOC -> T
        for test_t in [80.0, 100.0, 150.0, 196.77, 250.0, 300.0]:
            soc = mapper.soc_fraction_from_temperature(test_t)
            recovered_t = mapper.temperature_from_soc_fraction(soc)
            assert abs(recovered_t - test_t) < 1e-4, f"Roundtrip mismatch: {test_t} vs {recovered_t}"

    def test_server_side_conversion_endpoint(self) -> None:
        """Verify backend /api/calculate/thermal-convert provides authoritative conversion."""
        client = TestClient(app)

        # 50% SOC
        resp = client.post("/api/calculate/thermal-convert", json={"mode": "soc", "soc_percent": 50.0})
        assert resp.status_code == 200
        d = resp.json()
        assert abs(d["sand_temperature_c"] - 196.8) < 0.2
        assert abs(d["stored_energy_kwh"] - 7.50) < 0.05

        # From temperature 300°C -> 100% SOC
        resp2 = client.post("/api/calculate/thermal-convert", json={"mode": "temp", "sand_temperature_c": 300.0})
        assert resp2.status_code == 200
        d2 = resp2.json()
        assert abs(d2["soc_percent"] - 100.0) < 0.1
        assert abs(d2["stored_energy_kwh"] - 15.0) < 0.05


# ==============================================================================
# CHECK B: Thermal Power Availability
# ==============================================================================

class TestCheckB_ThermalPowerAvailability:
    """Validate heat exchanger physical availability and process coupling."""

    def test_hx_max_available_power_reference_values(self) -> None:
        """Verify: at 300°C P_HX_available ≈ 2.79 kW; at 196.8°C ≈ 1.68 kW."""
        mapper = ThermalStateMapper(80.0, 300.0)
        hx = HelicalAirHXModel(
            hx_area_m2=1.55,
            overall_u_w_m2k=9.0,
            airflow_m3_h=80.0,
            air_inlet_temperature_c=40.0,
            thermal_state_mapper=mapper,
        )

        p_300 = hx.p_max_at_temperature_kw(300.0)
        assert abs(p_300 - 2.79) < 0.05, f"Expected ~2.79 kW at 300°C, got {p_300}"

        p_50 = hx.p_max_at_soc_fraction_kw(0.50)
        assert abs(p_50 - 1.68) < 0.05, f"Expected ~1.68 kW at 50% SOC (~196.8°C), got {p_50}"

    def test_p_hx_delivered_bounded_by_available_and_stored_energy(self) -> None:
        """Verify: P_HX_delivered <= P_HX_available and P_HX_delivered <= available_energy / step."""
        mapper = ThermalStateMapper(80.0, 300.0)
        hx = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=9.0, airflow_m3_h=80.0, air_inlet_temperature_c=40.0)
        dryer = VirtualDryerModel(hx_model=hx, thermal_mapper=mapper)

        # Case 1: High sand temp (300°C), ample stored energy (15.0 kWh), demand = 1.50 kW
        # Delivered should be min(1.50, 2.79) = 1.50 kW
        res1 = dryer.step(sand_temperature_c=300.0, available_stored_energy_kwh=15.0)
        assert res1.hx_max_available_kw >= 2.75
        assert abs(res1.heat_supplied_from_tes_kw - 1.50) < 1e-3
        assert res1.backup_heater_power_kw == 0.0

        # Case 2: Sand temp 300°C (HX available = 2.79 kW), but stored energy is severely limited: 0.20 kWh in 15 min (0.25 h)
        # Max power from stored energy = 0.20 kWh * 0.90 / 0.25 h = 0.72 kW
        res2 = dryer.step(sand_temperature_c=300.0, available_stored_energy_kwh=0.20, dt_seconds=900.0)
        assert res2.heat_supplied_from_tes_kw <= 0.72 + 1e-4
        assert res2.heat_supplied_from_tes_kw <= res2.hx_max_available_kw
        # Backup heater covers remainder: 1.50 - 0.72 = 0.78 kW
        assert abs(res2.backup_heater_power_kw - (1.50 - res2.heat_supplied_from_tes_kw)) < 1e-3

    def test_differentiation_between_max_available_and_delivered(self) -> None:
        """Verify maximum available HX power is explicitly distinguished from actual delivered power."""
        mapper = ThermalStateMapper(80.0, 300.0)
        hx = HelicalAirHXModel(hx_area_m2=1.55, overall_u_w_m2k=9.0, airflow_m3_h=80.0, air_inlet_temperature_c=40.0)
        dryer = VirtualDryerModel(hx_model=hx, thermal_mapper=mapper)

        res = dryer.step(sand_temperature_c=300.0, available_stored_energy_kwh=10.0)
        # Max available is ~2.79 kW, but delivered is 1.50 kW
        assert res.hx_max_available_kw > res.heat_supplied_from_tes_kw
        assert abs(res.hx_max_available_kw - 2.79) < 0.05
        assert abs(res.heat_supplied_from_tes_kw - 1.50) < 0.01


# ==============================================================================
# CHECK C: Energy Conservation
# ==============================================================================

class TestCheckC_EnergyConservation:
    """Validate strict energy conservation, physical clamping, and curtailment tracking."""

    @pytest.fixture
    def tes_params(self) -> TESParameters:
        return TESParameters(
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

    def test_energy_conservation_residual_zero(self, tes_params: TESParameters) -> None:
        """Verify E_end = E_start + E_added - E_extracted - E_loss with residual < 1e-6 kWh."""
        dt_h = 0.25

        # Test varying operational setpoints
        test_cases = [
            (7.5, 9.0, 0.0),    # Full charge, no discharge
            (10.0, 0.0, 2.5),   # No charge, high discharge
            (6.0, 4.0, 1.5),    # Simultaneous charge and discharge
            (14.0, 0.0, 0.0),   # Standing idle (losses only)
        ]

        for s_start, chg, dis in test_cases:
            res: TESStepResult = compute_step(
                params=tes_params,
                soc_kwh=s_start,
                dt_h=dt_h,
                requested_charge_kw=chg,
                requested_discharge_kw=dis,
            )

            expected_delta = res.energy_stored_kwh - res.energy_withdrawn_kwh - res.storage_loss_kwh
            actual_delta = res.soc_end_kwh - s_start
            residual = abs(actual_delta - expected_delta)
            assert residual < 1e-6, f"Energy conservation violated! residual={residual:.8f} kWh"

    def test_100_percent_soc_clamping_and_curtailment(self, tes_params: TESParameters) -> None:
        """Verify: at 100% SOC, charging cannot exceed capacity and curtailed charge is stored separately."""
        dt_h = 0.25
        res = compute_step(
            params=tes_params,
            soc_kwh=15.0,  # 100% SOC
            dt_h=dt_h,
            requested_charge_kw=9.0,
            requested_discharge_kw=0.0,
        )

        # Stored energy cannot exceed 15.0 kWh
        assert res.soc_end_kwh <= 15.0 + 1e-9
        assert abs(res.soc_end_kwh - 15.0) < 1e-5

        # Most of the 9.0 kW request must be curtailed (only loss compensation allowed)
        assert res.charge_power_kw < 0.15
        assert res.curtailed_charge_kw > 8.85
        assert res.curtailed_charge_kwh > (8.85 * dt_h)
        assert "soc_max" in res.charge_limited_by

    def test_no_double_counting_of_losses(self, tes_params: TESParameters) -> None:
        """Verify efficiencies eta_c and eta_d are applied consistently without double counting."""
        dt_h = 1.0  # 1 hour
        # Start at 5.0 kWh, charge with 4.0 kW electric for 1 hour with eta_c=0.95
        # Loss per hour = (1 - (1 - 0.02)**(1/24)) * 5.0 kWh
        res = compute_step(
            params=tes_params,
            soc_kwh=5.0,
            dt_h=dt_h,
            requested_charge_kw=4.0,
            requested_discharge_kw=0.0,
        )

        assert abs(res.energy_stored_kwh - (4.0 * 0.95 * 1.0)) < 1e-6
        assert abs(res.charge_conversion_loss_kwh - (4.0 * 0.05 * 1.0)) < 1e-6


# ==============================================================================
# CHECK D: Grid Constraints
# ==============================================================================

class TestCheckD_GridConstraints:
    """Validate site electrical balance and strict 12.0 kW limit enforcement."""

    def test_grid_limit_enforced_across_all_loads(self) -> None:
        """Verify total grid import = P_heater + P_backup + P_blower + P_aux + P_site <= 12.0 kW."""
        tes_params = TESParameters(
            thermal_capacity_full_span_kwh=15.0,
            max_charge_power_kw=9.0,
            auxiliary_power_kw=0.05,
        )
        site_params = SiteParameters(
            grid_connection_limit_kw=12.0,
            other_loads_kw=2.0,
            process_heat_demand_kw=1.5,
        )

        # Baseline non-TES loads
        p_site = site_params.other_loads_kw      # 2.0 kW
        p_aux = tes_params.auxiliary_power_kw     # 0.05 kW
        p_blower = 0.05                          # 0.05 kW
        p_backup = 1.50                          # 1.50 kW backup heater
        non_tes = p_site + p_aux + p_blower + p_backup  # 3.60 kW

        # Headroom available for TES heater
        headroom = max(0.0, site_params.grid_connection_limit_kw - non_tes)  # 8.40 kW

        # If TES requests 9.0 kW, it must be capped to headroom (8.40 kW)
        res = compute_step(
            params=tes_params,
            soc_kwh=5.0,
            dt_h=0.25,
            requested_charge_kw=9.0,
            requested_discharge_kw=0.0,
            external_charge_limit_kw=headroom,
        )

        assert res.charge_power_kw <= headroom + 1e-6
        assert "grid_limit" in res.charge_limited_by

        total_import = res.charge_power_kw + p_backup + p_blower + p_aux + p_site
        assert total_import <= 12.0 + 1e-6
        assert abs(total_import - 12.0) < 1e-5

    def test_shadow_runtime_grid_balance_clamp(self) -> None:
        """Verify ShadowRuntimeService enforces strict 12.0 kW limit during step execution."""
        runtime = ShadowRuntimeService()
        now = datetime.now(timezone.utc)
        start_utc = now.replace(minute=0, second=0, microsecond=0)
        end_utc = start_utc.replace(minute=15)

        from unittest.mock import MagicMock
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = None

        from app.database.models import ShadowTESSession
        session = ShadowTESSession(
            id="test-session-grid",
            status="RUNNING",
            current_soc_fraction=0.10,
            current_stored_energy_kwh=1.50,
            current_sand_temperature_c=80.0,
            process_enabled=True,
            process_heat_demand_kw=1.50,
            planned_schedule=[],
        )

        # Request full 9.0 kW charge with dryer running (which requires backup heating since sand is cold)
        rec = runtime.execute_interval_step(
            db=mock_db,
            shadow_session=session,
            interval_start_utc=start_utc,
            interval_end_utc=end_utc,
            spot_price_eur_mwh=50.0,
            requested_charge_kw=9.0,
        )

        assert rec.grid_power_total_kw <= 12.0 + 1e-6
        assert rec.grid_power_limit_kw == 12.0
