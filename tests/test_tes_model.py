import pytest
from datetime import datetime, timezone

from app.config.parameters import TESParameters
from app.tes.model import EPS_KWH, VirtualTES, compute_step, retention_factor


@pytest.fixture
def default_tes_params():
    return TESParameters(
        capacity_kwh=15.0,
        max_charge_power_kw=9.0,
        max_discharge_power_kw=3.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.90,
        standing_loss_percent_per_day=2.0,
        soc_min_percent=10.0,
        soc_max_percent=100.0,
        initial_soc_percent=50.0,
    )


def test_standing_losses_retention(default_tes_params):
    # Over 24 hours of idling, retention should be exactly 98%
    r_24h = retention_factor(2.0, 24.0)
    assert abs(r_24h - 0.98) < 1e-12

    # 15 minutes = 0.25 h
    r_15m = retention_factor(2.0, 0.25)
    step = compute_step(
        params=default_tes_params,
        soc_kwh=10.0,
        dt_h=0.25,
        requested_charge_kw=0.0,
        requested_discharge_kw=0.0,
    )
    expected_loss = 10.0 * (1.0 - r_15m)
    assert abs(step.storage_loss_kwh - expected_loss) < 1e-12
    assert abs(step.soc_end_kwh - 10.0 * r_15m) < 1e-12


def test_charge_clipped_to_max_charge_power(default_tes_params):
    # Request 20 kW charge when max is 9 kW
    step = compute_step(
        params=default_tes_params,
        soc_kwh=5.0,
        dt_h=0.25,
        requested_charge_kw=20.0,
        requested_discharge_kw=0.0,
    )
    assert step.charge_power_kw == 9.0
    assert "max_charge_power" in step.charge_limited_by


def test_discharge_clipped_to_max_discharge_power(default_tes_params):
    # Request 5 kW discharge when max is 3 kW
    step = compute_step(
        params=default_tes_params,
        soc_kwh=10.0,
        dt_h=0.25,
        requested_charge_kw=0.0,
        requested_discharge_kw=5.0,
    )
    assert step.discharge_power_kw == 3.0
    assert "max_discharge_power" in step.discharge_limited_by


def test_soc_cannot_exceed_capacity(default_tes_params):
    # Near top (14.5 kWh), request full charge 9 kW for 15 min (would store 2.1375 kWh, exceeding 15 kWh)
    step = compute_step(
        params=default_tes_params,
        soc_kwh=14.5,
        dt_h=0.25,
        requested_charge_kw=9.0,
        requested_discharge_kw=0.0,
    )
    assert step.soc_end_kwh <= default_tes_params.soc_max_kwh + EPS_KWH
    assert step.soc_end_kwh == pytest.approx(default_tes_params.soc_max_kwh, abs=1e-8)
    assert "soc_max" in step.charge_limited_by


def test_soc_cannot_fall_below_soc_min(default_tes_params):
    # Near bottom (2.0 kWh, min is 1.5 kWh = 10%), try to discharge 3 kW for 15 min (withdrawing 3*0.25/0.9 = 0.833 kWh)
    step = compute_step(
        params=default_tes_params,
        soc_kwh=2.0,
        dt_h=0.25,
        requested_charge_kw=0.0,
        requested_discharge_kw=3.0,
    )
    assert step.soc_end_kwh >= default_tes_params.soc_min_kwh - EPS_KWH
    assert step.soc_end_kwh == pytest.approx(default_tes_params.soc_min_kwh, abs=1e-8)
    assert "soc_min" in step.discharge_limited_by
    # Delivered heat should be clipped proportionally
    assert step.discharge_power_kw < 3.0


def test_external_grid_headroom_clipping(default_tes_params):
    # Grid headroom is 5 kW (e.g. 12 kW grid limit - 7 kW other load)
    # Even if max charge is 9 kW and requested is 9 kW, charge power must not exceed 5 kW
    step = compute_step(
        params=default_tes_params,
        soc_kwh=5.0,
        dt_h=0.25,
        requested_charge_kw=9.0,
        requested_discharge_kw=0.0,
        external_charge_limit_kw=5.0,
    )
    assert step.charge_power_kw == 5.0
    assert "grid_limit" in step.charge_limited_by


def test_energy_balance_conservation(default_tes_params):
    # Simultaneous charging (6 kW) and discharging (1.5 kW)
    dt = 0.25
    soc_start = 7.5
    step = compute_step(
        params=default_tes_params,
        soc_kwh=soc_start,
        dt_h=dt,
        requested_charge_kw=6.0,
        requested_discharge_kw=1.5,
    )

    # Conservation equation:
    # E_elec = Heat_delivered + delta_SOC + Standing_loss + Charge_loss + Discharge_loss
    e_elec = step.charge_power_kw * dt
    heat_delivered = step.discharge_power_kw * dt
    delta_soc = step.soc_end_kwh - soc_start

    total_accounted = (
        heat_delivered
        + delta_soc
        + step.storage_loss_kwh
        + step.charge_conversion_loss_kwh
        + step.discharge_conversion_loss_kwh
    )
    assert abs(e_elec - total_accounted) < 1e-12


def test_pluggable_soc_dependent_discharge_limit(default_tes_params):
    from app.tes.model import ConstantDischargeLimit, LinearDeratingDischargeLimit

    # Default curve allows full 3.0 kW
    curve_const = ConstantDischargeLimit()
    assert curve_const.max_discharge_power_kw(3.0, default_tes_params) == 3.0

    # Derating curve: derate from 30% SOC down to 10% (soc_min) with 0.5 kW floor
    curve_derate = LinearDeratingDischargeLimit(
        derate_start_soc_percent=30.0,
        min_power_at_soc_min_kw=0.5,
    )

    # At 50% SOC (7.5 kWh): above 30%, returns full 3.0 kW
    assert curve_derate.max_discharge_power_kw(7.5, default_tes_params) == 3.0

    # At 10% SOC (1.5 kWh = soc_min): returns floor 0.5 kW
    assert curve_derate.max_discharge_power_kw(1.5, default_tes_params) == 0.5

    # At 20% SOC (3.0 kWh): exactly halfway between 10% and 30%, returns (3.0 + 0.5) / 2 = 1.75 kW
    p_20pct = curve_derate.max_discharge_power_kw(3.0, default_tes_params)
    assert abs(p_20pct - 1.75) < 1e-9

    # Test that compute_step clips discharge power using the curve
    step = compute_step(
        params=default_tes_params,
        soc_kwh=3.0,
        dt_h=0.25,
        requested_charge_kw=0.0,
        requested_discharge_kw=3.0,
        discharge_limit_curve=curve_derate,
    )
    assert abs(step.discharge_power_kw - 1.75) < 1e-9
    assert "soc_dependent_discharge_limit" in step.discharge_limited_by

    # Verify that energy balance conservation holds with derating curve
    e_elec = step.charge_power_kw * 0.25
    total_acc = (
        step.discharge_power_kw * 0.25
        + (step.soc_end_kwh - 3.0)
        + step.storage_loss_kwh
        + step.charge_conversion_loss_kwh
        + step.discharge_conversion_loss_kwh
    )
    assert abs(e_elec - total_acc) < 1e-12


def test_provisional_standing_loss_configuration():
    # Verify provisional status is explicitly surfaced
    params_provisional = TESParameters(
        standing_loss_is_provisional=True,
        standing_loss_model="provisional_fixed_rate_kw",
        standing_loss_fixed_kw=0.08,  # 80 W fixed standing loss
    )
    assert params_provisional.standing_loss_is_provisional is True
    assert params_provisional.standing_loss_model == "provisional_fixed_rate_kw"

    dt = 0.25
    soc_start = 10.0
    step = compute_step(
        params=params_provisional,
        soc_kwh=soc_start,
        dt_h=dt,
        requested_charge_kw=0.0,
        requested_discharge_kw=0.0,
    )
    expected_loss = 0.08 * 0.25  # 0.02 kWh
    assert abs(step.storage_loss_kwh - expected_loss) < 1e-12
    assert abs(step.soc_end_kwh - (soc_start - expected_loss)) < 1e-12

    # Verify core energy balance conservation
    total_acc = (
        step.discharge_power_kw * dt
        + (step.soc_end_kwh - soc_start)
        + step.storage_loss_kwh
        + step.charge_conversion_loss_kwh
        + step.discharge_conversion_loss_kwh
    )
    assert abs(total_acc) < 1e-12

