"""Effective variable electricity price.

    effective_variable_price = spot + supplier_markup + variable_grid_fee + variable_tax

Fixed monthly charges are excluded on purpose: they do not depend on dispatch timing.
"""

from __future__ import annotations

from app.config.parameters import TariffParameters


def effective_price_eur_mwh(spot_price_eur_mwh: float, tariff: TariffParameters) -> float:
    return (
        spot_price_eur_mwh
        + tariff.supplier_markup_eur_mwh
        + tariff.variable_grid_fee_eur_mwh
        + tariff.variable_tax_eur_mwh
    )


def energy_cost_eur(power_kw: float, dt_h: float, price_eur_mwh: float) -> float:
    """Cost of drawing ``power_kw`` for ``dt_h`` hours at ``price_eur_mwh``."""
    return power_kw * dt_h * price_eur_mwh / 1000.0
