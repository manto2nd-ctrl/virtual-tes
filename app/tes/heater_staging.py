"""Domain interface and actuation models for TES electrical heating.

Phase 5.8.3 distinguishes between:
1. CONTINUOUS OPTIMIZER REQUEST: the ideal continuous thermal charging rate
   computed by the LP optimizer (e.g. 2.67 kW).
2. PHYSICAL HEATER STAGING: the future discrete on/off cartridge heater switching
   (e.g. 6 x 1.5 kW drywell heaters) constrained by electrical wiring, phase balance,
   and contactor cycling rules.

Current mode: CONTINUOUS_SHADOW_MODEL (IdealizedContinuousHeaterModel).
Physical hardware staging remains future work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeaterStagingResult:
    """Outcome of mapping continuous optimizer power request to actuation staging."""

    requested_power_kw: float
    staged_power_kw: float
    active_heaters_count: int
    total_cartridges_count: int
    actuation_model: str  # "CONTINUOUS IDEALIZED" or "DISCRETE STAGED"
    hardware_staging_implemented: bool
    staging_warning: str
    discrete_stage_kw: float | None = None

    @property
    def actual_power_kw(self) -> float:
        return self.staged_power_kw

    @property
    def model_type(self) -> str:
        return self.actuation_model

    @property
    def warning_note(self) -> str:
        return self.staging_warning

    @property
    def active_stages(self) -> list[float]:
        return []


class HeaterStagingModel(ABC):
    """Abstract interface for heater actuation staging."""

    @abstractmethod
    def stage_power(self, requested_power_kw: float) -> HeaterStagingResult:
        """Map continuous requested electrical power to staged physical actuation."""
        ...

    def dispatch(self, requested_power_kw: float) -> HeaterStagingResult:
        """Alias for stage_power."""
        return self.stage_power(requested_power_kw)


class IdealizedContinuousHeaterModel(HeaterStagingModel):
    """Current Gen0 Phase 5.8 shadow model: continuous idealized power.

    Enforces rated installed capacity (default 9.0 kW) without altering the LP
    solution, while explicitly declaring that discrete hardware cartridge staging
    is not yet modeled.
    """

    def __init__(
        self,
        max_power_kw: float = 9.0,
        p_max_kw: float | None = None,
        cartridge_rating_kw: float = 1.5,
        num_cartridges: int = 6,
    ) -> None:
        self.max_power_kw = float(p_max_kw if p_max_kw is not None else max_power_kw)
        self.cartridge_rating_kw = float(cartridge_rating_kw)
        self.num_cartridges = int(num_cartridges)

    def stage_power(self, requested_power_kw: float) -> HeaterStagingResult:
        clipped = max(0.0, min(float(requested_power_kw), self.max_power_kw))
        # Equivalent cartridge count if ideal fraction
        approx_cartridges = (
            int(round(clipped / self.cartridge_rating_kw))
            if self.cartridge_rating_kw > 0
            else 0
        )
        return HeaterStagingResult(
            requested_power_kw=round(requested_power_kw, 2),
            staged_power_kw=round(clipped, 2),
            active_heaters_count=approx_cartridges,
            total_cartridges_count=self.num_cartridges,
            actuation_model="CONTINUOUS IDEALIZED",
            hardware_staging_implemented=False,
            staging_warning="Continuous optimizer request — physical heater staging not yet modeled.",
            discrete_stage_kw=None,
        )


class FutureDiscreteHeaterStagingModel(HeaterStagingModel):
    """Placeholder domain model for future discrete 3-phase cartridge staging.

    Example discrete stages: 0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0 kW.
    Note: Do not assume these exact stages are physically valid until actual
    three-phase wiring and relay interlocks are fabricated and calibrated.
    """

    def __init__(
        self,
        cartridge_rating_kw: float = 1.5,
        num_cartridges: int = 6,
    ) -> None:
        self.cartridge_rating_kw = float(cartridge_rating_kw)
        self.num_cartridges = int(num_cartridges)
        self.stages = [round(i * self.cartridge_rating_kw, 2) for i in range(self.num_cartridges + 1)]

    def stage_power(self, requested_power_kw: float) -> HeaterStagingResult:
        raise NotImplementedError("Physical discrete heater staging (6 x 1.5 kW drywells) remains future work; not yet implemented.")

    def dispatch(self, requested_power_kw: float) -> HeaterStagingResult:
        return self.stage_power(requested_power_kw)
