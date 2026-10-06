"""Historical backtesting package for Virtual TES."""

from app.backtest.domain import (
    BacktestConfig,
    BacktestReport,
    BacktestStrategyMetrics,
)
from app.backtest.engine import (
    BacktestRunner,
    evaluate_cheapest_n_heuristic,
    evaluate_direct_electric_heating,
    evaluate_perfect_foresight_lp,
    evaluate_realistic_rolling_lp,
)
from app.backtest.sensitivity import SensitivityAnalyzer, SensitivityCaseResult

__all__ = [
    "BacktestConfig",
    "BacktestReport",
    "BacktestStrategyMetrics",
    "BacktestRunner",
    "SensitivityAnalyzer",
    "SensitivityCaseResult",
    "evaluate_direct_electric_heating",
    "evaluate_cheapest_n_heuristic",
    "evaluate_realistic_rolling_lp",
    "evaluate_perfect_foresight_lp",
]
