"""Pareto frontier computation and uncertainty quantification for compound hazard analysis."""

from compound_hazard.analysis.pareto import (
    compound_hazard_index,
    normalize_hazards,
    pareto_front,
    sensitivity_sweep,
    optimal_altitude,
)
from compound_hazard.analysis.uncertainty import (
    bootstrap_pareto_ci,
    monte_carlo_compound_hazard,
)

__all__ = [
    "compound_hazard_index",
    "normalize_hazards",
    "pareto_front",
    "sensitivity_sweep",
    "optimal_altitude",
    "bootstrap_pareto_ci",
    "monte_carlo_compound_hazard",
]
