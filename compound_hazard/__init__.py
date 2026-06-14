"""
compound_hazard: Compound orbital hazard analysis for commercial human spaceflight.

Combines ionizing radiation dose (anchored to NASA OSDR RadLab ISS measurements)
and space debris flux (parameterized from ESA MASTER-8 / NASA ORDEM 3.2) into
a unified Pareto-optimal orbital placement framework.

Author: Minh Nguyen <Mnguyen@xorbita.com>
Affiliation: xOrbita Inc. / NASA OSDR Active Working Group
"""

__version__ = "1.0.0"
__author__ = "Minh Nguyen"
__email__ = "Mnguyen@xorbita.com"

from compound_hazard.models.radiation import RadiationModel, D_ISS_REF_DEFAULT
from compound_hazard.models.debris import DebrisFluxModel
from compound_hazard.analysis.pareto import (
    compound_hazard_index,
    pareto_front,
    sensitivity_sweep,
)
from compound_hazard.analysis.uncertainty import (
    bootstrap_pareto_ci,
    monte_carlo_compound_hazard,
)

__all__ = [
    "RadiationModel",
    "DebrisFluxModel",
    "D_ISS_REF_DEFAULT",
    "compound_hazard_index",
    "pareto_front",
    "sensitivity_sweep",
    "bootstrap_pareto_ci",
    "monte_carlo_compound_hazard",
]
