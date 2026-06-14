"""
Pareto frontier computation and compound hazard optimization.

The compound hazard index H(α) is a weighted sum of normalized radiation
dose and debris flux over the practical LEO altitude range (370–1200 km):

    H(α) = α · D_norm + (1 − α) · F_norm

where α ∈ [0, 1] is the weighting parameter (α=1 ₒ radiation-only,
±=0 → debris-only), and D_norm, F_norm are both normalized to [0, 1]
over the domain.

Pareto-optimal altitudes are those not dominated in both objectives
simultaneously — i.e., no other altitude has both lower radiation dose
AND lower debris flux.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize_hazards(
    dose_vals: NDArray[np.float64],
    debris_vals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Normalize both hazard metrics to [0, 1] over their shared domain.

    Parameters
    ----------
    dose_vals : NDArray[np.float64]
        Radiation dose values (any units).
    debris_vals : NDArray[np.float64]
        Debris flux index values (any units).

    Returns
    -------
    tuple of (dose_norm, debris_norm) each in [0, 1].
    """
    dose_norm = (dose_vals - dose_vals.min()) / (dose_vals.max() - dose_vals.min())
    debris_norm = (debris_vals - debris_vals.min()) / (debris_vals.max() - debris_vals.min())
    return np.clip(dose_norm, 0.0, 1.0), np.clip(debris_norm, 0.0, 1.0)


def compound_hazard_index(
    dose_norm: NDArray[np.float64],
    debris_norm: NDArray[np.float64],
    alpha: float = 0.5,
) -> NDArray[np.float64]:
    """
    Compute the compound hazard index.

    H(α) = α · dose_norm + (1 − α) · debris_norm

    Parameters
    ----------
    dose_norm : NDArray[np.float64]
        Normalized radiation dose ∈ [0, 1].
    debris_norm : NDArray[np.float64]
        Normalized debris flux ∈ [0, 1].
    alpha : float
        Radiation weighting parameter ∈ [0, 1].
        α=0 → debris-only; α=1 → radiation-only.

    Returns
    -------
    NDArray[np.float64]
        Compound hazard index ∈ [0, 1] at each altitude.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    return alpha * dose_norm + (1.0 - alpha) * debris_norm


def pareto_front(
    dose_vals: NDArray[np.float64],
    debris_vals: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """
    Identify Pareto-optimal (non-dominated) solutions.

    A point i is Pareto-optimal if no other point j has both
    dose_j ≤ dose_i AND debris_j ≤ debris_i with at least one strict inequality.

    Parameters
    ----------
    dose_vals : NDArray[np.float64]
        Radiation dose at each altitude (normalized or raw).
    debris_vals : NDArray[np.float64]
        Debris flux at each altitude (normalized or raw).

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask — True for Pareto-optimal altitudes.
    """
    n = len(dose_vals)
    on_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if on_pareto[i]:
            dominated = (
                (dose_vals <= dose_vals[i])
                & (debris_vals <= debris_vals[i])
                & ((dose_vals < dose_vals[i]) | (debris_vals < debris_vals[i]))
            )
            dominated[i] = False
            if dominated.any():
                on_pareto[i] = False
    return on_pareto


def optimal_altitude(
    altitudes_km: NDArray[np.float64],
    dose_norm: NDArray[np.float64],
    debris_norm: NDArray[np.float64],
    alpha: float = 0.5,
) -> float:
    """
    Find the altitude that minimizes the compound hazard index H(α).

    Parameters
    ----------
    altitudes_km : NDArray[np.float64]
        Altitude grid in km.
    dose_norm : NDArray[np.float64]
        Normalized radiation dose at each altitude.
    debris_norm : NDArray[np.float64]
        Normalized debris flux at each altitude.
    alpha : float
        Radiation weighting parameter ∈ [0, 1].

    Returns
    -------
    float
        Altitude (km) that minimizes H(α).
    """
    h = compound_hazard_index(dose_norm, debris_norm, alpha)
    return float(altitudes_km[np.argmin(h)])


def sensitivity_sweep(
    altitudes_km: NDArray[np.float64],
    dose_norm: NDArray[np.float64],
    debris_norm: NDArray[np.float64],
    n_alpha: int = 101,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Sweep α from 0 to 1 and record the optimal altitude at each value.

    This is the core robustness check: if the optimal altitude is stable
    across all α values, the main finding is independent of weighting choice.

    Parameters
    ----------
    altitudes_km : NDArray[np.float64]
        Altitude grid in km.
    dose_norm : NDArray[np.float64]
        Normalized radiation dose at each altitude.
    debris_norm : NDArray[np.float64]
        Normalized debris flux at each altitude.
    n_alpha : int
        Number of α values to evaluate (evenly spaced in [0, 1]).

    Returns
    -------
    tuple of:
        alphas : NDArray[np.float64]  — α values from 0 to 1
        opt_alts : NDArray[np.float64] — optimal altitude at each α (km)
    """
    alphas = np.linspace(0.0, 1.0, n_alpha)
    opt_alts = np.array(
        [optimal_altitude(altitudes_km, dose_norm, debris_norm, float(a)) for a in alphas]
    )
    return alphas, opt_alts
