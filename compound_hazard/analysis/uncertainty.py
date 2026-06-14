"""
Uncertainty quantification for the compound hazard analysis.

Two complementary approaches:

1. **Bootstrap CI on Pareto frontier** (bootstrap_pareto_ci):
   Resamples the RadLab monthly dose rate measurements with replacement
   to propagate measurement uncertainty through the radiation anchor.
   Produces 95% confidence bands on normalized dose and Pareto-optimal
   altitude range.

2. **Monte Carlo compound hazard** (monte_carlo_compound_hazard):
   Perturbs both models simultaneously — radiation anchor by its
   measured standard deviation, debris parameters by ±10% calibration
   uncertainty — to produce a full posterior distribution over the
   compound hazard index at each altitude.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bootstrap_pareto_ci(
    monthly_dose_means: NDArray[np.float64],
    altitudes_km: NDArray[np.float64],
    radiation_model_fn,
    debris_model_fn,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, NDArray[np.float64]]:
    """
    Bootstrap confidence intervals on Pareto frontier from RadLab measurement uncertainty.

    Strategy: resample the RadLab monthly dose means (each month's mean is one
    observation) with replacement to get a bootstrapped radiation anchor. Recompute
    normalized dose and record the resulting Pareto-optimal altitude range.

    Parameters
    ----------
    monthly_dose_means : NDArray[np.float64]
        Array of monthly mean dose rates from RadLab (µGy/day).
        Each entry is one month's aggregate — resampling simulates measurement uncertainty.
    altitudes_km : NDArray[np.float64]
        Altitude grid (km).
    radiation_model_fn : callable
        Function(d_iss_ref, alt_km) → dose array. Typically a closure over RadiationModel.
    debris_model_fn : callable
        Function(alt_km) → debris flux array.
    n_bootstrap : int
        Number of bootstrap iterations.
    ci_level : float
        Confidence level (default 0.95 → 95% CI).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'dose_norm_lo'  : lower CI bound on normalized dose at each altitude
        'dose_norm_hi'  : upper CI bound on normalized dose at each altitude
        'dose_norm_med' : median normalized dose at each altitude
        'pareto_alt_min': array of Pareto min altitudes across bootstrap iterations
        'pareto_alt_max': array of Pareto max altitudes across bootstrap iterations
        'opt_alt_equal' : optimal altitude at α=0.5 for each bootstrap iteration
        'ci_lo_pct'     : lower percentile used (e.g., 2.5 for 95% CI)
        'ci_hi_pct'     : upper percentile used (e.g., 97.5 for 95% CI)
    """
    rng = np.random.default_rng(random_seed)
    alpha_tail = (1.0 - ci_level) / 2.0
    lo_pct = 100.0 * alpha_tail
    hi_pct = 100.0 * (1.0 - alpha_tail)

    n_obs = len(monthly_dose_means)
    n_alts = len(altitudes_km)

    dose_norm_samples: list[NDArray[np.float64]] = []
    pareto_min_samples: list[float] = []
    pareto_max_samples: list[float] = []
    opt_alt_samples: list[float] = []

    debris_vals = np.asarray(debris_model_fn(altitudes_km), dtype=np.float64)

    for _ in range(n_bootstrap):
        # Resample monthly means with replacement
        boot_anchor = float(rng.choice(monthly_dose_means, size=n_obs, replace=True).mean())

        # Compute dose profile with this anchor
        dose_vals = np.asarray(radiation_model_fn(boot_anchor, altitudes_km), dtype=np.float64)

        # Normalize over the domain
        dose_norm = (dose_vals - dose_vals.min()) / (dose_vals.max() - dose_vals.min())
        debris_norm = (debris_vals - debris_vals.min()) / (debris_vals.max() - debris_vals.min())

        dose_norm_samples.append(dose_norm)

        # Pareto front
        from compound_hazard.analysis.pareto import pareto_front, compound_hazard_index
        pareto_mask = pareto_front(dose_norm, debris_norm)
        pareto_alts = altitudes_km[pareto_mask]
        pareto_min_samples.append(float(pareto_alts.min()))
        pareto_max_samples.append(float(pareto_alts.max()))

        # Optimal altitude at equal weighting
        h_equal = compound_hazard_index(dose_norm, debris_norm, 0.5)
        opt_alt_samples.append(float(altitudes_km[np.argmin(h_equal)]))

    dose_norm_arr = np.stack(dose_norm_samples, axis=0)  # (n_bootstrap, n_alts)

    return {
        "dose_norm_lo": np.percentile(dose_norm_arr, lo_pct, axis=0),
        "dose_norm_hi": np.percentile(dose_norm_arr, hi_pct, axis=0),
        "dose_norm_med": np.median(dose_norm_arr, axis=0),
        "pareto_alt_min": np.array(pareto_min_samples),
        "pareto_alt_max": np.array(pareto_max_samples),
        "opt_alt_equal": np.array(opt_alt_samples),
        "ci_lo_pct": lo_pct,
        "ci_hi_pct": hi_pct,
    }


def monte_carlo_compound_hazard(
    altitudes_km: NDArray[np.float64],
    d_iss_ref: float,
    d_iss_std: float,
    radiation_model_fn,
    debris_model_fn,
    debris_uncertainty_frac: float = 0.10,
    n_samples: int = 5000,
    random_seed: int = 42,
) -> dict[str, NDArray[np.float64]]:
    """
    Monte Carlo uncertainty propagation for the compound hazard index.

    Perturbs both models:
    - Radiation: anchor d_iss_ref sampled from N(d_iss_ref, d_iss_std²)
    - Debris: overall flux scale sampled from N(1.0, debris_uncertainty_frac²)

    Parameters
    ----------
    altitudes_km : NDArray[np.float64]
        Altitude grid (km).
    d_iss_ref : float
        Best-estimate ISS dose rate anchor (µGy/day).
    d_iss_std : float
        Standard deviation of ISS dose rate measurement (µGy/day).
    radiation_model_fn : callable
        Function(d_iss_ref, alt_km) → dose array.
    debris_model_fn : callable
        Function(alt_km) → debris flux array.
    debris_uncertainty_frac : float
        Fractional uncertainty on debris flux calibration (default 10%).
    n_samples : int
        Number of Monte Carlo samples.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'hazard_lo'   : lower 2.5th percentile of compound hazard at each altitude
        'hazard_hi'   : upper 97.5th percentile
        'hazard_med'  : median compound hazard
        'opt_alt_dist': distribution of optimal altitudes (at α=0.5) across samples
    """
    rng = np.random.default_rng(random_seed)

    hazard_samples: list[NDArray[np.float64]] = []
    opt_alt_dist: list[float] = []

    # Baseline debris (unperturbed shape)
    debris_base = np.asarray(debris_model_fn(altitudes_km), dtype=np.float64)

    from compound_hazard.analysis.pareto import (
        normalize_hazards,
        compound_hazard_index,
    )

    for _ in range(n_samples):
        # Perturb radiation anchor
        d_anchor = float(rng.normal(d_iss_ref, d_iss_std))
        d_anchor = max(d_anchor, d_iss_ref * 0.5)  # physical lower bound

        dose_vals = np.asarray(radiation_model_fn(d_anchor, altitudes_km), dtype=np.float64)

        # Perturb debris flux scale (log-normal to keep flux positive)
        debris_scale = float(rng.lognormal(mean=0.0, sigma=debris_uncertainty_frac))
        debris_vals = debris_base * debris_scale

        dose_norm, debris_norm = normalize_hazards(dose_vals, debris_vals)
        h = compound_hazard_index(dose_norm, debris_norm, alpha=0.5)

        hazard_samples.append(h)
        opt_alt_dist.append(float(altitudes_km[np.argmin(h)]))

    hazard_arr = np.stack(hazard_samples, axis=0)

    return {
        "hazard_lo": np.percentile(hazard_arr, 2.5, axis=0),
        "hazard_hi": np.percentile(hazard_arr, 97.5, axis=0),
        "hazard_med": np.median(hazard_arr, axis=0),
        "opt_alt_dist": np.array(opt_alt_dist),
    }
