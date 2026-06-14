"""Unit tests for Pareto frontier and compound hazard analysis functions."""

import numpy as np
import pytest
from compound_hazard.analysis.pareto import (
    normalize_hazards,
    compound_hazard_index,
    pareto_front,
    optimal_altitude,
    sensitivity_sweep,
)
from compound_hazard.models.radiation import RadiationModel, ALT_MIN_KM, ALT_MAX_KM
from compound_hazard.models.debris import DebrisFluxModel


@pytest.fixture
def altitude_grid() -> np.ndarray:
    return np.linspace(ALT_MIN_KM, ALT_MAX_KM, 200)


@pytest.fixture
def normalized_hazards(altitude_grid):
    rad = RadiationModel(d_iss_ref=207.0)
    deb = DebrisFluxModel()
    dose   = rad.dose_rate_solar_max(altitude_grid)
    debris = deb.flux(altitude_grid)
    return normalize_hazards(dose, debris)


def test_normalize_bounds(normalized_hazards) -> None:
    """Both normalized arrays should lie in [0, 1]."""
    dose_n, debris_n = normalized_hazards
    assert dose_n.min() >= 0.0 and dose_n.max() <= 1.0
    assert debris_n.min() >= 0.0 and debris_n.max() <= 1.0


def test_normalize_min_is_zero(normalized_hazards) -> None:
    """At least one value in each normalized array should equal 0."""
    dose_n, debris_n = normalized_hazards
    assert np.isclose(dose_n.min(), 0.0, atol=1e-10)
    assert np.isclose(debris_n.min(), 0.0, atol=1e-10)


def test_normalize_max_is_one(normalized_hazards) -> None:
    """At least one value in each normalized array should equal 1."""
    dose_n, debris_n = normalized_hazards
    assert np.isclose(dose_n.max(), 1.0, atol=1e-10)
    assert np.isclose(debris_n.max(), 1.0, atol=1e-10)


def test_compound_hazard_alpha_limits(normalized_hazards) -> None:
    """At α=0, H should equal debris_norm; at α=1, H should equal dose_norm."""
    dose_n, debris_n = normalized_hazards
    h_debris_only = compound_hazard_index(dose_n, debris_n, alpha=0.0)
    h_dose_only   = compound_hazard_index(dose_n, debris_n, alpha=1.0)
    np.testing.assert_allclose(h_debris_only, debris_n)
    np.testing.assert_allclose(h_dose_only, dose_n)


def test_compound_hazard_invalid_alpha(normalized_hazards) -> None:
    """Alpha outside [0, 1] should raise ValueError."""
    dose_n, debris_n = normalized_hazards
    with pytest.raises(ValueError):
        compound_hazard_index(dose_n, debris_n, alpha=1.5)
    with pytest.raises(ValueError):
        compound_hazard_index(dose_n, debris_n, alpha=-0.1)


def test_pareto_front_nonempty(altitude_grid, normalized_hazards) -> None:
    """Pareto front should contain at least one point."""
    dose_n, debris_n = normalized_hazards
    pareto = pareto_front(dose_n, debris_n)
    assert pareto.sum() > 0, "Pareto front is empty"


def test_pareto_front_in_low_altitude_band(altitude_grid, normalized_hazards) -> None:
    """Pareto-optimal altitudes should be in the low-altitude (sub-500 km) region."""
    dose_n, debris_n = normalized_hazards
    pareto = pareto_front(dose_n, debris_n)
    pareto_alts = altitude_grid[pareto]
    assert pareto_alts.max() <= 500.0, (
        f"Pareto front extends to {pareto_alts.max():.0f} km, expected ≤ 500 km"
    )


def test_optimal_altitude_in_iss_band(altitude_grid, normalized_hazards) -> None:
    """For all α values, optimal altitude should be in the ISS operational band."""
    dose_n, debris_n = normalized_hazards
    alphas, opt_alts = sensitivity_sweep(altitude_grid, dose_n, debris_n, n_alpha=21)
    for a, alt in zip(alphas, opt_alts):
        assert 370 <= alt <= 500, (
            f"Optimal altitude {alt:.0f} km at α={a:.2f} is outside the ISS band (370–500 km)"
        )


def test_sensitivity_sweep_length(altitude_grid, normalized_hazards) -> None:
    """Sensitivity sweep should return arrays of the requested length."""
    dose_n, debris_n = normalized_hazards
    alphas, opt_alts = sensitivity_sweep(altitude_grid, dose_n, debris_n, n_alpha=51)
    assert len(alphas) == 51
    assert len(opt_alts) == 51
    assert alphas[0] == pytest.approx(0.0)
    assert alphas[-1] == pytest.approx(1.0)
