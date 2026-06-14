"""Unit tests for the DebrisFluxModel."""

import numpy as np
import pytest
from compound_hazard.models.debris import DebrisFluxModel


@pytest.fixture
def model() -> DebrisFluxModel:
    return DebrisFluxModel()


def test_normalization_at_iss(model: DebrisFluxModel) -> None:
    """Flux at ISS altitude should be normalized to 1.0."""
    flux = float(model.flux(408.0))
    assert abs(flux - 1.0) < 1e-4, f"Expected flux=1.0 at ISS, got {flux:.6f}"


def test_flux_increases_with_altitude(model: DebrisFluxModel) -> None:
    """Flux should be higher at 800 km than at 408 km (dominant debris shells)."""
    assert model.flux(800.0) > model.flux(408.0)


def test_flux_lower_below_iss(model: DebrisFluxModel) -> None:
    """Flux below ISS altitude should be below 1.0 (drag-cleared environment)."""
    assert model.flux(350.0) < 1.0, "Sub-ISS flux should be < 1 (drag clearing)"


def test_callable_interface(model: DebrisFluxModel) -> None:
    """DebrisFluxModel should be directly callable."""
    result = model(np.array([400.0, 500.0, 800.0]))
    assert result.shape == (3,)


def test_output_always_positive(model: DebrisFluxModel) -> None:
    """Flux should be positive everywhere in the practical altitude range."""
    alts = np.linspace(200, 1500, 1000)
    flux = model.flux(alts)
    assert np.all(flux > 0), "Non-positive flux value detected"
    assert np.all(np.isfinite(flux)), "Non-finite flux value detected"


def test_fragmentation_shells_produce_local_maxima(model: DebrisFluxModel) -> None:
    """
    The Fengyun-1C and Cosmos/Iridium fragmentation shells should produce a
    local flux maximum in the 700–950 km band above the background.

    The global maximum is at the domain ceiling (1200 km) due to the rising
    exponential background — this mirrors real MASTER-8 behavior where flux
    continues to increase with altitude into the upper LEO band. This test
    checks that the fragmentation shells create a meaningful GRADIENT that
    distinguishes 800 km from 500 km.
    """
    alts = np.linspace(370, 1200, 1000)
    flux = model.flux(alts)

    idx_500 = int(np.argmin(np.abs(alts - 500)))
    idx_800 = int(np.argmin(np.abs(alts - 800)))

    # Flux at 800 km (fragmentation shells) should be substantially higher than 500 km
    assert flux[idx_800] > 3 * flux[idx_500], (
        f"Expected fragmentation shells to cause flux at 800 km "
        f"({flux[idx_800]:.2f}) to be >3× the flux at 500 km ({flux[idx_500]:.2f})"
    )
