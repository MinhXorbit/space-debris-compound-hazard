"""Unit tests for the RadiationModel."""

import numpy as np
import pytest
from compound_hazard.models.radiation import RadiationModel, ISS_ALTITUDE_KM


@pytest.fixture
def model() -> RadiationModel:
    return RadiationModel(d_iss_ref=207.0)


def test_anchor_point(model: RadiationModel) -> None:
    """Model must exactly reproduce its own anchor dose at ISS altitude."""
    d = model.verify_anchor()
    assert abs(d - model.d_iss_ref) < 1e-6, (
        f"Anchor verification failed: got {d:.4f}, expected {model.d_iss_ref:.4f}"
    )


def test_dose_increases_with_altitude(model: RadiationModel) -> None:
    """Dose should generally increase from 450 km to 1000 km (belt onset)."""
    d_low  = model.dose_rate(450.0)
    d_high = model.dose_rate(1000.0)
    assert d_high > d_low, "Expected dose to increase with altitude above belt onset"


def test_solar_min_higher_than_max(model: RadiationModel) -> None:
    """GCR dose at solar minimum should be higher than at solar maximum."""
    alts = np.array([370.0, 408.0, 500.0, 700.0])
    d_max = model.dose_rate_solar_max(alts)
    d_min = model.dose_rate_solar_min(alts)
    assert np.all(d_min >= d_max), "Solar-min dose should be >= solar-max dose at all altitudes"


def test_output_shape(model: RadiationModel) -> None:
    """Scalar and array inputs should return matching shapes."""
    alts = np.linspace(370, 1200, 100)
    d = model.dose_rate(alts)
    assert d.shape == (100,)

    d_scalar = model.dose_rate(408.0)
    assert d_scalar.shape == ()


def test_relative_dose_at_iss(model: RadiationModel) -> None:
    """Relative dose at ISS altitude should be 1.0."""
    rel = float(model.relative_dose(ISS_ALTITUDE_KM))
    assert abs(rel - 1.0) < 1e-6


def test_physical_bounds(model: RadiationModel) -> None:
    """Dose should remain positive and finite over the full altitude range."""
    alts = np.linspace(370, 1200, 500)
    d = model.dose_rate(alts)
    assert np.all(np.isfinite(d)), "Non-finite dose detected"
    assert np.all(d > 0), "Non-positive dose detected"
