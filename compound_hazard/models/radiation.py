"""
Altitude-dependent ionizing radiation dose rate model.

Anchored to empirical NASA OSDR RadLab DosTel1 measurements from the ISS
Columbus module (2009–2015, Solar Cycle 24). Physical scaling from literature:
- Berger et al. (2013): DOSTEL long-duration measurements on ISS.
- Reitz et al. (2005): MATROSHKA experiment orbital dosimetry.
- SPENVIS AP-8/AE-8 trapped particle database.
- CREME96 GCR attenuation model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Default ISS-altitude reference dose rate from RadLab aggregate
# (µGy/day, mean over DosTel1 measurements 2009–2015)
D_ISS_REF_DEFAULT: float = 207.0  # µGy/day — updated from live RadLab data at runtime

# ISS operational parameters
ISS_ALTITUDE_KM: float = 408.0
ISS_INCLINATION_DEG: float = 51.6

# Altitude limits for crewed commercial station operations
ALT_MIN_KM: float = 370.0   # lower: drag-induced decay threshold
ALT_MAX_KM: float = 1200.0  # upper: Kessler density + trapped proton operational limit


class RadiationModel:
    """
    Altitude-dependent ionizing radiation dose rate model anchored to RadLab data.

    Physical components modeled:
    - GCR (Galactic Cosmic Ray) background: increases slightly with altitude
      as Earth's geometric shadowing decreases.
    - SAA (South Atlantic Anomaly) trapped proton contribution: ~25% of total
      dose at ISS altitude; scales steeply with altitude below the inner belt.
    - Inner Van Allen belt onset: begins at ~620 km for 51.6° inclination,
      causes exponential rise in dose for crewed missions above ~700 km.

    Parameters
    ----------
    d_iss_ref : float
        Measured mean dose rate at ISS (408 km, 51.6°) in µGy/day.
        Defaults to the RadLab aggregate measured value.
    iss_altitude_km : float
        Reference altitude at which d_iss_ref was measured.
    iss_inclination_deg : float
        Orbital inclination of the reference measurement platform.
    """

    def __init__(
        self,
        d_iss_ref: float = D_ISS_REF_DEFAULT,
        iss_altitude_km: float = ISS_ALTITUDE_KM,
        iss_inclination_deg: float = ISS_INCLINATION_DEG,
    ) -> None:
        self.d_iss_ref = d_iss_ref
        self.iss_altitude_km = iss_altitude_km
        self.iss_inclination_deg = iss_inclination_deg

        # Inner belt onset altitude (km) at ISS inclination
        self._belt_threshold_km: float = 620.0
        # SAA fraction of total ISS dose
        self._saa_fraction: float = 0.25
        # Inner belt contribution fraction at ISS altitude (small but present)
        self._belt_fraction: float = 0.05

    def dose_rate(
        self,
        alt_km: ArrayLike,
        solar_min: bool = False,
    ) -> NDArray[np.float64]:
        """
        Compute altitude-dependent absorbed dose rate (µGy/day).

        Parameters
        ----------
        alt_km : array-like
            Orbital altitude(s) in km.
        solar_min : bool
            If True, apply solar-minimum GCR enhancement (~8% higher GCR flux
            at solar minimum per CREME96; conservative estimate).

        Returns
        -------
        NDArray[np.float64]
            Absorbed dose rate in µGy/day at each requested altitude.
        """
        alt = np.asarray(alt_km, dtype=np.float64)

        # GCR component: ~5% increase per 400 km altitude gain from
        # reduced Earth geometric shadowing (Earth subtends ~0.55 sr at 408 km,
        # less at higher altitudes). Clipped to physically motivated bounds.
        gcr_scaling = 1.0 + 0.0005 * (alt - self.iss_altitude_km)
        gcr_scaling = np.clip(gcr_scaling, 0.85, 1.10)

        # SAA-trapped proton component: positive correlation with altitude
        # up to inner belt threshold, then steep exponential rise.
        saa_scaling = np.where(
            alt < 600.0,
            1.0 + 0.003 * (alt - self.iss_altitude_km),
            1.0 + 0.003 * (600.0 - self.iss_altitude_km) + 0.015 * (alt - 600.0),
        )
        saa_scaling = np.clip(saa_scaling, 0.5, 8.0)

        # Inner Van Allen belt trapped proton contribution (begins ~620 km at 51.6°)
        belt_component = np.where(
            alt > self._belt_threshold_km,
            np.exp(0.006 * (alt - self._belt_threshold_km)) - 1.0,
            0.0,
        )

        # Combine components using measured ISS fractions as anchor
        dose_scaling = (
            (1.0 - self._saa_fraction) * gcr_scaling
            + self._saa_fraction * saa_scaling
            + self._belt_fraction * belt_component
        )

        # Solar-minimum GCR enhancement: ~8% higher at solar min (CREME96)
        if solar_min:
            dose_scaling = dose_scaling * 1.08

        return self.d_iss_ref * dose_scaling

    def dose_rate_solar_max(self, alt_km: ArrayLike) -> NDArray[np.float64]:
        """Dose rate at solar maximum conditions."""
        return self.dose_rate(alt_km, solar_min=False)

    def dose_rate_solar_min(self, alt_km: ArrayLike) -> NDArray[np.float64]:
        """Dose rate at solar minimum conditions (higher GCR)."""
        return self.dose_rate(alt_km, solar_min=True)

    def relative_dose(self, alt_km: ArrayLike, solar_min: bool = False) -> NDArray[np.float64]:
        """Dose rate relative to ISS baseline (dimensionless ratio)."""
        return self.dose_rate(alt_km, solar_min) / self.d_iss_ref

    def verify_anchor(self) -> float:
        """
        Return the model dose at the anchor altitude.

        Should equal d_iss_ref (within floating-point precision).
        """
        return float(self.dose_rate(self.iss_altitude_km))
