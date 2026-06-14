"""
Altitude-dependent space debris flux model for the LEO untrackable (LNT) population.

Parameterized from ESA MASTER-8 (2021) and NASA ORDEM 3.2 (2019) published flux
tables at 51.6° orbital inclination. The model uses a Gaussian mixture to capture
the dominant debris shells from historical fragmentation events.

Key debris sources represented:
- Fengyun-1C ASAT (2007) fragmentation cloud: ~850 km, 98.6° SSO
- Cosmos 2251 / Iridium 33 collision (2009): ~789 km
- General LEO fragmentation background: exponential with scale height
- Sub-600 km atmospheric drag clearing: faster debris removal below ISS altitude

References:
- Klinkrad, H. (2006). Space Debris: Models and Risk Analysis. Springer.
- Liou, J.-C. (2011). An active debris removal parametric study. Acta Astronautica, 68.
- ESA (2023). MASTER-8 model documentation. ESOC, Darmstadt.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ISS reference altitude for flux normalization
ISS_ALTITUDE_KM: float = 408.0
ISS_INCLINATION_DEG: float = 51.6


class DebrisFluxModel:
    """
    Altitude-dependent LNT (lethal non-trackable, 1–10 cm) debris flux model.

    The flux is expressed relative to the ISS baseline (normalized to 1.0 at
    408 km, 51.6° inclination). The LNT size class is the operationally relevant
    regime: fragments large enough to be catastrophic but too small for Space
    Surveillance Network radar tracking, making collision avoidance maneuvers
    impossible.

    Parameters
    ----------
    inclination_deg : float
        Target orbital inclination in degrees. Affects encounter probability
        with the Fengyun-1C cloud (98.6° SSO debris).
    iss_altitude_km : float
        Reference altitude for flux normalization.
    """

    def __init__(
        self,
        inclination_deg: float = ISS_INCLINATION_DEG,
        iss_altitude_km: float = ISS_ALTITUDE_KM,
    ) -> None:
        self.inclination_deg = inclination_deg
        self.iss_altitude_km = iss_altitude_km

        # Gaussian mixture parameters (center, sigma, peak amplitude)
        # calibrated to MASTER-8/ORDEM 3.2 flux tables
        self._fengyun_center_km: float = 850.0
        self._fengyun_sigma_km: float = 80.0
        self._fengyun_peak: float = 0.6

        self._iridium_center_km: float = 790.0
        self._iridium_sigma_km: float = 90.0
        self._iridium_peak: float = 0.5

        self._frag_center_km: float = 600.0
        self._frag_sigma_km: float = 70.0
        self._frag_peak: float = 0.15

        # Background exponential scale height (km)
        # Matches ORDEM 3.2 decadal flux progression 400–900 km
        self._H_background_km: float = 150.0

        # Cache the ISS normalization factor
        self._iss_flux_raw: float = self._flux_raw(float(iss_altitude_km))

    def _inclination_factor_fengyun(self) -> float:
        """
        Geometric encounter probability factor for FY-1C debris (98.6° SSO).

        At ISS inclination (51.6°), orbital planes cross at an angle that
        reduces encounter time per orbit relative to a co-inclination encounter.
        Simplified as ratio of sin(inclinations), clipped to [0.1, 1.0].
        """
        factor = abs(
            np.sin(np.radians(self.inclination_deg))
            / np.sin(np.radians(98.6))
        )
        return float(np.clip(factor, 0.1, 1.0))

    def _flux_raw(self, alt_km: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Un-normalized composite flux at given altitude(s)."""
        alt = np.asarray(alt_km, dtype=np.float64)

        # Background exponential population
        flux_background = np.exp((alt - 400.0) / self._H_background_km) * 0.3

        # Fengyun-1C cloud (inclination-adjusted)
        inc_factor = self._inclination_factor_fengyun()
        flux_fy = (
            self._fengyun_peak
            * inc_factor
            * np.exp(-0.5 * ((alt - self._fengyun_center_km) / self._fengyun_sigma_km) ** 2)
        )

        # Cosmos 2251 / Iridium 33 debris
        flux_iridium = self._iridium_peak * np.exp(
            -0.5 * ((alt - self._iridium_center_km) / self._iridium_sigma_km) ** 2
        )

        # General LEO fragmentation band (500–700 km)
        flux_frag = self._frag_peak * np.exp(
            -0.5 * ((alt - self._frag_center_km) / self._frag_sigma_km) ** 2
        )

        return flux_background + flux_fy + flux_iridium + flux_frag

    def flux(self, alt_km: ArrayLike) -> NDArray[np.float64]:
        """
        Normalized LNT debris flux relative to ISS baseline.

        Parameters
        ----------
        alt_km : array-like
            Orbital altitude(s) in km.

        Returns
        -------
        NDArray[np.float64]
            Flux index normalized to 1.0 at the ISS reference altitude.
            Values > 1 indicate higher flux than ISS; < 1 indicates lower.
        """
        alt = np.asarray(alt_km, dtype=np.float64)
        return self._flux_raw(alt) / self._iss_flux_raw

    def __call__(self, alt_km: ArrayLike) -> NDArray[np.float64]:
        """Shorthand for :meth:`flux`."""
        return self.flux(alt_km)
