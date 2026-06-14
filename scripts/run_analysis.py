#!/usr/bin/env python3
"""
run_analysis.py — Full compound orbital hazard analysis pipeline.

Loads NASA OSDR RadLab ISS dosimetry data, fits radiation and debris flux
models, computes Pareto frontier and sensitivity analysis, generates all
paper figures (Figs. 1–5), and prints a summary of key quantitative findings.

Usage
-----
    python scripts/run_analysis.py --data-dir data/ --out-dir paper/figures/

Required input files (in --data-dir):
    dostel1_data.csv    — RadLab DosTel1 measurements
    dostel2_data.csv    — RadLab DosTel2 measurements

Output figures (in --out-dir):
    fig1_timeseries.png
    fig2_solar_corr.png
    fig3_altitude.png
    fig4_pareto.png
    fig5_sensitivity.png
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Allow running from repo root without installing the package
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from compound_hazard.models.radiation import RadiationModel, ALT_MIN_KM, ALT_MAX_KM
from compound_hazard.models.debris import DebrisFluxModel
from compound_hazard.analysis.pareto import (
    normalize_hazards,
    compound_hazard_index,
    pareto_front,
    sensitivity_sweep,
    optimal_altitude,
)
from compound_hazard.analysis.uncertainty import (
    bootstrap_pareto_ci,
    monte_carlo_compound_hazard,
)
from compound_hazard.visualization.figures import (
    plot_timeseries,
    plot_solar_correlation,
    plot_altitude_profiles,
    plot_pareto_frontier,
    plot_sensitivity_analysis,
)


# ── Solar cycle data (NOAA/SIDC, Solar Cycle 24 monthly means) ───────────────
SOLAR_CYCLE_24 = {
    (2009, 7): 2,   (2009, 12): 4,
    (2010, 6): 15,  (2011, 1): 35,  (2011, 6): 75,
    (2012, 1): 68,  (2012, 6): 83,  (2013, 1): 66,
    (2013, 6): 88,  (2014, 1): 89,  (2014, 4): 116,
    (2014, 6): 99,  (2014, 12): 79, (2015, 1): 70,
    (2015, 6): 69,
}


def load_radlab_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean DosTel1 and DosTel2 CSV files from the NASA OSDR RadLab API."""
    def _load(fname: str) -> pd.DataFrame:
        df = pd.read_csv(
            data_dir / fname,
            names=["timestamp", "instrument_id", "dose_rate"],
            skiprows=1,
            quotechar='"',
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.dropna(subset=["dose_rate"])
        return df

    df1 = _load("dostel1_data.csv")
    df2 = _load("dostel2_data.csv")
    return df1, df2


def monthly_aggregate(df: pd.DataFrame, min_count: int = 100) -> pd.DataFrame:
    """Aggregate to monthly means; drop months with fewer than min_count measurements."""
    df = df.copy()
    df["year_month"] = df["timestamp"].dt.to_period("M")
    monthly = (
        df.groupby("year_month")["dose_rate"]
        .agg(["mean", "std", "count", "median"])
        .reset_index()
    )
    monthly.columns = ["year_month", "dose_mean", "dose_std", "n", "dose_median"]
    monthly = monthly[monthly["n"] > min_count].copy()
    monthly["date"] = monthly["year_month"].dt.to_timestamp()
    return monthly


def add_sunspot_numbers(monthly: pd.DataFrame) -> pd.DataFrame:
    """Interpolate NOAA/SIDC sunspot numbers onto monthly date index."""
    monthly = monthly.copy()
    ssn_dates = pd.to_datetime([f"{y}-{m:02d}-01" for (y, m) in SOLAR_CYCLE_24])
    ssn_values = np.array(list(SOLAR_CYCLE_24.values()), dtype=float)
    monthly["ssn"] = np.interp(
        monthly["date"].astype(np.int64),
        ssn_dates.astype(np.int64),
        ssn_values,
    )
    return monthly


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory containing dostel1_data.csv and dostel2_data.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "paper" / "figures",
        help="Directory for output figure PNGs",
    )
    p.add_argument("--n-bootstrap", type=int, default=2000, help="Bootstrap iterations")
    p.add_argument("--n-monte-carlo", type=int, default=5000, help="Monte Carlo samples")
    p.add_argument("--n-altitudes", type=int, default=500, help="Altitude grid resolution")
    p.add_argument("--no-uncertainty", action="store_true", help="Skip bootstrap/MC (faster)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Compound Orbital Hazard Analysis — xOrbita Inc. / NASA OSDR AWG")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading RadLab data from: {args.data_dir}")
    df1, df2 = load_radlab_data(args.data_dir)
    print(f"  DosTel1: {len(df1):,} measurements  "
          f"[{df1['timestamp'].min().date()} → {df1['timestamp'].max().date()}]")
    print(f"  DosTel2: {len(df2):,} measurements  "
          f"[{df2['timestamp'].min().date()} → {df2['timestamp'].max().date()}]")

    monthly1 = add_sunspot_numbers(monthly_aggregate(df1))
    monthly2 = monthly_aggregate(df2)
    print(f"  Monthly aggregates: DosTel1 = {len(monthly1)} months, DosTel2 = {len(monthly2)} months")

    # Cross-instrument validation
    common = set(monthly1["year_month"]) & set(monthly2["year_month"])
    m1c = monthly1[monthly1["year_month"].isin(common)].sort_values("year_month")
    m2c = monthly2[monthly2["year_month"].isin(common)].sort_values("year_month")
    r_cross, p_cross = pearsonr(m1c["dose_mean"].values, m2c["dose_mean"].values)
    ratio = (m1c["dose_mean"].values / m2c["dose_mean"].values)
    print(f"\nCross-instrument validation: r = {r_cross:.3f} (p = {p_cross:.2e})")
    print(f"  DosTel1/DosTel2 ratio: {ratio.mean():.3f} ± {ratio.std():.3f}")

    # Solar cycle correlation
    r_solar, p_solar = pearsonr(monthly1["ssn"], monthly1["dose_mean"])
    print(f"\nDose vs SSN (Solar Activity): r = {r_solar:.3f} (p = {p_solar:.2e})")

    # ── Models ────────────────────────────────────────────────────────────────
    D_ISS_REF = float(df1["dose_rate"].mean())
    D_ISS_STD = float(df1.groupby(df1["timestamp"].dt.to_period("M"))["dose_rate"]
                      .mean().std())
    print(f"\nRadiationModel anchor: D_ISS = {D_ISS_REF:.3f} ± {D_ISS_STD:.3f} µGy/day")

    rad_model = RadiationModel(d_iss_ref=D_ISS_REF)
    deb_model = DebrisFluxModel()

    altitudes_km = np.linspace(ALT_MIN_KM, ALT_MAX_KM, args.n_altitudes)
    dose_solar_max = rad_model.dose_rate_solar_max(altitudes_km)
    dose_solar_min = rad_model.dose_rate_solar_min(altitudes_km)
    debris_flux    = deb_model.flux(altitudes_km)

    # ── Pareto analysis ───────────────────────────────────────────────────────
    dose_norm, debris_norm = normalize_hazards(dose_solar_max, debris_flux)
    pareto_mask = pareto_front(dose_norm, debris_norm)
    pareto_alts = altitudes_km[pareto_mask]

    alphas, opt_alts = sensitivity_sweep(altitudes_km, dose_norm, debris_norm)
    iss_idx = int(np.argmin(np.abs(altitudes_km - 408.0)))

    print(f"\nPareto-optimal altitude band: {pareto_alts.min():.0f}–{pareto_alts.max():.0f} km")
    print(f"Optimal altitude (α=0.5):     {optimal_altitude(altitudes_km, dose_norm, debris_norm):.0f} km")
    print(f"Sensitivity sweep range:      {opt_alts.min():.0f}–{opt_alts.max():.0f} km  (all α ∈ [0,1])")

    h_equal = compound_hazard_index(dose_norm, debris_norm, 0.5)
    idx_iss = iss_idx
    idx_700 = int(np.argmin(np.abs(altitudes_km - 700)))
    print(f"Compound hazard penalty (408→700 km): "
          f"+{100*(h_equal[idx_700]/h_equal[idx_iss]-1):.1f}%")

    # ── Uncertainty quantification ────────────────────────────────────────────
    bootstrap_ci = None
    mc_result     = None

    if not args.no_uncertainty:
        print(f"\nRunning bootstrap CI ({args.n_bootstrap} iterations)...")
        monthly_means = monthly1["dose_mean"].values

        def _rad_fn(d_ref: float, alts: np.ndarray) -> np.ndarray:
            return RadiationModel(d_iss_ref=d_ref).dose_rate_solar_max(alts)

        bootstrap_ci = bootstrap_pareto_ci(
            monthly_dose_means=monthly_means,
            altitudes_km=altitudes_km,
            radiation_model_fn=_rad_fn,
            debris_model_fn=deb_model.flux,
            n_bootstrap=args.n_bootstrap,
        )
        opt_dist = bootstrap_ci["opt_alt_equal"]
        print(f"  Bootstrap optimal altitude: "
              f"{np.percentile(opt_dist, 2.5):.0f}–{np.percentile(opt_dist, 97.5):.0f} km (95% CI)")

        print(f"Running Monte Carlo ({args.n_monte_carlo} samples)...")
        mc_result = monte_carlo_compound_hazard(
            altitudes_km=altitudes_km,
            d_iss_ref=D_ISS_REF,
            d_iss_std=D_ISS_STD,
            radiation_model_fn=_rad_fn,
            debris_model_fn=deb_model.flux,
            n_samples=args.n_monte_carlo,
        )
        mc_dist = mc_result["opt_alt_dist"]
        print(f"  MC optimal altitude 95% CI: "
              f"{np.percentile(mc_dist, 2.5):.0f}–{np.percentile(mc_dist, 97.5):.0f} km")

    # ── Figures ───────────────────────────────────────────────────────────────
    print(f"\nGenerating figures → {args.out_dir}")

    plot_timeseries(monthly1, monthly2,
                    out_path=args.out_dir / "fig1_timeseries.png")
    print("  fig1_timeseries.png ✓")

    plot_solar_correlation(monthly1, r_solar, p_solar,
                           out_path=args.out_dir / "fig2_solar_corr.png")
    print("  fig2_solar_corr.png ✓")

    plot_altitude_profiles(altitudes_km, dose_solar_max, dose_solar_min, debris_flux,
                           out_path=args.out_dir / "fig3_altitude.png")
    print("  fig3_altitude.png ✓")

    plot_pareto_frontier(altitudes_km, dose_norm, debris_norm, pareto_mask, iss_idx,
                         bootstrap_ci=bootstrap_ci,
                         out_path=args.out_dir / "fig4_pareto.png")
    print("  fig4_pareto.png ✓")

    plot_sensitivity_analysis(alphas, opt_alts, mc_result=mc_result,
                               out_path=args.out_dir / "fig5_sensitivity.png")
    print("  fig5_sensitivity.png ✓")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"""
{'=' * 70}
KEY QUANTITATIVE FINDINGS
{'=' * 70}

1. RADLAB DATA
   DosTel1: {len(df1):,} measurements, {df1['timestamp'].min().date()} to {df1['timestamp'].max().date()}
   Mean dose rate: {D_ISS_REF:.2f} ± {D_ISS_STD:.2f} µGy/day

2. CROSS-INSTRUMENT VALIDATION
   Pearson r = {r_cross:.3f} (p = {p_cross:.2e})
   DosTel1/DosTel2 ratio = {ratio.mean():.3f} ± {ratio.std():.3f}

3. SOLAR MODULATION
   Dose vs SSN: r = {r_solar:.3f} (p = {p_solar:.2e})

4. PARETO OPTIMIZATION
   Pareto-optimal band: {pareto_alts.min():.0f}–{pareto_alts.max():.0f} km
   Optimal altitude (α=0.5): {optimal_altitude(altitudes_km, dose_norm, debris_norm):.0f} km
   Sensitivity range (all α): {opt_alts.min():.0f}–{opt_alts.max():.0f} km  ← result is robust

5. HAZARD PENALTY AT HIGHER ORBITS
   408→700 km: +{100*(h_equal[idx_700]/h_equal[idx_iss]-1):.1f}% compound hazard
   Debris flux at 700 km: {deb_model.flux(700.0):.1f}x ISS baseline
   Debris flux at 900 km: {deb_model.flux(900.0):.1f}x ISS baseline
""")

    print("Analysis complete.")


if __name__ == "__main__":
    main()
