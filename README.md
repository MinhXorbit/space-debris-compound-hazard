# space-debris-compound-hazard

**Compound orbital hazard analysis: ionizing radiation + space debris Pareto optimization for commercial human spaceflight.**

Companion code for:
> Nguyen, M. (2026). *Space Debris Risk in an Era of Commercial Human Spaceflight: Catalog Incompleteness, Ground-Loop Latency, and the Imperative for Autonomous Onboard Detection.* Submitted to *Acta Astronautica*.

---

## Overview

This repository implements a joint optimization of ionizing radiation dose and space debris flux for crewed LEO station orbital placement (370–1,200 km). The radiation model is anchored to **empirical ISS measurements** from the NASA OSDR RadLab database (DosTel1/DosTel2 instruments, Columbus Module, 2009–2015), rather than relying solely on theoretical models. The debris term is a phenomenological relative-flux index (see note below).

**Key finding:** The ISS altitude band (370–450 km) Pareto-dominates all higher altitudes over the practical crewed-station range, and this result is robust across all hazard weighting choices (α ∈ [0, 1]).

---

## Repository structure

```
space-debris-compound-hazard/
├── compound_hazard/           # Python package
│   ├── models/
│   │   ├── radiation.py       # RadiationModel (RadLab-anchored)
│   │   └── debris.py          # DebrisFluxModel (phenomenological index; shells from ORDEM/MASTER bands)
│   ├── analysis/
│   │   ├── pareto.py          # Pareto frontier, compound hazard index, sensitivity sweep
│   │   └── uncertainty.py     # Bootstrap CI, Monte Carlo uncertainty propagation
│   └── visualization/
│       └── figures.py         # Figure generation (Figs. 1-5)
├── scripts/
│   └── run_analysis.py        # Full pipeline CLI
├── data/                      # RadLab CSVs (DosTel1/DosTel2)
├── tests/                     # pytest suite
├── requirements.txt
└── setup.py
```

---

## Quick start

```bash
git clone https://github.com/MinhXorbit/space-debris-compound-hazard.git
cd space-debris-compound-hazard
pip install -e .
pytest -q                                   # run the test suite (21 tests)
python scripts/run_analysis.py --data-dir data/ --out-dir figures/
```

The RadLab data is publicly accessible with no authentication. If the CSVs are not already in `data/`:

```bash
curl "https://visualization.osdr.nasa.gov/radlab/api/data/?id=dostel1&format=csv" -o data/dostel1_data.csv
curl "https://visualization.osdr.nasa.gov/radlab/api/data/?id=dostel2&format=csv" -o data/dostel2_data.csv
```

---

## Key quantitative results

All radiation statistics are reproducible directly from the RadLab CSVs.

| Quantity | Value |
|----------|-------|
| RadLab measurements used | 1,185,792 (DosTel1) + 1,053,184 (DosTel2) |
| Cross-instrument Pearson r | 0.957 (p = 1.1×10⁻¹⁷) |
| Dose vs solar activity (SSN), mean | r = +0.764 (p = 2.6×10⁻¹⁰) |
| GCR baseline vs SSN, median | r = −0.265 (p = 0.068; not significant at 0.05) |
| Annual dose range (Solar Cycle 24) | 6.3 mGy/yr (2010) to 10.3 mGy/yr (2013) |
| Pareto-optimal altitude band | 370–450 km |
| Optimal altitude (all α ∈ [0, 1]) | 370 km |
| Debris flux index, 408→500 km | ~2.0× |
| Debris flux index, 408→600 km | ~4.2× |
| Debris flux index, 408→700 km | ~8.3× |
| Debris flux index, 408→900 km | ~28× |
| Compound hazard penalty, 408→500 km (α = 0.5) | +93% |

---

## Models

### Radiation model (`RadiationModel`)

Altitude-dependent absorbed dose rate (µGy/day), anchored to the RadLab DosTel1 ISS mean. Three physical components:
- **GCR**: slight altitude increase from reduced Earth shadowing
- **SAA-trapped protons**: ~25% of ISS dose; increases below the inner belt threshold
- **Inner Van Allen belt**: exponential onset above ~620 km at 51.6° inclination

### Debris flux model (`DebrisFluxModel`)

Normalized LNT (lethal non-trackable, 1–10 cm) debris flux **index** (ISS = 1.0). This is a phenomenological model, **not** a fit to gridded flux tables: the Gaussian-shell *centers* match the dominant debris bands reported by ESA MASTER-8 / NASA ORDEM 3.2, but the shell amplitudes are illustrative. The relative altitude ranking is robust to those amplitudes; absolute hazard ratios should be read as order-of-magnitude estimates. Four components:
- Exponential background population
- Fengyun-1C fragmentation cloud (2007, ~850 km)
- Cosmos 2251 / Iridium 33 collision debris (2009, ~789 km)
- General LEO fragmentation band (500–700 km)

A direct fit to gridded MASTER-8/ORDEM 3.2 output is identified as future work.

### Compound hazard index

H(α) = α · D_norm + (1 − α) · F_norm, where both hazards are normalized to [0, 1] over the 370–1,200 km domain and α ∈ [0, 1] is the weighting parameter.

---

## Uncertainty quantification

- **Bootstrap CI** (`bootstrap_pareto_ci`): resamples monthly RadLab dose means (2,000 iterations) to propagate measurement uncertainty through the radiation anchor and Pareto frontier.
- **Monte Carlo** (`monte_carlo_compound_hazard`): perturbs both model anchors simultaneously (radiation Gaussian; debris log-normal ±10%) across 5,000 samples to produce a posterior over the optimal altitude.

---

## Citation

```bibtex
@article{nguyen2026debris,
  title   = {Space Debris Risk in an Era of Commercial Human Spaceflight:
             Catalog Incompleteness, Ground-Loop Latency, and the Imperative
             for Autonomous Onboard Detection},
  author  = {Nguyen, Minh},
  journal = {Acta Astronautica},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Author

**Minh Nguyen**, xOrbita Inc., Dallas, TX
NASA Open Science Data Repository (OSDR) Active Working Group Member
[Mnguyen@xorbita.com](mailto:Mnguyen@xorbita.com)

## License

MIT License. See `LICENSE` for details.
