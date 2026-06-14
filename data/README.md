# Data

This directory holds the NASA OSDR RadLab ISS dosimetry CSV files.
These are **not committed to the repository** (see `.gitignore`) due to
their size (~150 MB combined for the full 2009–2015 dataset).

## Downloading the data

The RadLab API is publicly accessible with no authentication required:

```
https://visualization.osdr.nasa.gov/radlab/api/
```

### DosTel1 (primary instrument)

```bash
curl "https://visualization.osdr.nasa.gov/radlab/api/data/?id=dostel1&format=csv" \
     -o data/dostel1_data.csv
```

### DosTel2 (cross-validation instrument)

```bash
curl "https://visualization.osdr.nasa.gov/radlab/api/data/?id=dostel2&format=csv" \
     -o data/dostel2_data.csv
```

Both files have the format:
```
timestamp,instrument_id,dose_rate
"2009-07-01T00:00:00Z","dostel1",207.4
...
```

## Instruments

| Instrument | Full name | Location | Date range |
|------------|-----------|----------|------------|
| DosTel1    | DOSTEL Scintillation Telescope 1 | Columbus Module, ISS | 2009–2015 |
| DosTel2    | DOSTEL Scintillation Telescope 2 | Columbus Module, ISS | 2009–2015 |

DosTel instruments measure absorbed dose rate (µGy/day) from ionizing
radiation encountered in the ISS orbital environment (408 km, 51.6°
inclination). The dataset spans Solar Cycle 24 from solar minimum (2009)
through solar maximum (April 2014).

## Citation

If you use this data, please cite:

> Berger, T., et al. (2013). *DOSTEL measurements onboard the
> Columbus Laboratory of the ISS in the years 2009–2011.*
> Radiation Measurements, 51–52, 49–56.

> NASA Open Science Data Repository (OSDR).
> *RadLab ISS Dosimetry Database.*
> https://visualization.osdr.nasa.gov/radlab/
