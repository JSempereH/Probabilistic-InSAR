# Probabilistic InSAR

A learning implementation of Chang & Hanssen (2016), *"A Probabilistic
Approach for InSAR Time-Series Postprocessing"* (IEEE TGRS 54(1)): multiple
hypotheses testing (MHT) and Baarda's B-method of testing applied to InSAR
kinematic time series, for probabilistic model selection and unwrapping
error detection.

Companion blog series: [InSAR from Scratch](https://jsempereh.github.io/notes/)
(parts I-IV cover the theory this code implements).

## Structure

- `src/insar_mht/`: the library
  - `library.py`: canonical kinematic functions M1-M6 (Eq. 14)
  - `hypotheses.py`: Table I hypothesis combinations, M5/M6 per-epoch families
  - `testing.py`: GLS fit, OMT, L-matrices, B-method calibration, test ratios
  - `dia.py`: Detection-Identification-Adaptation loop (Fig. 2)
  - `rpn.py`: reference point noise estimation ("Shenzhen algorithm", Sec. III-B)
  - `unwrap.py`: residual unwrapping error detection/correction (Sec. III-C)
  - `reliability.py`: parameter precision (Qxx) and reliability (Sec. II-C.3)
  - `io/egms.py`: reader for EGMS CSV point products
  - `io/temperature.py`: daily temperature record for the M2 thermal column
- `notebooks/`: step-by-step walkthroughs, from a single point's functional
  model to a full AOI case study, mirroring the blog series
- `tests/`: pytest sanity checks for the core numerics
- `data/`: local EGMS downloads (gitignored)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Getting an EGMS AOI

See the conversation / project notes for the full walkthrough. Short version:

1. Create a free account at the [Copernicus Land Monitoring
   Service](https://land.copernicus.eu/).
2. Open the [EGMS Explorer](https://egms.land.copernicus.eu/) viewer, draw
   or search for your AOI, and check which 100x100 km tile(s) and orbit
   direction(s) (ascending/descending) cover it.
3. Download the **Calibrated (L2b)** product for that tile (LOS series
   referenced to a GNSS-consistent datum) as CSV, or the **Ortho (L3)**
   product if you want pre-decomposed vertical/east-west motion.
4. Drop the CSV under `data/` and load it with `insar_mht.io.egms.load_egms_csv`.
