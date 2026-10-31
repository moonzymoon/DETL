# DETL — From Detector Scores to ISA-18.2 Alarm Events

Code and archived result caches for the paper **"From Detector Scores to
ISA-18.2 Alarm Events: Evaluation-Protocol Audit, Deployable Hard Logic,
and Operating Boundaries"** (submission to *Chemical Engineering Research
and Design*).

The layer sits **after** an anomaly detector's score stream and tunes the
ISA-18.2 / EEMUA 191 alarm operators (threshold, deadband, on-delay
timer, off-delay timer) directly against event-level KPIs under explicit
alarm-rate and false-alarm-load constraints. No detector is trained or
modified here.

## What is here

- `src/` — full experiment pipeline
  - `src/experiments/a0*.py` — the corrected-protocol (protocol M)
    suite: layer training and evaluation (`a0_lib.py`), unified-domain
    search, fresh-fault cohorts (`a0u`), oracle frontier (`a0j`),
    selection-rule audit (`a0p`/`a0z`), scorer replications
    (`a0w`/`a0z`), causal rolling comparator (`a0ab`), cadence
    sensitivity (`a0ac`), threshold-margin analysis (`a0z`), bootstrap
    and fault-clustered intervals (`a0h`/`a0v`/`a0o`)
  - `src/experiments/e_r48_usad.py` — faithful self-implementation of
    USAD (Audibert et al., KDD 2020) used as the third scorer family
  - `src/tuning/` — the differentiable tuning layer (soft operators,
    constrained training) and the exact hard ISA-18.2 chain
  - `src/kpi/` — event-level KPI panel (event-F1 any-overlap and
    IoU≥0.1, AAR, FAR, TIA, stale alarms)
- `results/audit/` — the frozen cache behind every table and figure in
  the paper and its supplementary material (see the reproducibility
  manifest, supplementary Table 10)

## Reproducing

```bash
pip install -r src/requirements.txt
python src/experiments/a0_te_iforest.py      # main comparison, TE iForest
python src/experiments/a0u_newruns_ext.py    # 24-run fresh-fault extension
python src/experiments/a0ab_dynthresh_v2.py  # causal rolling comparator
python src/experiments/a0y_boundary_figure.py  # boundary figure (Fig. 4)
```

All training is deterministic (fixed seeds, full-batch, double
precision); repeated runs reproduce the cached metrics bit-for-bit.

## Data

TE mode-1 run files (following Reinartz et al., 2021) and the extended
fresh-fault cohort sources are described in `src/datasets/`; SWaT/WADI
are obtained from iTrust under data-access agreement and cannot be
redistributed; all other datasets are public (MetroPT3 via UCI, BATADAL,
NoBOOM, TSB-AD-MV).

## Archived version

This repository is archived on Zenodo; the DOI recorded in the paper's
data-availability statement points to the archived submission version.
