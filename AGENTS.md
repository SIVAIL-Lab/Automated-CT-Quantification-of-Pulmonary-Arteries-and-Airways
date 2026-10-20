# Repository guidelines

This is the script-based reproduction repository for the PARSE pulmonary-artery
and airway study and AirRC airway lumen/wall Dice and IoU validation. Keep shared
algorithms in `scripts/`, the Fiji bridge in `scripts/fiji/`, local synthetic tests
in the ignored `tests/` directory, and global algorithm parameters in `configs/`.
The supported workflows are `reproduce_parse.py` and
`prepare_airrc.py` / `run_airrc_validation_batch.py`; retain their shared stage
helpers and statistical outputs. Keep independent reference-parameter validation,
case galleries and 3D rendering outside this repository. Do not turn this project
into a general-purpose package or add HIPAS-specific workflows.

Keep clinical scans, reference labels, masks, skeletons, centerline coordinates,
sample-level geometry, native CT review panels, intermediate products, caches
and logs in the ignored `data/<dataset>/` tree. Model weights, Fiji and plugin
JARs are installed separately and must not be committed. Never publish original
image identifiers, identifying metadata, credentials or local source paths.

This repository publishes code only. Do not commit patient measurements, cohort
statistics, figures, reports, source hashes or case-specific configurations,
including de-identified results. There is no results publication exception.
Global algorithm settings may be tracked. Keep tests and their synthetic phantoms
local and ignored; do not commit or upload the `tests/` directory.
Keep user-supplied case adjustments outside the tracked source tree or
under ignored local configuration paths; they are optional external inputs,
never built-in patient data. Default airway execution uses automatic topology.
Retain quality flags and distinguish engineering inputs from expert validation.

Preserve the published metric IDs, physical units, largest-26-component scope,
coverage and quality flags. Fiji branches and EDT tree segments have different
definitions. Estimated tube volumes must not be described as mask volumes or
physiological blood volumes. Airway lumen metrics include the segmented trachea.
The wall method is an independent approximation; AirRC model-training overlap
remains unverified.

Use four spaces, snake_case and explicit input/output paths. Documentation,
commands, figures and generated presentation text use English. Run commands in
the prepared `ai` conda environment when available. Reuse meaningful synthetic
tests; they must not require clinical data, inference hardware or model weights.
Fiji tests may skip when Fiji is unavailable. Document changed commands and
dependencies in README.md. Keep commits focused and leave source repositories
unchanged. Code is licensed GPL-3.0-or-later; preserve upstream attribution and
the separate terms of datasets, models and external tools.
