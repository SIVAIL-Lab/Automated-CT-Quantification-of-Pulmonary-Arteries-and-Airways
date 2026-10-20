# Automated CT Quantification of Pulmonary Arteries and Airways

Script-based research code for pulmonary artery
and airway quantification on **PARSE**, with airway lumen/wall **Dice and IoU**
validation on **AirRC**. This repository contains analysis code and global
algorithm settings only. It contains no patient data, measurements, figures or
case-specific configurations. Obtain inputs and generate outputs locally.
The supported workflows are PARSE quantification and AirRC segmentation overlap
validation. Shared stage scripts implement these two workflows.

## Installation

The tested platform is Linux with Python 3.14.6 in a conda environment. The exact
Python dependencies used for the study implementation are recorded in
[requirements.txt](requirements.txt). Install a PyTorch build appropriate for
your CUDA runtime before installing the remaining pinned requirements. Inference
also supports CPU.

```bash
conda create -n ct-quantification python=3.14
conda activate ct-quantification
python -m pip install -r requirements.txt
```

AirRC preparation also requires the system `zip` utility to join its split
annotation archive. On Debian or Ubuntu, install it with `sudo apt install zip`.

Install [Fiji](https://imagej.net/software/fiji/) separately with
[Skeletonize3D](https://imagej.net/plugins/skeletonize3d) and
[AnalyzeSkeleton](https://imagej.net/plugins/analyze-skeleton/). The study used
Skeletonize3D 2.1.1 and AnalyzeSkeleton 3.4.2. Point to the executable explicitly:

```bash
export FIJI_BIN="/path/to/Fiji.app/ImageJ-linux64"
```

TotalSegmentator downloads its own model weights into the user cache. Weights,
Fiji, plugins and environments are not distributed in this repository.
Main-workflow statistical plots use Matplotlib's noninteractive backend.

## Data

Obtain the 100-case PARSE training release from the
[official challenge](https://parse2022.grand-challenge.org/Dataset/) and retain
the released image and artery-label pairs:

```text
data/parse/train/PA000005/image/PA000005.nii.gz
data/parse/train/PA000005/label/PA000005.nii.gz
```

AirRC supplies expert-reviewed multistructure annotations for 254 LUNA16 scans.
Obtain [the annotations](https://doi.org/10.6084/m9.figshare.26878867) and the
corresponding [LUNA16 CT parts](https://luna16.grand-challenge.org/Data/).
Keep the downloaded annotation archive `26878867.zip` and CT archives
`subset0.zip` through `subset9.zip` under `data/airrc/`. The adapter joins the
split annotation archive, maps scans to local `AIRRC000001`-style aliases, and
aligns a separate CT to the unchanged 1-mm annotation grid. The source-series
mapping remains local and must not be published.

All clinical inputs, masks, skeletons, centerline coordinates, sampled cross
sections and temporary outputs belong under the ignored `data/<dataset>/`
tree. Final measurements and figures are also generated locally and must not be
committed. Dataset access and reuse follow the original providers' terms.

## Reproduce PARSE

Run a representative case in a dedicated output directory. The pipeline creates
its own outputs and requires no historical run, active-publication file or cache
from the development repository.

```bash
python scripts/reproduce_parse.py \
  --input-root data/parse/train \
  --output-root data/parse/example \
  --case PA000005 --device gpu --fiji "$FIJI_BIN" --workers 2
```

For the complete input cohort, omit `--case` and choose a separate output root:

```bash
python scripts/reproduce_parse.py \
  --input-root data/parse/train --output-root data/parse/study \
  --device gpu --fiji "$FIJI_BIN" --workers 2
```

The stages are segmentation, selected-component metrics, airway parameters,
EDT tree/radius, orthogonal cross sections, and final statistics. `--stage`
selects one stage. Outputs are organized into `raw/`, `measurements/`, `airway/`,
`radius/`, `cross_sections/` and `statistics/` below the supplied output root.
Radius-volume summaries are written alongside the other statistics. Once the
radius and cross-section stages are complete, regenerate statistics with
`--stage statistics`, keeping the same input/output roots and case selection.
Case membership is fixed for each radius/cross-section run; continue those stages
with the same case list. Completed airway outputs are immutable, so run that
step into a new directory when recomputation is needed. Do not expand a one-case
run to the full cohort in the same output directory.

The complete workflow already computes the four airway parameters with the
global algorithm profile and automatic generation assignment. To run this step
separately after the measurement stage:

```bash
python scripts/run_parse_airway.py \
  --measurement-root data/parse/example/measurements \
  --output-root data/parse/example/airway_parameters \
  --case PA000005 --workers 1 \
  --surface-profile configs/airway_surface.json
```

For a completed full PARSE measurement cohort, use its measurement root and
`--all --workers 4` instead of `--case PA000005 --workers 1`. No case-specific
adjustments are loaded by default. If needed, explicitly supply a private
external schema-2 configuration with `--generation-adjustments /path/to/local.json`.
Such engineering inputs must match the source graphs and do not establish expert
anatomical or wall-boundary acceptance. They must not be committed to this
repository. Automatic generation assignments can differ from previously adjusted
study measurements; this code-only release does not promise those adjusted values.

The statistics stage generates the main radius, cross-sectional area and volume
distributions. Run a retained stage script with `--help` for explicit input/output
options. Generated tables, plots and diagnostic outputs remain local.

## Reproduce AirRC Dice and IoU

Prepare the local inventory, then run the overlap-only batch. The study used a
single GPU with at most four inference processes. Worker counts are limits;
reduce them for the available GPU and host memory.

```bash
python scripts/prepare_airrc.py --list
python scripts/run_airrc_validation_batch.py \
  --dataset-root data/airrc --run-id airway_overlap \
  --scope segmentation_overlap --device gpu \
  --workers 4 --inference-workers 4 --report-every 0 --build-report
```

Use `--resume` with the same run ID and options to continue a stopped run.
Resume verifies the saved code and input identities; use a new run ID after
changing calculation code. The supported scope is `segmentation_overlap`.
This workflow compares the raw predicted lumen and wall masks with the
corresponding reference labels; it includes the trachea as segmented and all
connected components. It does not run airway-wall parameter validation or depend
on PARSE outputs. Cohort statistics use saved Dice/IoU measurements; there is no
periodic source-mask remeasurement. Runtime outputs stay under `data/airrc/`.

## Measurement definitions

| Measurement | Definition used in this study |
| --- | --- |
| PARSE scope | Largest 26-connected artery and airway components, selected independently; original lung/lobe unions remain unchanged |
| Basic graph metrics | Fiji skeleton length and AnalyzeSkeleton branch counts; these differ from the EDT tree's segment identities |
| EDT radius | Distance to the nearest background voxel center in physical millimeters; each segment uses the length-weighted mean |
| Orthogonal area | Perpendicular mask sections at physical centerline positions; the segment value is a length-weighted mean over measurable adjacent intervals |
| Radius groups | `(0, 2]`, `(2, 4]`, `(4, 6]`, and `>6 mm`, using the unrounded segment mean EDT radius |
| Area groups | BV5 `<5`, BV5-10 `5–10` inclusive, and BV10 `>10 mm²` |
| Estimated volume fractions | Segment mean area × measured length, summed by group and divided by the patient's total measurable-segment estimate |
| Airway parameters | LA/WT/WAP: equal branch means at generations 3–5; Pi10: Statsmodels RLM/HuberT at generations 0–6; trachea is generation 0 |
| Patient statistics | Each patient receives equal weight; P10/P25/P50/P75 use linear interpolation; pooled volume ratios are distinct statistics |
| Missing measurements | Retain unavailable reasons, coverage and quality flags; missing areas do not enter estimated-volume denominators |

The wall estimator is `branch_local_coupled_surfaces_v2`, an independent
source-guided approximation using coupled discrete surfaces. It is not the
original OPFRONT implementation. EDT radius is a voxel-distance estimate, and
estimated tube volume is distinct from segmentation-mask volume and physiological
blood volume. Preserved numerical and topology flags remain part of each result.

AirRC and PARSE have different documented source cohorts. The deployed
TotalSegmentator `lung_vessels` task is partly based on AirRC-related work, and
its exact training-case overlap has not been established. The AirRC scores are
annotation comparisons and do not establish independent external generalization.

## Syntax check

```bash
python -m py_compile scripts/*.py
```

The local `tests/` directory is excluded from the published repository.

## License and references

The project code is licensed **GPL-3.0-or-later**; see [LICENSE](LICENSE).
External software, pretrained weights and datasets retain their own terms.
The Fiji bridge calls GPLv3+ Skeletonize3D and AnalyzeSkeleton plugins;
TotalSegmentator is an external Apache-2.0 dependency. The code license does not
relicense the underlying datasets, which are not distributed here.

Please cite the methods and datasets you use:

- [PARSE challenge overview](https://arxiv.org/abs/2304.03708).
- [AirRC dataset and annotation methods](https://doi.org/10.1038/s41597-025-06074-6).
- [LUNA16](https://luna16.grand-challenge.org/) and its underlying LIDC-IDRI data.
- [TotalSegmentator and task references](https://github.com/wasserth/TotalSegmentator#subtasks).
- [Skeletonize3D](https://imagej.net/plugins/skeletonize3d) and [AnalyzeSkeleton](https://imagej.net/plugins/analyze-skeleton/).
- [Airway wall method family](https://doi.org/10.1016/j.media.2014.02.004); the independent approximation and its differences are declared above.
