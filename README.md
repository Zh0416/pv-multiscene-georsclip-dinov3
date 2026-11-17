# PV multi-scene segmentation study

This is an independent implementation workspace for the pre-registered
GeoRSCLIP + DINOv3-SAT photovoltaic segmentation study. The unchanged
[design](PV_MULTISCENE_MODEL_DESIGN.md) is the scientific protocol. This
repository contains no image, label, checkpoint, cached feature or trained
model. It does not claim a model result.

The current Kaggle configuration uses GeoRSCLIP ViT-L/14-336 in an isolated
output directory. Follow [the L/14 Kaggle steps](docs/L14_336.md) to reuse the
locked split and run the new development diagnostic. The original B/32
configuration remains in `config.kaggle.b32.json`; the E03/E05/E04/E08
instructions below describe that historical run.
After reviewing the L/14 E08 diagnostic, use [the isolated E08 development
workflow](docs/E08_PROMOTION.md) to build a verified cache and run G2-G5.

The implemented path covers strict TIFF audit, provisional 5 km site groups,
locked 4-fold development assignment, frozen encoder interfaces, feature
cache provenance, inner-site out-of-fold calibration and prototype responses,
C0a-C7 CAMs (C0b only when a valid site permutation exists), a recoverable
fusion gate, local cross-attention, S2-S7/S11/S13/S14 decoder variants,
site-level metrics and development G2-G5 summaries. See
[implementation status](docs/IMPLEMENTATION.md) for the remaining registered
baselines and experiments.

After a completed B/32 development run, `pvstudy --config config.kaggle.b32.json
diagnose-semantic` runs an isolated E05 comparison of native512 against
overlapping native224 patch maps. It reuses the locked split and original
cache, loads only GeoRSCLIP, and writes a compact review bundle without
retraining or replacing the existing CAM/decoder results. See
[E05 Kaggle instructions](docs/E05.md).

The follow-up E03 command `pvstudy --config config.kaggle.b32.json
diagnose-hard-negatives` uses the completed E05 raw maps as the full-text
reference and fits separate four-fold calibrators for positive-only text
scores. It compares both fixed geometries without modifying earlier results.
See [E03 Kaggle instructions](docs/E03.md).
The isolated E04 final-block and exploratory E08 local-CLS diagnostics are
documented in [E04/E08 Kaggle instructions](docs/E04_E08.md).

## Setup

Use Python 3.9+ and a CUDA PyTorch build for real encoder runs. In Kaggle,
place a writable copy of this repository under `/kaggle/working` and install
it with its declared dependencies. `config.kaggle.json` contains the new L/14
checkpoint; the historical commands below use `config.kaggle.b32.json`.
`config.example.json` remains a template for other environments. The DINOv3 source checkout is
prepared separately at the pinned official commit and recorded in the cache
identity.

```bash
pip install -e .
pvstudy --config config.kaggle.b32.json prepare-source
pvstudy --config config.kaggle.b32.json audit
pvstudy --config config.kaggle.b32.json verify-encoders --id '正样本/PV-01.tif'
pvstudy --config config.kaggle.b32.json export-qc
# Review qc_review.html and export the actual QC.json to this working directory.
pvstudy --config config.kaggle.b32.json lock-split --qc-file QC.json
pvstudy --config config.kaggle.b32.json cache
pvstudy --config config.kaggle.b32.json run-fold --fold 0 --seed 42
pvstudy --config config.kaggle.b32.json run-fold --fold 0 --seed 42 --variant S2
pvstudy --config config.kaggle.b32.json run-fold --fold 0 --seed 42 --variant S6
pvstudy --config config.kaggle.b32.json summarize-dev
```

`audit` writes a **candidate**, not a locked split. Inspect images, masks,
source scenes and duplicate candidates before `lock-split --qc-file ...`.
Development training refuses a candidate split. `cache` excludes test images
unless the split is locked and `--include-test` is explicit. No command in
this version evaluates test labels; G6 remains unavailable.

Read [Kaggle steps](docs/KAGGLE.md), [data contract](docs/DATA_PROTOCOL.md),
[manual quality review](docs/QUALITY_REVIEW.md), and
[local acceptance](docs/ACCEPTANCE.md) before a full run.
