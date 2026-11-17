# Implementation map

| Step | Module | Implemented behavior |
| --- | --- | --- |
| G0 | `data.py`, `splits.py`, `cli.py` | Strict TIFF/hash audit; 5 km site components; seeded provisional test/4-fold split; QC lock |
| G1 | `encoders.py`, `source.py`, `cache.py` | Pinned official DINOv3 checkout, strict GeoRSCLIP state dict, native CLS identity check, dense patch path; DINO last-four arithmetic mean; frozen cache provenance |
| G2 | `priors.py` | Site-weighted soft-label calibration, inner-site OOF bank, K=1/4/8, purity 0.5/0.7, semantic seed, bank/self blend, optional OOF spatial recalibration |
| G3 | `models.py`, `experiment.py` | C5/C5g/C6 gate, C7 current-image semantic removal, CAM and route-weight maps, site metrics, bank drift diagnostic |
| G4/G5 | `models.py`, `experiment.py` | S2-S7, S11, S13, S14 with identical decoder capacity; M channel and optional local M attention bias |
| Analysis | `metrics.py`, `analysis.py` | Site macro AP/Dice/IoU, negative FP, area Brier/ECE/MAE, paired site bootstrap helper, G2-G5 development checks |
| E05 diagnostic | `semantic_probe.py`, `GeoRSCLIP.encode_sliding224` | Isolated native512/native224 sliding comparison, four outer-fold calibrations, paired site statistics, synchronized GPU timing, resumable raw maps and review previews; no main-pipeline replacement |
| E03 diagnostic | `hard_negative_probe.py`, `GeoRSCLIP.encode_semantic_pair` | Full hard-negative score versus positive-only cosine on identical native512 geometry, with sliding224 as E05 follow-up; separate outer-fold calibrations and E05 baseline verification |

Registered items **not yet implemented**: GeoRSCLIP value-only/true V-V,
promotion of the E05 sliding path into the full training pipeline; random-text C8/S12; S0/S1,
S8/S10, S9 and external pretrained S15/S16; optional loss/normalization/
window ablations; full development hyperparameter selection, deployment
threshold selection and the G6 final-dev retraining/once-only test workflow.
These IDs cannot be requested as aliases of other models. The project does
not produce H1-H4 confirmation or a final deployment candidate yet.

The native 512 GeoRSCLIP patch projection is an explicit experimental
adapter: patch token projection is mechanically consistent with the CLS
path, but its semantic localization value is unproven until G1/G2 on the
actual checkpoint and development sites. No output is labelled a true V-V
attention map. The 16 x 16 GeoRSCLIP map is interpolated to 32 x 32 without
claiming additional semantic resolution.

The fixed image normalization values are OpenAI CLIP mean/std and the
SAT-493M DINO mean/std used by the existing local interface. The shallow
RGB branch uses fixed uint8/255. Decoder training stops gradients through
M in both its direct channel and attention bias. Its full-resolution loss
is BCE-with-logits plus per-image soft Dice; M uses clamped BCE plus the
same soft Dice. All label-to-patch targets are area averages.
