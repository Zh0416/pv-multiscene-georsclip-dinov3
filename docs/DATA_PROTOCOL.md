# Data and isolation contract

`image_dir` and `label_dir` contain matching relative `.tif`/`.tiff` paths,
including any positive/negative subdirectories. Every pair must be a 512 x
512, three-band uint8 image and a one-band uint8 label with only values 0/1.
CRS must identify as EPSG:4326, and raster transforms must match exactly.
The audit stores relative IDs, full SHA-256 values, bounds, geodesic meters
per pixel, positive/negative status and target fraction. It does not alter
source files. Thumbnail matches are only candidates for human near-duplicate
review; no automatic perceptual duplicate guarantee is claimed.

The 5 km graph connects pairs whose geographic footprints are closer than
5 km; connected components are indivisible sites. `--manual-groups` accepts
a JSON list of ID lists for confirmed same-scene images beyond 5 km. Run
`audit` again after changing this list. A 10 km grouping is stored only for
sensitivity diagnostics and never changes the locked test assignment.

Seed 20261116 selects an approximately 20-image test partition while keeping
sites intact and balancing positive/negative images. Seeds 20261117-19 select
four development folds with a negative image in every fold. The exact
assignment and `splits.py` hash are saved in `outputs/split_lock.json`. If
manual scene review finds a cross-partition source scene, rerun the audit
before locking. Do not select splits from model outcomes.

`lock-split --qc-file QC.json` requires `reviewed_ids` to list every manifest
ID and true values for `band_order_confirmed`, `label_alignment_confirmed`,
and `source_scene_checked`. If duplicate candidates exist, it also requires
`duplicate_candidates_resolved`. The QC file's SHA-256 is recorded. This
explicit lock is an audit assertion by the researcher; code cannot perform
the visual checks on their behalf.

Training folds use only development sites. For each outer training fold,
four inner site partitions produce out-of-fold semantic probabilities and
bank responses. Calibration and prototypes for an inner held-out site use
the other three parts. Seed/R thresholds are chosen from those out-of-fold
training responses. Outer validation uses the full outer training bank and
calibrators. Test records cannot be loaded while the split is provisional.
Images, labels, feature caches and predictions stay in ignored `outputs/`.
