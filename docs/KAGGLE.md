# Kaggle execution

Copy `config.example.json` to a writable `config.json` and set these paths
after the Kaggle inputs are mounted:

| Key | Required content |
| --- | --- |
| `image_dir` | Root of nested 512 x 512 RGB TIFF images |
| `label_dir` | Root of labels with matching relative TIFF paths |
| `georsclip_weights` | `RS5M_ViT-B-32.pt` |
| `dinov3_weights` | SAT-493M ViT-L/16 `.pth` |
| `dinov3_source` | Source tree containing `dinov3/hub/backbones.py` |
| `output_dir` | Writable path, e.g. `/kaggle/working/pvstudy` |

The same values can be supplied as `--image-dir`, `--label-dir`,
`--georsclip-weights`, `--dinov3-weights`, `--dinov3-source`, and
`--output-dir` before the subcommand. `--device cuda` is required for a
practical full run. Neither checkpoint is downloaded by the program. The
GeoRSCLIP full SHA-256 and DINO SHA-256 prefix in the example are checked
before model use; update only after verifying the exact replacement weights.
If Kaggle strips `.git` from a DINO source mount, set the verified
`dinov3_source_commit` in JSON. The cache then records the declared commit,
marks it as not Git-verified, and hashes every mounted `dinov3/*.py` file.

1. Install a CUDA PyTorch build compatible with the Kaggle GPU, then
   `pip install -e .`. Record package versions and the DINO source commit.
2. Run `pvstudy --config config.json audit`. Inspect representative positive,
   negative, smallest-target and difficult-background overlays. Resolve any
   near duplicate or scene relation and rerun with `--manual-groups` if needed.
3. Write a QC JSON with all manifest IDs and the confirmations described in
   `DATA_PROTOCOL.md`; run `pvstudy --config config.json lock-split --qc-file QC.json`.
4. Run `verify-encoders --id '<relative-image-id>'`. It checks strict weight
   loading, the 224 native GeoRSCLIP CLS identity, 512 dense shapes, DINO
   four-layer output, determinism, finiteness and freeze state. Semantic
   ranking and calibration still require development data evaluation.
5. Run `cache`. This writes only development features; the cache index binds
   image/checkpoint/source hashes, prompts, preprocessing and augmentation.
6. For each fold 0-3 and seed 42-44, run `run-fold --fold F --seed S` once
   to fit the CAM stage. Then run supported `--variant` values using the same
   fold/seed and hyperparameters. The later runs load the fixed CAM maps.
7. Run `summarize-dev`. Missing fold/seed files yield `incomplete`, not a
   pass. Use its G2-G5 checks only after G1 and manual QC have passed.

The current feature cache supports `augmentation=none`. Applying random RGB
colour or geometry changes only to the shallow decoder while reusing frozen
features would violate the registered input alignment, so this is rejected.
Training uses each image once per epoch, physical batch 1, gradient
accumulation 4, AdamW, up to 100 epochs and validation patience 12.

The 100-image experiment, full baseline grid and G6 test are not performed
by this repository version. Do not turn on `--include-test` before the
registered development selection and final test procedure are implemented
and locked.
