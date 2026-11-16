# Kaggle execution

`config.kaggle.json` records the Kaggle checkpoint and image paths supplied
by the dataset owner, plus the confirmed sibling label directory:

```text
/kaggle/input/datasets/zh0416/pv-dinov3sat-encoder-checkpoints/RS5M_ViT-B-32.pt
/kaggle/input/datasets/zh0416/pv-dinov3sat-encoder-checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
/kaggle/input/datasets/zh0416/pv-100scence/PV-多场景/images
/kaggle/input/datasets/zh0416/pv-100scence/PV-多场景/labels
```

Kaggle execution has not verified that these files are mounted or readable.
The configuration also uses `/kaggle/working/dinov3-source` for source and
`/kaggle/working/pvstudy` for writable outputs. The dedicated Kaggle
configuration is intentionally rejected on Windows, where `/kaggle/...`
would otherwise resolve to a local drive. For another mount layout, copy
`config.example.json` to a writable `config.json` and set these paths:

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

1. Place a writable copy of this repository under `/kaggle/working`, install
   a CUDA PyTorch build compatible with the Kaggle GPU, then run
   `pip install -e .` from that copy. With Kaggle Internet enabled, run
   `pvstudy --config config.kaggle.json prepare-source`; it clones the official
   DINOv3 repository and checks out commit
   `6876159a11b4df116f30f667f8c9888617df0751`. If Internet is disabled,
   mount that pinned source as a Kaggle dataset and change `dinov3_source` to
   the mounted folder. An existing checkout at a different commit is rejected.
2. Run `pvstudy --config config.kaggle.json audit`. Inspect representative positive,
   negative, smallest-target and difficult-background overlays. Resolve any
   near duplicate or scene relation and rerun with `--manual-groups` if needed.
3. Write a QC JSON with all manifest IDs and the confirmations described in
   `DATA_PROTOCOL.md`; run `pvstudy --config config.kaggle.json lock-split --qc-file QC.json`.
4. Run `pvstudy --config config.kaggle.json verify-encoders --id '正样本/PV-01.tif'`.
   It checks strict weight loading, the 224 native GeoRSCLIP CLS identity,
   512 dense shapes, DINO
   four-layer output, determinism, finiteness and freeze state. Semantic
   ranking and calibration still require development data evaluation.
5. Run `pvstudy --config config.kaggle.json cache`. This writes only development
   features; the cache index binds image/checkpoint/source hashes, prompts,
   preprocessing and augmentation.
6. For each fold 0-3 and seed 42-44, run
   `pvstudy --config config.kaggle.json run-fold --fold F --seed S` once
   to fit the CAM stage. Then run supported `--variant` values using the same
   fold/seed and hyperparameters. The later runs load the fixed CAM maps.
7. Run `pvstudy --config config.kaggle.json summarize-dev`. Missing fold/seed
   files yield `incomplete`. Use its G2-G5 checks only after G1 and manual QC
   have passed.

The current feature cache supports `augmentation=none`. Applying random RGB
colour or geometry changes only to the shallow decoder while reusing frozen
features would violate the registered input alignment, so this is rejected.
Training uses each image once per epoch, physical batch 1, gradient
accumulation 4, AdamW, up to 100 epochs and validation patience 12.

The 100-image experiment, full baseline grid and G6 test are not performed
by this repository version. Do not turn on `--include-test` before the
registered development selection and final test procedure are implemented
and locked.
