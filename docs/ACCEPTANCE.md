# Local acceptance record, 2026-11-16 (Asia/Shanghai)

This record distinguishes code checks from scientific acceptance. The
unmodified design document has the same SHA-256 as its source:
`f7f7b55d0be43eb80ad77ed05302c07f0ef68606f5a5c34b221f5f8e291d6d87`.

| Gate | Local result | Remaining evidence |
| --- | --- | --- |
| G0 | Automated audit passed 100 paired 512 x 512 RGB/0-1-label TIFFs: 80 positive, 20 negative; no exact image duplicates or identical median-threshold thumbnails. Candidate 5 km grouping has 66 sites; candidate test has 21 images (17 positive, 4 negative) across 19 sites; four development validation folds have 20/19/20/20 images and four negative images each. | Manual band/label/scene overlay review, near-duplicate review and QC lock. Five 10 km components cross the candidate 5 km test boundary; preserve this limitation. |
| G1 | Encoder interfaces are implemented and synthetic tensor contracts are tested. | No OpenCLIP package or DINO source checkout was available locally; PyTorch is CPU-only. Actual weight loading, 512 forward, dense semantic ranking, preprocessing and GPU memory remain unverified. |
| G2 | Synthetic inner-site bank, calibration, seed fallback and output-finiteness test passed. | Real out-of-fold AP, coverage, calibration reliability and bank-drift thresholds are unmeasured. |
| G3 | Synthetic fusion range and three-route sum tests passed. | Real C0a-C7 comparisons, negative FP and route distribution are unmeasured. |
| G4 | 512 output shape and one CPU synthetic optimizer step passed. | Real S2-S6 paired site Dice and false-positive guardrail are unmeasured. |
| G5 | Window center/edge indexing and masked attention forward passed. | Real S7-S6 paired Dice and latency are unmeasured. |
| G6 | Test partition remains provisional and sealed. | Final model selection and one-time test are not implemented or run. |

`python -m compileall -q pvstudy tests` passed. `python -m unittest
discover -s tests -v` passed 12 tests, including one-epoch CPU optimizer
steps for the fusion gate and decoder, CAM site pooling and cache integrity.
An eight-site synthetic TIFF/cache integration test ran one CAM fold and S2,
confirmed the CAM result was reused, and produced two validation predictions.
`summarize-dev` correctly returns
`incomplete` while no fold results exist. A `ruff` check was not available:
package installation failed because the local Python environment's network
proxy could not connect to PyPI. No quantitative segmentation result or
paper claim has been produced.

The package wheel built successfully with the locally available Python
environment using setuptools 72.1. The default Python 3.9 installation has
setuptools 58.1 and cannot interpret the PEP 621 metadata without build
isolation; Kaggle setup needs a modern build backend as declared in
`pyproject.toml`.

The local candidate manifest and split are in ignored `outputs/`, so paths,
labels, caches and candidate assignments are not uploaded with source.

## Kaggle mount update

The user supplied the two exact checkpoint paths and confirmed the image and
sibling label directories in `config.kaggle.json`. The pinned DINOv3 source
commit exists in the official GitHub repository; `prepare-source` now checks
out that commit when Kaggle Internet access is available. These Kaggle mount
paths, the source checkout, actual checkpoint forward passes and GPU memory
have **not** been tested from this local machine. A read-only Kaggle API query
was unavailable locally because the CLI environment lacks `kaggle.json`.
