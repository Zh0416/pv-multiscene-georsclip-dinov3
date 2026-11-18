# PV multi-scene segmentation study

The current experiment uses frozen GeoRSCLIP ViT-L/14-336 and DINOv3-SAT
on the locked 100-pair photovoltaic dataset. GeoRSCLIP local-window CLS
scores (E08) provide the semantic CAM; DINOv3-SAT features provide the
prototype-bank CAM and visual decoder features. The 79-image development
partition uses four site-held-out folds and seeds 42/43/44. The 21-image
test partition remains sealed.

Read the [current L/14 experiment protocol](docs/L14_CURRENT_PROTOCOL.md)
for the comparison matrix, prior results, failure criteria and Kaggle
commands. The fixed data and checkpoint paths are in
[`config.kaggle.e08.json`](config.kaggle.e08.json). The original
[pre-registration](PV_MULTISCENE_MODEL_DESIGN.md) remains unchanged as a
historical scientific record; the E08 route is a later exploratory amendment.

The current code implements strict data audit and split locking, encoder
checks, E08 raw-map provenance and feature-cache verification, inner-site
out-of-fold calibration and prototypes, C0a-C7 CAMs, C5/C5g/C6 gates,
S2-S7/S11/S13/S14 decoder variants, site-level metrics, L/14 paired
development reports and downloadable JSON review bundles. It does not
implement all registered controls or the G6 final retraining and test
workflow. See the [implementation map](docs/IMPLEMENTATION.md).

For a Kaggle GPU notebook with this repository checked out under
`/kaggle/working` and the complete reviewed `l14-semantic-v3` diagnostic
still available:

```python
from pathlib import Path
import subprocess
import sys

repo = Path('/kaggle/working/pv-multiscene-georsclip-dinov3')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.',
                'open_clip_torch==4.2.2'], cwd=repo, check=True)
subprocess.run([sys.executable, 'scripts/run_l14_development.py', '--stage', 'core'],
               cwd=repo, check=True)
```

If that diagnostic output was lost, restore the reviewed full backup described
in the [current protocol](docs/L14_CURRENT_PROTOCOL.md). A new diagnostic
rebuild needs review before the E08 development configuration is updated.

The script resumes completed fold results, writes
`/kaggle/working/pvstudy-l14-e08/l14_development_core.json`, and packages
the report plus result JSONs and provenance into
`l14_development_core_review.zip`. Save the Kaggle notebook version with
its output or download the ZIP before the session ends. The ZIP is for
analysis, not for resuming training: it omits feature arrays, predictions
and weights.

ViT-B/32 code and its E03/E05 diagnostics remain in the repository solely
for historical reproducibility. They are **not** part of the current L/14
experiment and their metrics are not paired controls. Historical run
instructions are in [E03](docs/E03.md), [E05](docs/E05.md) and
[E04/E08](docs/E04_E08.md).
