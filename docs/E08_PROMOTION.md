# L/14 E08 development promotion

The L/14-336 diagnostic on the locked 79-image development set selected the
fixed 128-pixel crop, 64-pixel stride, 49-window positive-only CLS route for
the next development experiment. The reviewed `l14_review.zip` contained
`l14_summary.json` SHA-256
`7f1bf50672f51b79e5fad24fadf64a0d66317fb8e6d383729b2c2130abfac55d`.
Its diagnostic identity SHA-256 was
`0051de72878ee10bbebea41cb74de714a0432af5a5cc8fa4ed4f378765368eb7`.
This is an exploratory architecture amendment, not a retroactive E04 success.

The E08 diagnostic produced site-macro AP 0.60062 versus the same-split
constant 0.21065, positive fold deltas in 4/4 folds, mean negative-site
false-positive area 0.01465 and median Tesla T4 extraction time 6.00 seconds
per image. Negative image PV-93 reached false-positive area 0.140625. The
registered E04 value-only and V-V paths both stayed below the constant AP.
These numbers justify a development-only downstream check; they do not prove
G2-G5 or final test performance.

The new `config.kaggle.e08.json` writes to `/kaggle/working/pvstudy-l14-e08`.
It reads *raw*, uncalibrated `window_cls_positive_only` arrays from the
complete `/kaggle/working/pvstudy-l14-336/diagnostics/l14-semantic-v1`
directory. The compact `l14_review.zip` does not contain those arrays. The
cache command checks the pinned summary, diagnostic identity, unchanged
encoder code, locked manifest/split and every raw-map digest. It extracts
DINOv3 features and the unchanged native full-image CLS score, then writes
the E08 raw map as `semantic_raw`. `verify-e08-cache` checks every stored map
element against its diagnostic raw map. The four-fold training code then fits
new inner-training-site calibrators; diagnostic fold calibrators and OOF
probabilities are not reused.

Run the following in a Kaggle GPU notebook. The first cell updates the code,
copies only the existing locked manifest/split, and verifies the required
raw source before loading either checkpoint:

```python
from pathlib import Path
import subprocess
import sys

repo = Path('/kaggle/working/pv-multiscene-georsclip-dinov3')
branch = 'codex/e05-semantic-probe'
if (repo / '.git').is_dir():
    subprocess.run(['git', 'fetch', 'origin', branch], cwd=repo, check=True)
    local = subprocess.run(['git', 'show-ref', '--verify', '--quiet', f'refs/heads/{branch}'],
                           cwd=repo)
    if local.returncode == 0:
        subprocess.run(['git', 'switch', branch], cwd=repo, check=True)
    else:
        subprocess.run(['git', 'switch', '--track', f'origin/{branch}'], cwd=repo, check=True)
    subprocess.run(['git', 'pull', '--ff-only', 'origin', branch], cwd=repo, check=True)
else:
    subprocess.run(['git', 'clone', '--branch', branch,
                    'https://github.com/Zh0416/pv-multiscene-georsclip-dinov3.git',
                    str(repo)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], cwd=repo, check=True)
base = [sys.executable, '-m', 'pvstudy.cli', '--config', 'config.kaggle.e08.json']
source = Path('/kaggle/working/pvstudy-l14-336')
raw = source / 'diagnostics/l14-semantic-v1/raw'
if not (source / 'dataset_manifest.json').is_file() or len(list(raw.glob('*.npz'))) != 79:
    raise FileNotFoundError('Complete L/14 diagnostic output, including 79 raw maps, is required')
subprocess.run(base + ['prepare-source'], cwd=repo, check=True)
subprocess.run(base + ['reuse-lock', '--source', str(source)], cwd=repo, check=True)
subprocess.run(base + ['verify-encoders', '--id', '正样本/PV-01.tif'], cwd=repo, check=True)
subprocess.run(base + ['cache'], cwd=repo, check=True)
subprocess.run(base + ['verify-e08-cache'], cwd=repo, check=True)
```

If a Kaggle reset removed the raw directory, the review ZIP alone cannot
restore it. Reattach a saved complete diagnostic directory, or rerun
`diagnose-l14` with the original L/14 configuration and review its new
summary. A changed summary hash requires updating the pinned E08 configuration
after that review; do not bypass the check.

Run the 12 CAM jobs first. The loop skips complete result files, so it can be
restarted after a Kaggle session ends. Each `run-fold` checks the current
E08 cache verification and locked data identity:

```python
for fold in range(4):
    for seed in (42, 43, 44):
        result = Path(f'/kaggle/working/pvstudy-l14-e08/fold-{fold}/seed-{seed}/result-cam.json')
        if not result.is_file():
            subprocess.run(base + ['run-fold', '--fold', str(fold), '--seed', str(seed)],
                           cwd=repo, check=True)
subprocess.run(base + ['summarize-cam'], cwd=repo, check=True)
```

Review `/kaggle/working/pvstudy-l14-e08/cam_development_gates.json` before
training decoders. It reports G2/G3 using the same three-seed, four-fold
development grid. If either gate fails, preserve its negative result and do
not label the corresponding E08 fusion route effective.

For the implemented S2/S6/S7 decoder comparisons, continue with:

```python
for variant in ('S2', 'S6', 'S7'):
    for fold in range(4):
        for seed in (42, 43, 44):
            result = Path(f'/kaggle/working/pvstudy-l14-e08/fold-{fold}/seed-{seed}/result-{variant}.json')
            if not result.is_file():
                subprocess.run(base + ['run-fold', '--fold', str(fold), '--seed', str(seed),
                                       '--variant', variant], cwd=repo, check=True)
subprocess.run(base + ['summarize-dev'], cwd=repo, check=True)
```

The final report is `/kaggle/working/pvstudy-l14-e08/development_gates.json`.
These commands cover the implemented G2-G5 comparisons for the E08 route.
They do not implement the full registered baseline matrix or G6 final-dev
retraining and once-only test. Test images remain sealed.
