# Current L/14-336 + DINOv3-SAT experiment

This is a development-stage amendment after observing failure of the earlier
B/32 semantic path. **ViT-B/32 is excluded from the current experiment matrix**;
its saved results are historical context, not paired controls for the new model.
The unchanged original registration remains in `PV_MULTISCENE_MODEL_DESIGN.md`.
Do not describe this amendment as if E08 had been registered before inspecting
development results.

## Locked inputs

- Dataset: the existing 100 image/label TIFF pairs at the paths in
  `config.kaggle.e08.json`; keep the manually confirmed 79-development /
  21-test site split. Test labels remain sealed.
- Encoders: the existing `RS5M_ViT-L-14-336.pt` and
  `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` at the configured Kaggle
  paths. Do not change checkpoint, normalization, DINO source commit or
  512-pixel source geometry in this run.
- Semantic route: the reviewed `l14-semantic-v2` E08 raw maps, 49 local
  128-pixel windows at stride 64 resized to native 336; positive-only text.
  Each outer training fold independently fits its calibrator. The native
  full-image CLS still supplies `s_global`.
- DINO route: frozen ViT-L/16 last-four intermediate layers averaged to
  1024 x 32 x 32; training-site out-of-fold prototype bank plus optional
  image-local prototype. No validation/test labels enter calibration or bank.

The reviewed single-route E08 AP was 0.60062 versus constant AP 0.21065.
The subsequent Kaggle development-gate report supplied by the user found G2
passed, G3 failed (`C6-C5g` AP +0.01110 versus +0.02 required), G4 failed
(`S6-S2` Dice -0.00143), and G5 failed (`S7-S6` Dice +0.00035). These
developer-set observations motivate the current attribution run, not a claim
that the full fusion model succeeded. The original B/32 metrics must not be
mixed into L/14 paired comparisons.

## Core comparison

Run four site-held-out folds with seeds 42/43/44, fixed 100-epoch maximum,
effective batch four, identical frozen cache and training settings. Use
site-macro AP for 32 x 32 CAM, positive-site Dice@0.5 for 512 x 512 masks,
and negative-site false-positive area as the separate safety measure.
Report each site's three-seed mean and paired bootstrap interval. An absent
or negative result stays in the table.

| Comparison | Question | Existing decision rule |
| --- | --- | --- |
| `C1/C2 - C0a` | Are semantic and bank CAMs individually informative? | G2: AP +0.05, at least 3/4 positive folds; bank coverage >=95% and drift pass |
| `C6 - C5g` | Does full recovery fusion beat the strongest simple CAM? | G3: AP +0.02, at least 3/4 positive folds, negative FP delta <=0.005 |
| `C6 - C7` | Does current-image GeoRSCLIP information help the full CAM? | F07: AP +0.01, at least 3/4 positive folds and FP guardrail |
| `S14 - S2` | Does simple C5g prior help the visual decoder? | Descriptive comparison; no post-hoc success threshold |
| `S6 - S14` | Does recovery fusion improve on the same decoder using C5g? | D17: Dice +0.01, 3/4 positive folds, FP delta <=0.005 |
| `S6 - S2` | Does the complete prior improve segmentation? | G4: Dice +0.02, 3/4 positive folds, FP delta <=0.005 |
| `S7 - S6` | Does the attention bias add value? | G5: Dice +0.01, 3/4 positive folds, FP delta <=0.005 |

G3/G4/G5 remain failed for the reported run. Running the same configuration
again can check reproducibility or recover missing per-site results; it does
not erase the earlier negative result. `S14` can be compared with `S2` as a
candidate simpler method, but cannot retroactively make C6 pass G3.

## Extended attribution

The current code also supports `S3`, `S4`, `S5`, `S11` and `S13` with the
same decoder and frozen inputs. Compare `S6-S11` for current-image semantic
contribution, `S6-S13` for learned recovery versus simple gate, and
`S3/S4/S5` for one-route and fixed-fusion controls. Run all on the same
four folds and three seeds. This is a development-stage attribution matrix;
selecting the best row on this data is not independent confirmation.

The original design additionally calls for `S0/S1/S8/S9/S10/S12/S15/S16`,
random-text `C8`, common hyperparameter selection, full-development
retraining and G6. Those are **not implemented** in this repository. No
one-time test or H1-H4 confirmatory claim is authorized by a completed core
or extended report. Full design completion requires implementing the
remaining controls, freezing selection rules, and only then evaluating the
21 held-out images once. If the full fusion continues to fail G3/G4, the
paper must present it as a negative result and select a qualified simpler
model from development evidence.

## Kaggle execution and preservation

Use the current `codex/l14-dinov3-study` branch and install the repository.
The complete `l14-semantic-v2` directory with 79 `raw/*.npz` files must be
present; the compact diagnostic review ZIP alone cannot rebuild it. From a
Kaggle notebook:

```python
from pathlib import Path
import subprocess
import sys

repo = Path('/kaggle/working/pv-multiscene-georsclip-dinov3')
branch = 'codex/l14-dinov3-study'
if (repo / '.git').is_dir():
    subprocess.run(['git', 'fetch', 'origin', branch], cwd=repo, check=True)
    local = subprocess.run(['git', 'show-ref', '--verify', '--quiet',
                            f'refs/heads/{branch}'], cwd=repo)
    switch = ['git', 'switch', branch] if local.returncode == 0 else [
        'git', 'switch', '--track', f'origin/{branch}']
    subprocess.run(switch, cwd=repo, check=True)
    subprocess.run(['git', 'pull', '--ff-only', 'origin', branch], cwd=repo, check=True)
else:
    subprocess.run(['git', 'clone', '--branch', branch,
                    'https://github.com/Zh0416/pv-multiscene-georsclip-dinov3.git',
                    str(repo)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], cwd=repo, check=True)
subprocess.run([sys.executable, 'scripts/run_l14_development.py', '--stage', 'core'],
               cwd=repo, check=True)
```

If the earlier Kaggle output has disappeared, stop before the `core` command
and rebuild the diagnostic with the same Kaggle data and checkpoints. Keep
the `pvstudy-e05-recovery` input dataset attached for the locked manifest
and split:

```python
subprocess.run([sys.executable, 'scripts/rebuild_l14_diagnostic.py'],
               cwd=repo, check=True)
```

This writes a new `l14-semantic-v3` run and prints its review ZIP and summary
SHA-256. Download the ZIP for review. Do not change the pinned E08 config or
run the development matrix until that new diagnostic has been checked; a
fresh timing measurement normally changes the summary hash even when all
semantic maps agree. The recovery input restores the old split, not model
features or trained results.

After the core result, inspect `l14_development_core.json`. To run the
additional supported variants, use `--stage extended`; completed result
files are reused, and a separate `l14_development_extended.json` is written.
Each stage creates `l14_development_<stage>_review.zip` containing result
JSONs, lock/cache identities and per-file hashes. Download this ZIP or save
the Kaggle notebook version with outputs **before the session ends**. It
does not contain raw predictions, cache arrays or model weights; preserving
those needs the full Kaggle output or a separate dataset snapshot.
