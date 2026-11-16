# QC correction record

`QC.confirmed-20261117.json` is derived from the user's original
`QC (2).json` (SHA-256
`a8f472921a5ae8b2a94c53cb60431a94d15004dfa60c4a3b4f9bcaa9b5d1a708`).
The user explicitly rechecked PV-09, PV-10, PV-34, PV-38 and PV-64 on
2026-11-17 and reported that their earlier "missing annotation" flags were
mistakes. Only those five per-image statuses, the reviewed-ID list, the
pending-issue list and explanatory notes were changed. The original QC
file was not overwritten. No source mask or dataset file was modified.

The corrected QC has SHA-256
`5d88c8852b255e4196446020f930cc076fe453c9a89cb29996cf9c10c639c6f0`.
Its identity matches the uploaded G0 manifest and provisional split. It
passes `validate_qc`, but that verifies structure and declared human review;
it cannot independently establish label completeness or source-scene
independence. In particular, PV-64 and PV-66 are only about 5.008 km apart,
and the five cross-boundary 10 km sensitivity components remain a documented
limitation.

On the existing Kaggle session, pull this commit and run:

```bash
python -m pvstudy.cli --config config.kaggle.json lock-split --qc-file protocol/QC.confirmed-20261117.json
python -m pvstudy.cli --config config.kaggle.json cache
```

The lock command must succeed before caching. The cache should include only
the 79 development images. Preserve the generated manifest, split lock,
cache index and this QC file with the experiment artifacts.
