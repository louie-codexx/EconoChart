# Data directory

Only `samples/`, `manifests/`, and this file belong in the main Git repository.

```text
data/
├── README.md
├── manifests/                  tracked dataset design and release metadata
├── samples/econochart_v2/      tracked 8-entity integration sample
└── generated/                  ignored full data built on AutoDL
    ├── econochart_v2/
    │   └── subsets/            ignored frozen SFT/GRPO/validation budgets
    └── public/
        ├── chartqa/
        └── chartqapro/
```

Build the full domain dataset:

```bash
econochart-build --config configs/data/econochart_v2.yaml
econochart-validate --dataset-root data/generated/econochart_v2 --full-image-scan
econochart-build-subsets --config configs/data/training_subsets.yaml
```

The subset builder verifies the frozen full-dataset manifest hash and writes a
checksum-bearing `subset_manifest.json`. It selects by entity/chart coverage and
task quotas; it never truncates the first N JSONL rows. The 512-row development
panel is sampled from `val`, while the complete 2,496-row `test` split remains
unchanged for milestone evaluation.

Prepare public data from official sources:

```bash
econochart-prepare-public --config configs/data/public_datasets.yaml --dataset chartqa
econochart-prepare-public --config configs/data/public_datasets.yaml --dataset chartqapro
```

Never force-add `data/generated`, legacy `data/raw`/`data/processed`, or public caches. If a full dataset is published later, use a dataset host/object store, include the manifest/checksums, source licenses, builder Git commit, and link it from the README.
