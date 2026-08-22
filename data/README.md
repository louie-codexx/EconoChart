# Data directory

Only `samples/`, `manifests/`, and this file belong in the main Git repository.

```text
data/
├── README.md
├── manifests/                  tracked dataset design and release metadata
├── samples/econochart_v2/      tracked 8-entity integration sample
└── generated/                  ignored full data built on AutoDL
    ├── econochart_v2/
    └── public/
        ├── chartqa/
        └── chartqapro/
```

Build the full domain dataset:

```bash
econochart-build --config configs/data/econochart_v2.yaml
econochart-validate --dataset-root data/generated/econochart_v2 --full-image-scan
```

Prepare public data from official sources:

```bash
econochart-prepare-public --config configs/data/public_datasets.yaml --dataset chartqa
econochart-prepare-public --config configs/data/public_datasets.yaml --dataset chartqapro
```

Never force-add `data/generated`, legacy `data/raw`/`data/processed`, or public caches. If a full dataset is published later, use a dataset host/object store, include the manifest/checksums, source licenses, builder Git commit, and link it from the README.
