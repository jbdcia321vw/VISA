# Data

Datasets are **not** included in this anonymous release (they are large and
publicly available). Please download the MMRec datasets (`baby`, `clothing`,
`sports`) from the original MMRec toolbox release and place them as:

```
data/
├── baby/
│   ├── baby.inter
│   ├── image_feat.npy
│   ├── text_feat.npy
│   └── user_feat.npy
├── clothing/...
└── sports/...
```

Then run, e.g.:

```bash
cd src
python main.py -m VISA -d baby
```

The `data_path` is configured in `src/configs/overall.yaml` (default `../data/`).
The model caches graph/PCA artefacts under `cache/` and `data/<dataset>/` on first
run; both are excluded here and will be regenerated automatically.
