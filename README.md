# VISA

A **training-free** multimodal recommendation model.

VISA builds item–item graphs from (i) training-interaction co-occurrence and
(ii) modality (text / image) kNN similarity, filters both with a polynomial
graph filter, and adds a lightweight user–user spectral branch. No parameters
are trained: a single forward pass produces the top-N scores.

## Run

```bash
pip install -r requirements.txt
cd src
python main.py -m VISA -d baby
```

Datasets are **not** included — see [`DATA.md`](DATA.md) for the expected layout
(`data_path` is set in `src/configs/overall.yaml`).
Per-dataset hyper-parameters: `src/configs/model/VISA_{baby,clothing,sports}.yaml`.
