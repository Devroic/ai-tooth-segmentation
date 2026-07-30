# Tooth Segmentation — Panoramic Radiographs

Binary and multi-class tooth segmentation, applied to panoramic dental
radiographs.

Given a panoramic X-ray, the system:

1. **Binary segmentation** — identifies which pixels belong to a tooth vs.
   background.
2. **Multi-class segmentation** — segments and numbers each individual tooth
   using FDI (ISO-3950) notation, and derives its tooth-group
   (incisor / canine / premolar / molar) from the FDI code.

## Project layout

```
Dockerfile, .dockerignore     container image for the web app (no Python install needed)
run.bat                       one-command launcher (sets up .venv, runs either app)
tests/                         unit tests for taxonomy.py and pipeline post-processing
Datasets/                    11 raw source datasets (not modified, gitignored)
docs/                         project reference material
models/                        shipped inference weights (tracked in git, see below)
  binary_seg/best.pt
  multiclass_seg/best.pt
src/tooth_seg/
  taxonomy.py                FDI <-> Universal numbering, tooth-group mapping
  data/converters/           one converter per source dataset -> unified schema
  inference/pipeline.py      combined binary + multiclass inference pipeline
scripts/
  data/
    prepare_all.py           run every converter: Datasets/ -> outputs/unified/*.json
    analyze_datasets.py      dataset analysis + plots
    dedupe_and_split.py      cross-dataset duplicate detection + train/val/test split
    materialize_yolo.py      build final YOLOv8-seg dataset folders
  model/
    train_binary.py          train the binary segmentation model
    train_multiclass.py      train the multi-class (32-FDI) segmentation model
    evaluate.py               evaluate a trained model on its test split
    predict_samples.py        run both models on sample images, save visualizations
    calibrate_confidence.py   measure precision/recall/F1 vs confidence on the test set
app/app.py                    Gradio dev tool: upload a radiograph, see results
web/
  backend/main.py             FastAPI wrapper around the same inference pipeline
  frontend/                   React UI for clinical use (odontogram, reports, etc.)
  frontend/dist/               prebuilt frontend bundle (tracked in git, see below)
outputs/                      generated data/reports/retraining runs (gitignored, not source)
```

## Quick start (no training required)

The two trained checkpoints (`models/binary_seg/best.pt`,
`models/multiclass_seg/best.pt`, ~6.5MB each) and the prebuilt frontend
(`web/frontend/dist/`) are committed to the repo, so both apps work
immediately after cloning - no dataset, no training run, no `npm install`.

`run.bat` sets up the venv on first run (if missing) and launches either app:

```
run.bat            # prompts: web app or Gradio
run.bat web         # clinical web app -> http://127.0.0.1:8000
run.bat gradio      # Gradio dev tool
```

Or do it by hand:

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .

.venv\Scripts\python.exe app/app.py                                        # Gradio
.venv\Scripts\python.exe -m uvicorn web.backend.main:app --port 8000       # web app
```

`Datasets/` and `outputs/` remain gitignored - they're only needed if you
want to retrain from scratch (see "Reproducing the pipeline" below).

## Tests

**Python** (`tests/`): `tooth_seg.taxonomy`, the FDI post-processing
(`resolve_duplicate_fdi`), and an end-to-end pipeline run against the shipped
weights and a committed sample radiograph (`tests/fixtures/`).

```
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests/
```

**Frontend** (`web/frontend/src/**/*.test.js`): the logic-bearing parts of
the UI - duplicate-FDI grouping/sorting in `Odontogram`, confidence tiering
in `DetectionsTable`.

```
cd web/frontend
npm install
npm test
```

## Docker (no Python required)

The web app also runs in a container - nothing to install beyond Docker
itself:

```
docker build -t tooth-seg .
docker run -p 8000:8000 tooth-seg
```

Open `http://127.0.0.1:8000`. The image bundles the shipped weights and
prebuilt frontend, so this works right after cloning, no other setup.
(Gradio isn't included in the image - it's a local dev tool, see `run.bat`
above.)

## Data

11 candidate dataset folders were inspected; **all are panoramic/periapical
X-ray radiographs** (verified by opening sample images, not just folder
names) — see `outputs/reports/dataset_analysis.json` for the full breakdown.
7 were used:

| id  | role                      | images | annotation format                | numbering      |
| --- | ------------------------- | ------ | -------------------------------- | -------------- |
| 34  | multiclass (true mask)    | 772    | COCO polygons                    | FDI            |
| 11  | multiclass (true mask)    | 598    | Supervisely bitmaps              | Universal 1-32 |
| 3   | multiclass (true mask)    | 1000   | Labelbox-style polygon fragments | Universal 1-32 |
| 30  | multiclass (**box-only**) | 1448   | YOLO bounding boxes              | FDI            |
| 42  | binary only               | 6225   | paired PNG masks                 | —              |
| 20  | binary only               | 1776   | paired BMP masks                 | —              |
| 18  | binary only               | 2000   | paired PNG masks (low-res)       | —              |

Datasets 2, 5, 12, 27, 43 were excluded (unannotated, pathology-only labels,
or duplicate/low-value relative to the above).

All annotations are converted into one **unified schema** (see
`src/tooth_seg/data/converters/common.py`) with FDI numbering as the
canonical tooth identity, and tooth-group derived from FDI via
`tooth_seg.taxonomy.fdi_to_group`.

After conversion: 13,819 images total, 136,935 tooth annotations, all 32 FDI
classes present with a reasonably balanced distribution (~1,900-3,200
annotations per tooth code; the expected dip is at wisdom teeth 18/28/38/48,
which are frequently missing/unerupted). See
`outputs/reports/dataset_analysis.json` and the accompanying plots.

Final training-ready pools (after materialization, resizing, re-encoding,
and the dedup-consistent split):

| pool                         | train | val | test | imgsz |
| ---------------------------- | ----- | --- | ---- | ----- |
| multiclass (32 FDI classes)  | 2649  | 315 | 350  | 640   |
| binary (tooth vs background) | 4373  | 522 | 564  | 512   |

**Cross-dataset deduplication:** several source datasets are suspiciously
similarly sized (e.g. dataset 11 and part of dataset 20 both have ~598-600
images) and likely share source images from the same public corpora.
`scripts/data/dedupe_and_split.py` hashes every image (difference-hash) and
ensures duplicate images always land in the same split, so no test data
leaks into training via a differently-annotated copy. See
`outputs/reports/dedup_report.json` for what was found.

## Training environment

Python 3.12 was used (3.14 is too new for the current PyTorch/Ultralytics
wheels). No dedicated GPU was available on the development machine (Intel Iris Xe
integrated graphics only) — everything here is configured to train on CPU
with small models (YOLOv8n-seg) and modest image sizes. A 1-epoch timing
test (4373 images, imgsz=512, batch=8) took ~34 minutes and already reached
mask mAP50=0.71 from COCO-pretrained transfer learning, which set the epoch
budgets used for the real runs (15 epochs binary, 25 epochs multiclass, both
with early-stopping patience) - a multi-hour, not multi-day, commitment. The
two models were trained **in parallel** (12 logical cores available) to cut
total wall-clock time versus running them sequentially.

If you have access to a GPU (e.g. Google Colab / Kaggle, free tier), the
same scripts will use it automatically if `torch.cuda.is_available()` - just
change `device="cpu"` to `device=0` in `scripts/model/train_binary.py` /
`train_multiclass.py`, and you can raise `--imgsz`/`--epochs`/`--batch`
substantially for better accuracy.

## Reproducing the pipeline (optional - retraining from scratch)

Not required for everyday use (see "Quick start" above) - only needed if you
want to retrain on the raw datasets yourself. Requires `Datasets/` locally
(not included in the repo, 11GB).

```
# 1. Convert all raw datasets to the unified schema
.venv\Scripts\python.exe scripts/data/prepare_all.py

# 2. Dataset analysis
.venv\Scripts\python.exe scripts/data/analyze_datasets.py

# 3. Cross-dataset dedup + split assignment
.venv\Scripts\python.exe scripts/data/dedupe_and_split.py

# 4. Build YOLOv8-seg-ready directories (binary + multiclass)
.venv\Scripts\python.exe scripts/data/materialize_yolo.py

# 5. Train (binary then multiclass) - each takes a while on CPU
.venv\Scripts\python.exe scripts/model/train_binary.py
.venv\Scripts\python.exe scripts/model/train_multiclass.py

# 6. Evaluate on the held-out test split
.venv\Scripts\python.exe scripts/model/evaluate.py --weights outputs/runs/binary_seg/weights/best.pt --data outputs/data/binary/data.yaml
.venv\Scripts\python.exe scripts/model/evaluate.py --weights outputs/runs/multiclass_seg/weights/best.pt --data outputs/data/multiclass/data.yaml

# 7. Sample prediction visualizations
.venv\Scripts\python.exe scripts/model/predict_samples.py

# 8. (optional) Re-check the confidence threshold against the new weights
.venv\Scripts\python.exe scripts/model/calibrate_confidence.py

# 9. Copy the new checkpoints over the shipped ones used by the apps
cp outputs/runs/binary_seg/weights/best.pt models/binary_seg/best.pt
cp outputs/runs/multiclass_seg/weights/best.pt models/multiclass_seg/best.pt
```

## Results

Both models were trained on CPU only (YOLOv8n-seg, 12-core machine, no
dedicated GPU) and evaluated on their held-out **test** split (never seen
during training or validation, and dedup-consistent with train/val so no
near-duplicate radiograph leaked across the split).

| model                        | epochs | mask precision | mask recall | mask mAP50 | mask mAP50-95 |
| ---------------------------- | ------ | -------------- | ----------- | ---------- | ------------- |
| binary (tooth vs background) | 15     | 0.818          | 0.835       | **0.852**  | 0.506         |
| multiclass (32 FDI classes)  | 25     | 0.798          | 0.838       | **0.874**  | 0.515         |

Full metrics (including per-FDI-class breakdown for the multiclass model)
are in `outputs/runs/eval/eval_test.json` and
`outputs/runs/eval-2/eval_test.json`; confusion matrices and PR curves are
saved alongside them. Every one of the 32 FDI classes scores mask mAP50
between 0.66 and 0.95 on the test set - no class collapsed to near-zero,
though the third-molar/wisdom-tooth classes (18/28/38/48) and tooth 31 are
the weakest, consistent with them being the rarest and most
position-ambiguous teeth in the training data.

**Confidence threshold:** the web app runs detection at a fixed `conf=0.15`
(`web/frontend/src/App.jsx`), not a typical F1-optimal default, by deliberate
design - it's recall-first. `scripts/model/calibrate_confidence.py` measures
why on the held-out test set:

| confidence               | mean precision | mean recall | mean F1 |
| ------------------------- | -------------- | ----------- | ------- |
| 0.15 (used by the web app) | 0.599          | **0.944**   | 0.731   |
| 0.25 (Gradio default)      | 0.698          | 0.908       | 0.787   |
| 0.43 (F1-optimal)          | 0.813          | 0.827       | **0.819** |

At 0.15, the model misses only ~6% of real teeth, versus ~17% at the
F1-optimal point - the traded-off precision is an acceptable cost because
low-confidence detections are flagged for review in the UI rather than
hidden (see `LOW_CONFIDENCE_THRESHOLD` in `web/frontend/src/dental.js`,
set just above the 0.43 precision/recall break-even).

Qualitative samples (original / binary overlay / multiclass FDI+group
overlay) are in `outputs/reports/sample_predictions/`, generated by
`scripts/model/predict_samples.py`. Predictions correctly follow the
anatomical left-to-right, quadrant-consistent tooth order in every sample
inspected.

**Known limitation:** the model occasionally predicts the same FDI code for
two different visible teeth - most often a tooth and its bilateral mirror
(11 vs 21) - since panoramic X-rays only weakly disambiguate left/right by
position and the model has no explicit left-right symmetry constraint.
`tooth_seg.inference.pipeline.resolve_duplicate_fdi` post-processes raw
detections to mitigate this: overlapping same-FDI detections (the same tooth
flagged twice) keep only the higher-confidence one, and non-overlapping
same-FDI pairs (a genuine mirror mixup) get the anatomically-wrong one
relabeled to its mirror FDI instead of left as a duplicate. Residual
duplicates the UI still flags for review are cases this heuristic can't
confidently resolve (e.g. the mirror code is already taken by another
detection).

## Two UIs, one inference pipeline

There are two separate front ends, both calling the exact same
`tooth_seg.inference.pipeline.ToothSegPipeline` - neither reimplements any
model or post-processing logic, so they always agree:

- **`app/app.py`** - a single-file Gradio tool for developers: fastest way
  to sanity-check a newly trained checkpoint. Exposes the raw ML controls
  (confidence-threshold slider, binary tooth-mask view, editable weight
  paths) that make sense for a technical user comparing checkpoints.
  Unchanged by the web app below. Run: `.venv\Scripts\python.exe app/app.py`
- **`web/`** - a React frontend + thin FastAPI backend, built for a
  non-technical clinical user. Deliberately shows less than Gradio: no
  confidence-threshold control (that's an ML tuning concept, not a
  clinical one - detections are always shown in full, with confidence
  communicated visually instead) and no binary-mask view (not clinically
  meaningful on its own). Shows the annotated radiograph alongside an
  **odontogram** (the standard tooth-chart layout dentists already read),
  color-codes confidence per tooth (green/amber/red, both in the table and
  as a dashed low-confidence outline on the odontogram), and flags teeth
  with duplicate/ambiguous FDI predictions for manual review. The backend
  serves the prebuilt frontend directly, so everyday use is also a single
  command, no Node/npm required at runtime:
  `.venv\Scripts\python.exe -m uvicorn web.backend.main:app --port 8000`,
  then open `http://127.0.0.1:8000`. See `web/frontend/README.md` for the
  frontend-development (hot-reload) workflow.
