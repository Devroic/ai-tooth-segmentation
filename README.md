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
models/                        shipped inference weights (tracked in git)
  binary_seg/best.pt
  multiclass_seg/best.pt
src/tooth_seg/
  taxonomy.py                FDI <-> Universal numbering, tooth-group mapping
  inference/pipeline.py      combined binary + multiclass inference pipeline
app/app.py                    Gradio dev tool: upload a radiograph, see results
web/
  backend/main.py             FastAPI wrapper around the same inference pipeline
  frontend/                   React UI (odontogram, reports, etc.)
  frontend/dist/               prebuilt frontend bundle (tracked in git)
```

## Quick start

The trained model weights and the prebuilt frontend are committed to the
repo, so the app works immediately after cloning - no setup beyond Python
itself.

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

## Using the app

1. Open the app (`http://127.0.0.1:8000` for the web app) and upload a
   panoramic dental X-ray (JPG or PNG).
2. Click "Analyze X-ray." You'll get back the annotated radiograph, an
   **odontogram** (a dental-chart view of every detected tooth, colored by
   type), and a table listing each detected tooth's FDI number, name, and
   confidence.
3. A dashed outline or a "Review" badge means that particular result is
   lower-confidence or ambiguous - worth a second look rather than blindly
   trusted.
4. Click "Download report (PDF)" to save/print a clean copy of the results.

## Docker (no Python required)

The web app also runs in a container - nothing to install beyond Docker
itself:

```
docker build -t tooth-seg .
docker run -p 8000:8000 tooth-seg
```

Open `http://127.0.0.1:8000`. The image bundles the shipped weights and
prebuilt frontend, so this works right after cloning, no other setup.
(Gradio isn't included in the image - it's a separate local tool, see below.)

## Two apps, one inference pipeline

- **The web app** (`web/`, React + FastAPI) is the one described above under
  "Using the app" - upload an X-ray, get an annotated image, odontogram, and
  report.
- **`app/app.py`** is a separate, simpler Gradio tool with raw controls
  (an adjustable confidence slider, a binary tooth/background view, editable
  model file paths) - useful for comparing model checkpoints directly, not
  needed for everyday use. Run: `.venv\Scripts\python.exe app/app.py`

Both call the exact same underlying pipeline
(`tooth_seg.inference.pipeline.ToothSegPipeline`), so results always match
between them. See `web/frontend/README.md` for the frontend-development
(hot-reload) workflow.
