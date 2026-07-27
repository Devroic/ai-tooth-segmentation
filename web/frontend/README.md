# Tooth Segmentation — Web UI

React (Vite) frontend for the clinical-facing tooth segmentation app. Talks
to the FastAPI backend in `../backend`, which wraps the same
`tooth_seg.inference.pipeline` used by the developer-facing Gradio app
(`app/app.py`) - all three call identical inference code, so results always
match.

## Everyday use (one command, no Node/npm needed)

The backend serves the already-built frontend directly:

```
# from the project root, same .venv as everything else
.venv\Scripts\python.exe -m uvicorn web.backend.main:app --port 8000
```

Open **http://127.0.0.1:8000**. That's it - one terminal, one URL, exactly
like the Gradio app.

(This works because `dist/` - the prebuilt frontend - is committed to the
repo and rebuilt via `npm run build` whenever the frontend changes; see
below. After a fresh `git pull` you don't need Node/npm at all just to use
the app.)

## Frontend development (editing the UI)

If you're changing frontend code, you want hot-reload instead of rebuilding
by hand each time - that needs two terminals:

```
# 1. Backend (project root)
.venv\Scripts\python.exe -m uvicorn web.backend.main:app --reload --reload-dir web/backend --reload-dir src --port 8000

# 2. Frontend (this directory)
npm install
npm run dev
```

Open the URL Vite prints (`http://localhost:5173`). Its dev server proxies
`/api/*` to the backend on port 8000 (see `vite.config.js`), so no CORS
setup is needed. Once you're done editing, run `npm run build` so the
one-command production path above picks up your changes.

## Build for production

```
npm run build
```

Outputs static files to `dist/` (tracked in git - commit the rebuilt
`dist/` alongside any frontend source change), which `web/backend/main.py`
serves automatically if present.
