# Tooth Segmentation — Web UI

React (Vite) frontend for the clinical-facing tooth segmentation app. Talks
to the FastAPI backend in `../backend`, which wraps the same
`tooth_seg.inference.pipeline` used by the developer-facing Gradio app
(`app/app.py`) - both UIs call identical inference code, so results always
match.

## Run locally

```
# 1. Backend (from the project root, same .venv as everything else)
.venv\Scripts\python.exe -m uvicorn web.backend.main:app --reload --port 8000

# 2. Frontend (from this directory)
npm install
npm run dev
```

Then open the URL Vite prints (default `http://localhost:5173`). The dev
server proxies `/api/*` to the backend on port 8000 (see `vite.config.js`),
so no CORS setup is needed locally.

## Build for production

```
npm run build
```

Outputs static files to `dist/`. Serve them with any static file server, or
have the FastAPI backend serve them directly (not wired up yet).
