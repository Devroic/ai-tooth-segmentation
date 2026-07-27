import { useEffect, useState } from 'react'
import { analyzeImage, checkHealth } from './api'
import UploadPanel from './components/UploadPanel'
import ImageViewer from './components/ImageViewer'
import Odontogram from './components/Odontogram'
import DetectionsTable from './components/DetectionsTable'
import './App.css'

// Not user-configurable - low so borderline teeth surface for review, see dental.js.
const ANALYZE_CONF = 0.15

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    checkHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  function handleFileSelected(f) {
    setFile(f)
    setPreviewUrl(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }

  async function handleAnalyze() {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const data = await analyzeImage(file, ANALYZE_CONF)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const modelUnavailable = health && !health.multiclass_model_loaded

  return (
    <div className="app">
      <header className="app-header">
        <h1>Tooth Segmentation</h1>
        <p>Upload a panoramic X-ray to identify and number each tooth.</p>
      </header>

      {modelUnavailable && (
        <div className="banner banner-warning">
          The tooth-identification model isn't available on the server right now. Train it first (see README).
        </div>
      )}
      {error && <div className="banner banner-error">{error}</div>}

      <div className="app-layout">
        <aside className="app-sidebar">
          <UploadPanel
            onFileSelected={handleFileSelected}
            previewUrl={previewUrl}
            onAnalyze={handleAnalyze}
            loading={loading}
            disabled={!file || modelUnavailable}
          />
        </aside>

        <main className="app-main">
          {result ? (
            <>
              <ImageViewer originalUrl={previewUrl} multiclassOverlay={result.multiclass_overlay} />
              <section className="panel">
                <h2>Odontogram</h2>
                <Odontogram detections={result.detections} />
              </section>
              <section className="panel">
                <h2>Detected teeth ({result.detections.length})</h2>
                <DetectionsTable detections={result.detections} />
              </section>
            </>
          ) : (
            <div className="empty-main">
              <p>Upload an X-ray and click "Analyze X-ray" to see results.</p>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
