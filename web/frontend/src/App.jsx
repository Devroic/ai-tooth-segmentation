import { useEffect, useState } from 'react'
import { analyzeImage, checkHealth } from './api'
import UploadPanel from './components/UploadPanel'
import ImageViewer from './components/ImageViewer'
import Odontogram from './components/Odontogram'
import DetectionsTable from './components/DetectionsTable'
import './App.css'

// Recall-first: 94% recall on the held-out test set at 0.15 vs 80% at the
// F1-optimal 0.41 (see scripts/model/calibrate_confidence.py) - the lost
// precision is acceptable since low-confidence hits are flagged for review
// rather than hidden (see dental.js). Not user-configurable.
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
        <p>
          Upload a panoramic dental X-ray and this tool will find each tooth
          and label it using FDI notation, the two-digit numbering system
          dentists use worldwide (for example, "11" is the upper right
          central incisor - the front-most tooth on the upper right).
        </p>
      </header>

      {modelUnavailable && (
        <div className="banner banner-warning">
          The tooth-identification model couldn't be loaded on the server.
          If you're running this yourself, check that the files under
          <code>models/</code> weren't removed or corrupted (see README).
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
              <div className="print-header">
                <h1>Tooth Segmentation Report</h1>
                <p>Generated {new Date().toLocaleString()}</p>
              </div>
              <div className="report-actions">
                <button className="report-button" onClick={() => window.print()}>
                  Download report (PDF)
                </button>
              </div>
              <ImageViewer originalUrl={previewUrl} multiclassOverlay={result.multiclass_overlay} />
              <section className="panel">
                <h2>Odontogram</h2>
                <p className="panel-subtitle">
                  A dental chart of every detected tooth, arranged the way a
                  dentist would view it. Each tile is colored by tooth type
                  and labeled with its FDI number (see legend below).
                </p>
                <Odontogram detections={result.detections} />
              </section>
              <section className="panel">
                <h2>Detected teeth ({result.detections.length})</h2>
                <p className="panel-subtitle">
                  "Confidence" is how sure the AI is about each tooth. Lower
                  values (shown in orange/red) or a "Review" badge mean that
                  result is less certain and worth a second look.
                </p>
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
