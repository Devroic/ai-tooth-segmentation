import { useEffect, useState } from 'react'
import { analyzeImage, checkHealth } from './api'
import UploadPanel from './components/UploadPanel'
import ImageViewer from './components/ImageViewer'
import Odontogram from './components/Odontogram'
import DetectionsTable from './components/DetectionsTable'
import './App.css'

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [conf, setConf] = useState(0.25)
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
      const data = await analyzeImage(file, conf)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const modelsMissing = health && !health.binary_model_loaded && !health.multiclass_model_loaded

  return (
    <div className="app">
      <header className="app-header">
        <h1>Tooth Segmentation</h1>
        <p>Upload a panoramic radiograph to segment and number each tooth.</p>
      </header>

      {modelsMissing && (
        <div className="banner banner-warning">
          No trained model weights found on the server. Train the models first (see README).
        </div>
      )}
      {error && <div className="banner banner-error">{error}</div>}

      <div className="app-layout">
        <aside className="app-sidebar">
          <UploadPanel
            onFileSelected={handleFileSelected}
            previewUrl={previewUrl}
            conf={conf}
            onConfChange={setConf}
            onAnalyze={handleAnalyze}
            loading={loading}
            disabled={!file || modelsMissing}
          />
        </aside>

        <main className="app-main">
          {result ? (
            <>
              <ImageViewer
                originalUrl={previewUrl}
                binaryOverlay={result.binary_overlay}
                multiclassOverlay={result.multiclass_overlay}
              />
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
              <p>Upload a radiograph and click "Analyze radiograph" to see results.</p>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
