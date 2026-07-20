import { useState } from 'react'
import './ImageViewer.css'

export default function ImageViewer({ originalUrl, multiclassOverlay }) {
  const tabs = [
    { key: 'multiclass', label: 'Annotated X-ray', src: multiclassOverlay },
    { key: 'original', label: 'Original X-ray', src: originalUrl },
  ].filter((t) => t.src)

  const [active, setActive] = useState(tabs[0]?.key)
  const current = tabs.find((t) => t.key === active) || tabs[0]

  return (
    <div className="image-viewer">
      <div className="image-viewer-tabs">
        {tabs.map((t) => (
          <button
            key={t.key}
            className={`image-viewer-tab ${t.key === current?.key ? 'active' : ''}`}
            onClick={() => setActive(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="image-viewer-frame">
        {current && <img src={current.src} alt={current.label} />}
      </div>
    </div>
  )
}
