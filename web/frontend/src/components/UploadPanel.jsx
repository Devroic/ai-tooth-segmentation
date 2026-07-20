import { useRef, useState } from 'react'
import './UploadPanel.css'

export default function UploadPanel({ onFileSelected, previewUrl, conf, onConfChange, onAnalyze, loading, disabled }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  function handleFiles(files) {
    if (files && files[0]) onFileSelected(files[0])
  }

  return (
    <div className="upload-panel">
      <div
        className={`drop-zone ${dragOver ? 'drag-over' : ''} ${previewUrl ? 'has-preview' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          handleFiles(e.dataTransfer.files)
        }}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Uploaded radiograph" className="drop-zone-preview" />
        ) : (
          <div className="drop-zone-prompt">
            <span className="drop-zone-icon">+</span>
            <p>Click or drag a panoramic radiograph here</p>
            <p className="drop-zone-hint">JPG or PNG</p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      <label className="conf-label">
        Confidence threshold: <strong>{conf.toFixed(2)}</strong>
        <input
          type="range"
          min="0.05"
          max="0.9"
          step="0.05"
          value={conf}
          onChange={(e) => onConfChange(Number(e.target.value))}
        />
      </label>

      <button className="analyze-button" onClick={onAnalyze} disabled={disabled || loading}>
        {loading ? 'Analyzing…' : 'Analyze radiograph'}
      </button>
    </div>
  )
}
