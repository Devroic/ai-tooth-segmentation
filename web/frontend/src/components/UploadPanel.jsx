import { useRef, useState } from 'react'
import './UploadPanel.css'

export default function UploadPanel({ onFileSelected, previewUrl, onAnalyze, loading, disabled }) {
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
            <p>Click or drag a panoramic X-ray here</p>
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

      <button className="analyze-button" onClick={onAnalyze} disabled={disabled || loading}>
        {loading ? 'Analyzing…' : 'Analyze X-ray'}
      </button>
    </div>
  )
}
