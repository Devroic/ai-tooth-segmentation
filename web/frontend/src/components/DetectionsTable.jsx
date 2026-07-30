import { GROUP_COLORS, LOW_CONFIDENCE_THRESHOLD } from '../dental'
import './DetectionsTable.css'

export function confidenceTier(confidence) {
  if (confidence < LOW_CONFIDENCE_THRESHOLD) return 'low'
  if (confidence < 0.7) return 'medium'
  return 'high'
}

export default function DetectionsTable({ detections }) {
  if (detections.length === 0) {
    return <p className="empty-state">No teeth detected in this image.</p>
  }

  const fdiCounts = {}
  for (const d of detections) fdiCounts[d.FDI] = (fdiCounts[d.FDI] || 0) + 1

  const sorted = [...detections].sort((a, b) => a.FDI.localeCompare(b.FDI))

  return (
    <table className="detections-table">
      <thead>
        <tr>
          <th>FDI</th>
          <th>Tooth</th>
          <th>Group</th>
          <th>Confidence</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((d, i) => {
          const tier = confidenceTier(d.Confidence)
          const duplicate = fdiCounts[d.FDI] > 1
          return (
            <tr key={i}>
              <td className="fdi-cell">{d.FDI}</td>
              <td>{d.Tooth}</td>
              <td>
                <span className="group-badge" style={{ backgroundColor: GROUP_COLORS[d.Group] }}>
                  {d.Group}
                </span>
              </td>
              <td>
                <div className="confidence-bar-track">
                  <div className={`confidence-bar-fill tier-${tier}`} style={{ width: `${d.Confidence * 100}%` }} />
                </div>
                <span className="confidence-value">{(d.Confidence * 100).toFixed(0)}%</span>
              </td>
              <td>
                {duplicate && <span className="review-badge" title="Matched more than one detection - review">Review</span>}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
