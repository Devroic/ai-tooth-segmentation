import { GROUP_COLORS } from '../dental'
import './DetectionsTable.css'

export default function DetectionsTable({ detections }) {
  if (detections.length === 0) {
    return <p className="empty-state">No teeth detected in this image.</p>
  }

  const sorted = [...detections].sort((a, b) => a.FDI.localeCompare(b.FDI))

  return (
    <table className="detections-table">
      <thead>
        <tr>
          <th>FDI</th>
          <th>Tooth</th>
          <th>Group</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((d, i) => (
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
                <div className="confidence-bar-fill" style={{ width: `${d.Confidence * 100}%` }} />
              </div>
              <span className="confidence-value">{(d.Confidence * 100).toFixed(0)}%</span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
