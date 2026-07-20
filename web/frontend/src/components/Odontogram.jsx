import { GROUP_COLORS, ODONTOGRAM_ROWS, toothDisplayName } from '../dental'
import './Odontogram.css'

// Detections for the same FDI code (a known model limitation - see README)
// are grouped so the chart surfaces the ambiguity instead of hiding it.
function groupByFdi(detections) {
  const byFdi = {}
  for (const d of detections) {
    if (!byFdi[d.FDI]) byFdi[d.FDI] = []
    byFdi[d.FDI].push(d)
  }
  return byFdi
}

export default function Odontogram({ detections }) {
  const byFdi = groupByFdi(detections)

  return (
    <div className="odontogram">
      {ODONTOGRAM_ROWS.map((row, rowIdx) => (
        <div className="odontogram-row" key={rowIdx}>
          {row.map((fdi) => {
            const hits = byFdi[fdi] || []
            const best = hits[0]
            const color = best ? GROUP_COLORS[best.Group] : null
            const duplicate = hits.length > 1

            return (
              <div
                key={fdi}
                className={`tooth-cell ${best ? 'detected' : 'empty'} ${duplicate ? 'duplicate' : ''}`}
                style={best ? { backgroundColor: color, opacity: 0.35 + best.Confidence * 0.65 } : undefined}
                title={
                  best
                    ? `${toothDisplayName(fdi)} — ${(best.Confidence * 100).toFixed(0)}% confidence` +
                      (duplicate ? ` (also matched ${hits.length - 1} other detection${hits.length > 2 ? 's' : ''} — review)` : '')
                    : `${toothDisplayName(fdi)} — not detected`
                }
              >
                <span className="tooth-fdi">{fdi}</span>
                {duplicate && <span className="tooth-flag">!</span>}
              </div>
            )
          })}
        </div>
      ))}
      <div className="odontogram-legend">
        {Object.entries(GROUP_COLORS).map(([group, color]) => (
          <span className="legend-item" key={group}>
            <span className="legend-swatch" style={{ backgroundColor: color }} />
            {group}
          </span>
        ))}
        <span className="legend-item">
          <span className="legend-swatch legend-swatch-empty" />
          not detected
        </span>
      </div>
    </div>
  )
}
