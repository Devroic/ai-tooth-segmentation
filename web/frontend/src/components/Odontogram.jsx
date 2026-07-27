import { GROUP_COLORS, LOW_CONFIDENCE_THRESHOLD, ODONTOGRAM_ROWS, toothDisplayName } from '../dental'
import './Odontogram.css'

// Groups duplicate FDI predictions (see README) so the chart flags the
// ambiguity; each group sorted by confidence descending.
function groupByFdi(detections) {
  const byFdi = {}
  for (const d of detections) {
    if (!byFdi[d.FDI]) byFdi[d.FDI] = []
    byFdi[d.FDI].push(d)
  }
  for (const hits of Object.values(byFdi)) {
    hits.sort((a, b) => b.Confidence - a.Confidence)
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
            const lowConfidence = best && best.Confidence < LOW_CONFIDENCE_THRESHOLD

            const tooltipParts = best
              ? [`${toothDisplayName(fdi)} — ${(best.Confidence * 100).toFixed(0)}% confidence`]
              : [`${toothDisplayName(fdi)} — not detected`]
            if (duplicate) {
              tooltipParts.push(`also matched ${hits.length - 1} other detection${hits.length > 2 ? 's' : ''} — review`)
            }
            if (lowConfidence) {
              tooltipParts.push('low confidence — review')
            }

            return (
              <div
                key={fdi}
                className={[
                  'tooth-cell',
                  best ? 'detected' : 'empty',
                  duplicate ? 'duplicate' : '',
                  lowConfidence ? 'low-confidence' : '',
                ].join(' ').trim()}
                style={best ? { backgroundColor: color, opacity: 0.35 + best.Confidence * 0.65 } : undefined}
                title={tooltipParts.join(' — ')}
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
        <span className="legend-item">
          <span className="legend-swatch legend-swatch-low-confidence" />
          low confidence
        </span>
        <span className="legend-item">
          <span className="legend-swatch legend-swatch-duplicate">!</span>
          needs review (duplicate)
        </span>
      </div>
    </div>
  )
}
