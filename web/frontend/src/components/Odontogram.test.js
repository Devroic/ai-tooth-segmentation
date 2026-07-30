import { describe, expect, it } from 'vitest'
import { groupByFdi } from './Odontogram'

function detection(fdi, confidence) {
  return { FDI: fdi, Tooth: fdi, Group: 'incisor', Confidence: confidence }
}

describe('groupByFdi', () => {
  it('groups detections by FDI code', () => {
    const grouped = groupByFdi([detection('11', 0.9), detection('12', 0.8)])
    expect(Object.keys(grouped).sort()).toEqual(['11', '12'])
  })

  it('sorts each FDI group by confidence descending', () => {
    const grouped = groupByFdi([detection('11', 0.4), detection('11', 0.9), detection('11', 0.6)])
    expect(grouped['11'].map((d) => d.Confidence)).toEqual([0.9, 0.6, 0.4])
  })

  it('leaves a single detection per FDI untouched', () => {
    const grouped = groupByFdi([detection('11', 0.5)])
    expect(grouped['11']).toHaveLength(1)
  })
})
