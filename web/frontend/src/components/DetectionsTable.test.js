import { describe, expect, it } from 'vitest'
import { confidenceTier } from './DetectionsTable'
import { LOW_CONFIDENCE_THRESHOLD } from '../dental'

describe('confidenceTier', () => {
  it('is "low" below the low-confidence threshold', () => {
    expect(confidenceTier(LOW_CONFIDENCE_THRESHOLD - 0.01)).toBe('low')
  })

  it('is "medium" between the low threshold and 0.7', () => {
    expect(confidenceTier(LOW_CONFIDENCE_THRESHOLD)).toBe('medium')
    expect(confidenceTier(0.69)).toBe('medium')
  })

  it('is "high" at or above 0.7', () => {
    expect(confidenceTier(0.7)).toBe('high')
    expect(confidenceTier(1.0)).toBe('high')
  })
})
