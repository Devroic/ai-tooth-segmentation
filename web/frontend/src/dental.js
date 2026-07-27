// Mirrors src/tooth_seg/taxonomy.py - keep in sync.

export const GROUP_COLORS = {
  incisor: '#4285F4',
  canine: '#DB4437',
  premolar: '#F4B400',
  molar: '#0F9D58',
}

// Below this, flag for review instead of filtering out.
export const LOW_CONFIDENCE_THRESHOLD = 0.5

const ORDINAL_NAME = {
  1: 'Central Incisor',
  2: 'Lateral Incisor',
  3: 'Canine',
  4: 'First Premolar',
  5: 'Second Premolar',
  6: 'First Molar',
  7: 'Second Molar',
  8: 'Third Molar',
}

export function fdiToGroup(fdi) {
  const position = Number(fdi[1])
  if (position <= 2) return 'incisor'
  if (position === 3) return 'canine'
  if (position <= 5) return 'premolar'
  return 'molar'
}

export function toothDisplayName(fdi) {
  const quadrant = fdi[0]
  const position = Number(fdi[1])
  const arch = quadrant === '1' || quadrant === '2' ? 'Upper' : 'Lower'
  const side = quadrant === '1' || quadrant === '4' ? 'Right' : 'Left'
  return `${arch} ${side} ${ORDINAL_NAME[position]} (${fdi})`
}

// Radiograph convention: patient's right on the left.
export const ODONTOGRAM_ROWS = [
  ['18', '17', '16', '15', '14', '13', '12', '11', '21', '22', '23', '24', '25', '26', '27', '28'],
  ['48', '47', '46', '45', '44', '43', '42', '41', '31', '32', '33', '34', '35', '36', '37', '38'],
]
