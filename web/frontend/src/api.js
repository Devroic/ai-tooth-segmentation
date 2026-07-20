export async function analyzeImage(file, conf) {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(`/api/analyze?conf=${conf}`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

export async function checkHealth() {
  const res = await fetch('/api/health')
  if (!res.ok) throw new Error('Backend unreachable')
  return res.json()
}
