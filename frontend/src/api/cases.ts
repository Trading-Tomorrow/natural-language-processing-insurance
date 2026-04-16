import type { AnalyzeResponse, CaseCreate, CaseRead, ImageRead } from '@/types'

const BASE = import.meta.env.VITE_API_BASE_URL ?? ''

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const message =
      body?.detail
        ? Array.isArray(body.detail)
          ? body.detail.map((d: { msg: string }) => d.msg).join(', ')
          : String(body.detail)
        : res.statusText
    throw new Error(message)
  }
  return res.json() as Promise<T>
}

export const casesApi = {
  list(): Promise<CaseRead[]> {
    return request<CaseRead[]>('/cases')
  },

  get(id: number): Promise<CaseRead> {
    return request<CaseRead>(`/cases/${id}`)
  },

  create(body: CaseCreate): Promise<CaseRead> {
    return request<CaseRead>('/cases', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  },

  uploadImages(caseId: number, files: File[], statementId?: number): Promise<ImageRead[]> {
    const fd = new FormData()
    for (const f of files) fd.append('files', f)
    const qs = statementId != null ? `?statement_id=${statementId}` : ''
    return request<ImageRead[]>(`/cases/${caseId}/images${qs}`, {
      method: 'POST',
      body: fd,
    })
  },

  analyze(caseId: number): Promise<AnalyzeResponse> {
    return request<AnalyzeResponse>(`/cases/${caseId}/analyze`, { method: 'POST' })
  },
}

export function imageUrl(filePath: string): string {
  if (!filePath) return ''
  if (filePath.startsWith('http')) return filePath
  return `${BASE}/${filePath.replace(/^\//, '')}`
}
