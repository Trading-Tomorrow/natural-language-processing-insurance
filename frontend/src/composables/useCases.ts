/**
 * Module-level shared state — all components calling useCases()
 * share the same reactive `cases` array without Pinia.
 */
import { ref } from 'vue'
import { casesApi } from '@/api/cases'
import type { CaseRead } from '@/types'

const cases = ref<CaseRead[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
let fetched = false

export function useCases() {
  async function fetchCases(force = false) {
    if (fetched && !force) return
    loading.value = true
    error.value = null
    try {
      cases.value = await casesApi.list()
      fetched = true
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to load cases'
    } finally {
      loading.value = false
    }
  }

  function upsertCase(updated: CaseRead) {
    const idx = cases.value.findIndex((c) => c.id === updated.id)
    if (idx >= 0) cases.value[idx] = updated
    else cases.value.unshift(updated)
  }

  return { cases, loading, error, fetchCases, upsertCase }
}
