<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterView, useRoute, useRouter } from 'vue-router'
import { useCases } from '@/composables/useCases'
import AppSpinner from '@/components/AppSpinner.vue'

const route = useRoute()
const router = useRouter()
const { cases, loading, error, fetchCases } = useCases()

const query = ref('')

const filtered = computed(() => {
  const q = query.value.toLowerCase()
  if (!q) return cases.value
  return cases.value.filter(
    (c) =>
      c.claim_id.toLowerCase().includes(q) ||
      c.incident_type.toLowerCase().includes(q) ||
      c.location.toLowerCase().includes(q),
  )
})

function isActive(id: number) {
  return route.params.id === String(id)
}

onMounted(() => fetchCases())
</script>

<template>
  <div class="flex h-full overflow-hidden">

    <!-- ── Sidebar ────────────────────────────────────────────────── -->
    <aside class="w-64 flex-shrink-0 flex flex-col bg-base-900 border-r border-base-700">

      <!-- Brand -->
      <div class="h-12 px-4 flex items-center justify-between border-b border-base-700">
        <span class="text-sm font-semibold text-dim-100 tracking-tight">ClaimGuard</span>
        <span class="text-[10px] text-dim-500 font-mono">v0.1</span>
      </div>

      <!-- Search + New -->
      <div class="px-3 py-3 border-b border-base-700 space-y-2">
        <div class="relative">
          <svg class="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-dim-500 pointer-events-none" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 105 11a6 6 0 0012 0z"/>
          </svg>
          <input
            v-model="query"
            type="search"
            placeholder="Search…"
            class="w-full bg-base-800 text-dim-100 text-xs pl-8 pr-3 py-1.5 rounded-md border border-base-600 focus:border-base-500 focus:outline-none placeholder-dim-700"
          />
        </div>
        <button @click="router.push('/cases/new')" class="btn-primary w-full justify-center text-xs py-1.5">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15"/>
          </svg>
          New case
        </button>
      </div>

      <!-- List -->
      <div class="flex-1 overflow-y-auto">
        <div v-if="loading" class="flex justify-center py-8">
          <AppSpinner class="w-4 h-4 text-dim-500" />
        </div>

        <div v-else-if="error" class="px-4 py-4 text-xs text-risk/70">
          {{ error }}
          <button @click="fetchCases(true)" class="underline ml-1 text-dim-300">Retry</button>
        </div>

        <template v-else>
          <button
            v-for="c in filtered"
            :key="c.id"
            @click="router.push(`/cases/${c.id}`)"
            class="relative w-full text-left px-4 py-3 border-b border-base-800 hover:bg-base-800/60 transition-colors"
            :class="isActive(c.id) ? 'bg-base-800' : ''"
          >
            <span v-if="isActive(c.id)" class="absolute left-0 top-2 bottom-2 w-px bg-dim-100 rounded-full" />

            <div class="flex items-start justify-between gap-2">
              <div class="min-w-0">
                <p class="text-[10px] font-mono text-dim-500 truncate leading-none mb-1">{{ c.claim_id }}</p>
                <p class="text-xs font-medium truncate leading-snug"
                  :class="isActive(c.id) ? 'text-dim-100' : 'text-dim-300'">
                  {{ c.incident_type }}
                </p>
                <p class="text-[10px] text-dim-500 truncate mt-0.5">{{ c.location }}</p>
              </div>

              <span
                v-if="c.verdict"
                class="flex-shrink-0 w-1.5 h-1.5 rounded-full mt-1.5"
                :class="c.verdict === 'true' ? 'bg-ok' : 'bg-risk'"
                :title="c.verdict === 'true' ? 'Genuine' : 'Fraud'"
              />
            </div>
          </button>

          <p v-if="filtered.length === 0" class="px-4 py-8 text-xs text-dim-500 text-center">
            No cases found.
          </p>
        </template>
      </div>

      <!-- Footer -->
      <div class="px-4 py-3 border-t border-base-700">
        <p class="text-[10px] text-dim-700">{{ cases.length }} cases loaded</p>
      </div>

    </aside>

    <!-- ── Main ──────────────────────────────────────────────────── -->
    <main class="flex-1 overflow-y-auto bg-base-950">
      <RouterView />
    </main>

  </div>
</template>
