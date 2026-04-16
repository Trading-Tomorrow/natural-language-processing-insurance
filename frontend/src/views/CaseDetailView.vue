<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { casesApi } from '@/api/cases'
import { useCases } from '@/composables/useCases'
import type { CaseRead } from '@/types'
import StatementCard from '@/components/StatementCard.vue'
import ImageGallery from '@/components/ImageGallery.vue'
import AnalysisResult from '@/components/AnalysisResult.vue'
import AppSpinner from '@/components/AppSpinner.vue'
import AppAlert from '@/components/AppAlert.vue'

const props = defineProps<{ id: string }>()
const { upsertCase } = useCases()

const caseData = ref<CaseRead | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const analyzing = ref(false)
const analyzeError = ref<string | null>(null)
const uploadLoading = ref(false)
const uploadError = ref<string | null>(null)

function formatDate(iso: string) {
  return new Date(iso).toLocaleString('pt-PT', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

async function loadCase() {
  loading.value = true
  error.value = null
  try {
    caseData.value = await casesApi.get(Number(props.id))
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load case'
  } finally {
    loading.value = false
  }
}

async function analyze() {
  if (!caseData.value) return
  analyzing.value = true
  analyzeError.value = null
  try {
    const res = await casesApi.analyze(caseData.value.id)
    caseData.value = res.case
    upsertCase(res.case)
  } catch (e) {
    analyzeError.value = e instanceof Error ? e.message : 'Analysis failed'
  } finally {
    analyzing.value = false
  }
}

async function handleFileUpload(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length || !caseData.value) return
  uploadLoading.value = true
  uploadError.value = null
  try {
    const imgs = await casesApi.uploadImages(caseData.value.id, Array.from(input.files))
    caseData.value = { ...caseData.value, images: [...caseData.value.images, ...imgs] }
  } catch (e) {
    uploadError.value = e instanceof Error ? e.message : 'Upload failed'
  } finally {
    uploadLoading.value = false
    input.value = ''
  }
}

onMounted(loadCase)
watch(() => props.id, loadCase)
</script>

<template>
  <div v-if="loading" class="flex justify-center items-center h-full">
    <AppSpinner class="w-5 h-5 text-dim-500" />
  </div>

  <div v-else-if="error" class="max-w-2xl mx-auto px-8 py-12">
    <AppAlert :message="error" />
  </div>

  <div v-else-if="caseData" class="max-w-2xl mx-auto">

    <!-- ── Header ────────────────────────────────────────────────── -->
    <div class="px-8 pt-10 pb-8 border-b border-base-800">
      <div class="flex items-start justify-between gap-4">
        <div>
          <p class="text-[10px] font-mono text-dim-500 mb-2">{{ caseData.claim_id }}</p>
          <h1 class="text-xl font-semibold text-dim-100 leading-tight">{{ caseData.incident_type }}</h1>
          <p class="text-sm text-dim-500 mt-1.5">{{ caseData.location }}</p>
          <p class="text-xs text-dim-700 mt-1">{{ formatDate(caseData.created_at) }}</p>
        </div>
        <span
          v-if="caseData.verdict"
          class="flex-shrink-0 text-xs font-medium px-2.5 py-1 rounded-md border mt-1"
          :class="caseData.verdict === 'true'
            ? 'text-ok border-ok/30 bg-ok/5'
            : 'text-risk border-risk/30 bg-risk/5'"
        >
          {{ caseData.verdict === 'true' ? 'Genuine' : 'Fraud' }}
        </span>
      </div>
    </div>

    <div class="px-8 py-8 space-y-10">

      <!-- Detected Damages -->
      <div v-if="caseData.detected_damages.length">
        <p class="section-title">Detected damages</p>
        <div class="flex flex-wrap gap-1.5">
          <span v-for="d in caseData.detected_damages" :key="d"
            class="text-xs text-dim-300 border border-base-600 bg-base-800 px-2.5 py-1 rounded-md">
            {{ d }}
          </span>
        </div>
      </div>

      <!-- Statements -->
      <div>
        <p class="section-title">Statements · {{ caseData.statements.length }}</p>
        <div class="space-y-2">
          <StatementCard v-for="s in caseData.statements" :key="s.id" :statement="s" />
          <p v-if="!caseData.statements.length" class="text-sm text-dim-500">No statements.</p>
        </div>
      </div>

      <!-- Photos -->
      <div>
        <p class="section-title">Accident photos</p>
        <ImageGallery :images="caseData.images" />
        <AppAlert v-if="uploadError" :message="uploadError" class="mt-2" />
        <label class="mt-2 flex items-center justify-center gap-2 w-full h-14 rounded-md border border-dashed border-base-600 hover:border-base-500 cursor-pointer transition-colors text-dim-500 hover:text-dim-300 text-xs">
          <AppSpinner v-if="uploadLoading" class="w-3.5 h-3.5" />
          <template v-else>
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
            </svg>
            Upload photos
          </template>
          <input type="file" accept="image/*" multiple class="hidden" @change="handleFileUpload" />
        </label>
      </div>

      <!-- Analysis -->
      <div>
        <p class="section-title">Analysis</p>

        <AnalysisResult v-if="caseData.verdict !== null" :case-data="caseData" />

        <div v-else class="card p-6 text-center">
          <p class="text-sm text-dim-500 mb-4">No analysis yet.</p>
          <AppAlert v-if="analyzeError" :message="analyzeError" class="mb-4 text-left" />
          <button @click="analyze" :disabled="analyzing" class="btn-primary">
            <AppSpinner v-if="analyzing" class="w-3.5 h-3.5" />
            <svg v-else class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z"/>
            </svg>
            {{ analyzing ? 'Running…' : 'Run analysis' }}
          </button>
        </div>

        <div v-if="caseData.verdict !== null" class="mt-4">
          <AppAlert v-if="analyzeError" :message="analyzeError" class="mb-2" />
          <button @click="analyze" :disabled="analyzing" class="btn-secondary text-xs">
            <AppSpinner v-if="analyzing" class="w-3 h-3" />
            <svg v-else class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99"/>
            </svg>
            {{ analyzing ? 'Running…' : 'Re-run' }}
          </button>
        </div>
      </div>

    </div>
  </div>
</template>
