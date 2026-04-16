<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { casesApi } from '@/api/cases'
import { useCases } from '@/composables/useCases'
import AppSpinner from '@/components/AppSpinner.vue'
import AppAlert from '@/components/AppAlert.vue'

const router = useRouter()
const { upsertCase } = useCases()

const ROLES = ['driver', 'passenger', 'witness', 'insurance_adjuster', 'third_party_driver']

interface DraftStatement { role: string; vehicle: string; text: string }

const form = reactive({ location: '', incident_type: '' })
const statements = ref<DraftStatement[]>([{ role: 'driver', vehicle: '', text: '' }])
const loading = ref(false)
const error = ref<string | null>(null)

function addStatement() {
  statements.value.push({ role: 'driver', vehicle: '', text: '' })
}
function removeStatement(i: number) {
  if (statements.value.length > 1) statements.value.splice(i, 1)
}

function generateClaimId() {
  const d = new Date()
  const ymd = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`
  const rand = Math.random().toString(36).slice(2, 6).toUpperCase()
  return `CASE-${ymd}-${rand}`
}

async function submit() {
  if (!form.location || !form.incident_type) {
    error.value = 'Location and incident type are required.'
    return
  }
  loading.value = true
  error.value = null
  try {
    const created = await casesApi.create({
      claim_id: generateClaimId(),
      location: form.location,
      incident_type: form.incident_type,
      detected_damages: [],
      statements: statements.value
        .filter((s) => s.text.trim())
        .map((s) => ({ role: s.role, vehicle: s.vehicle || null, text: s.text })),
    })
    upsertCase(created)
    router.push(`/cases/${created.id}`)
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to create case'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="max-w-2xl mx-auto px-8 py-10">

    <div class="mb-8">
      <h1 class="text-xl font-semibold text-dim-100">New case</h1>
      <p class="text-sm text-dim-500 mt-1">Submit a claim for fraud analysis.</p>
    </div>

    <AppAlert v-if="error" :message="error" class="mb-6" />

    <form @submit.prevent="submit" class="space-y-6">

      <!-- Incident -->
      <div class="card p-5 space-y-4">
        <p class="text-xs text-dim-500">Incident</p>

        <div>
          <label class="field-label">Incident type <span class="text-risk">*</span></label>
          <input v-model="form.incident_type" type="text" required placeholder="e.g. Rear-end collision" class="inp" />
        </div>

        <div>
          <label class="field-label">Location <span class="text-risk">*</span></label>
          <input v-model="form.location" type="text" required placeholder="e.g. A1 motorway, km 45, Lisbon" class="inp" />
        </div>
      </div>

      <!-- Statements -->
      <div class="card p-5 space-y-3">
        <div class="flex items-center justify-between">
          <p class="text-xs text-dim-500">Statements</p>
          <button type="button" @click="addStatement" class="btn-secondary text-xs py-1 px-2">
            <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15"/>
            </svg>
            Add
          </button>
        </div>

        <div v-for="(stmt, i) in statements" :key="i"
          class="rounded-md bg-base-800 border border-base-700 p-3.5 space-y-2.5">
          <div class="flex items-center gap-2">
            <select v-model="stmt.role"
              class="bg-base-900 text-dim-300 text-xs px-2 py-1.5 rounded-md border border-base-600 focus:border-base-500 focus:outline-none">
              <option v-for="r in ROLES" :key="r" :value="r">{{ r.replace(/_/g, ' ') }}</option>
            </select>
            <input v-model="stmt.vehicle" type="text" placeholder="Vehicle (optional)"
              class="flex-1 bg-base-900 text-dim-300 text-xs px-2.5 py-1.5 rounded-md border border-base-600 focus:border-base-500 focus:outline-none placeholder-dim-700" />
            <button type="button" @click="removeStatement(i)" :disabled="statements.length === 1"
              class="text-dim-700 hover:text-risk transition-colors disabled:opacity-30 disabled:cursor-not-allowed p-1">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
              </svg>
            </button>
          </div>
          <textarea v-model="stmt.text" rows="3" :placeholder="`Statement…`"
            class="w-full bg-base-900 text-dim-300 text-sm px-3 py-2 rounded-md border border-base-600 focus:border-base-500 focus:outline-none placeholder-dim-700 resize-none leading-relaxed"></textarea>
        </div>
      </div>

      <!-- Actions -->
      <div class="flex items-center gap-3">
        <button type="submit" :disabled="loading" class="btn-primary">
          <AppSpinner v-if="loading" class="w-3.5 h-3.5" />
          {{ loading ? 'Submitting…' : 'Submit case' }}
        </button>
        <button type="button" @click="router.back()" class="btn-secondary">Cancel</button>
      </div>
    </form>
  </div>
</template>
