<script setup lang="ts">
import type { CaseRead } from '@/types'

const props = defineProps<{ caseData: CaseRead }>()

const isFraud = props.caseData.verdict === 'not_true'
const prob = props.caseData.probability_true ?? 0
const probPct = Math.round(prob * 1000) / 10
</script>

<template>
  <div class="space-y-3">

    <!-- Verdict -->
    <div class="card p-5 flex items-center justify-between gap-6">
      <div>
        <p class="text-xs text-dim-500 mb-1">Verdict</p>
        <p class="text-lg font-semibold" :class="isFraud ? 'text-risk' : 'text-ok'">
          {{ isFraud ? 'Suspected fraud' : 'Genuine claim' }}
        </p>
      </div>
      <div class="text-right flex-shrink-0">
        <p class="text-xs text-dim-500 mb-1">Probability true</p>
        <p class="text-2xl font-semibold font-mono" :class="isFraud ? 'text-risk' : 'text-ok'">
          {{ probPct }}<span class="text-sm">%</span>
        </p>
        <div class="w-20 h-0.5 rounded-full bg-base-700 mt-2 ml-auto">
          <div
            class="h-0.5 rounded-full"
            :class="isFraud ? 'bg-risk' : 'bg-ok'"
            :style="{ width: `${probPct}%` }"
          />
        </div>
      </div>
    </div>

    <!-- Reasoning -->
    <div v-if="caseData.reasoning" class="card p-4">
      <p class="text-xs text-dim-500 mb-2">Reasoning</p>
      <p class="text-sm text-dim-300 leading-relaxed">{{ caseData.reasoning }}</p>
    </div>

    <!-- Incongruences -->
    <div v-if="caseData.incongruences?.length" class="card p-4">
      <p class="text-xs text-dim-500 mb-3">Incongruences</p>
      <ul class="space-y-2">
        <li v-for="item in caseData.incongruences" :key="item"
          class="text-sm text-dim-300 flex items-start gap-2">
          <span class="mt-1.5 w-1 h-1 rounded-full bg-risk flex-shrink-0" />
          {{ item }}
        </li>
      </ul>
    </div>

  </div>
</template>
