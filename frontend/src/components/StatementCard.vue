<script setup lang="ts">
import type { StatementRead } from '@/types'
defineProps<{ statement: StatementRead }>()
</script>

<template>
  <div class="card px-4 py-3">
    <div class="flex items-center gap-2 mb-2">
      <span class="text-[11px] font-medium text-dim-300">{{ statement.role.replace(/_/g, ' ') }}</span>
      <span v-if="statement.vehicle" class="text-[11px] text-dim-500">· {{ statement.vehicle }}</span>
    </div>
    <p class="text-sm text-dim-300 leading-relaxed">{{ statement.text }}</p>
  </div>
</template>
