<script setup lang="ts">
import { useRouter } from 'vue-router'
const router = useRouter()
</script>

<template>
  <div class="flex flex-col items-center justify-center h-full">
    <p class="text-sm text-dim-500 mb-4">Select a case or create a new one.</p>
    <button @click="router.push('/cases/new')" class="btn-primary text-xs">
      <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15"/>
      </svg>
      New case
    </button>
  </div>
</template>
