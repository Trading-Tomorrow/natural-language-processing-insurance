<script setup lang="ts">
defineProps<{ message: string }>()
</script>

<template>
  <div class="rounded-md border border-risk/20 bg-risk/5 px-3 py-2.5 text-sm text-risk/80">
    {{ message }}
  </div>
</template>
