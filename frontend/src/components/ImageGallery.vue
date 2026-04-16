<script setup lang="ts">
import { imageUrl } from '@/api/cases'
import type { ImageRead } from '@/types'

defineProps<{ images: ImageRead[] }>()
</script>

<template>
  <div v-if="images.length" class="grid grid-cols-3 gap-2">
    <div v-for="img in images" :key="img.id"
      class="relative rounded-lg overflow-hidden aspect-square bg-slate-800">
      <img :src="imageUrl(img.file_path)" :alt="img.damage_types?.join(', ') || 'Accident photo'"
        class="w-full h-full object-cover" />
      <!-- Damage tags overlay -->
      <div v-if="img.damage_types?.length"
        class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-2 py-1.5">
        <div class="flex flex-wrap gap-1">
          <span v-for="t in img.damage_types" :key="t"
            class="text-[9px] font-medium bg-red-900/80 text-red-300 px-1 py-0.5 rounded">
            {{ t }}
          </span>
        </div>
      </div>
    </div>
  </div>
  <p v-else class="text-xs text-slate-600 italic">No photos uploaded yet.</p>
</template>
