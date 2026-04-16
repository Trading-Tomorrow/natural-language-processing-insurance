<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import L from 'leaflet'

const props = defineProps<{ location: string }>()

const mapEl = ref<HTMLDivElement | null>(null)
const map = ref<L.Map | null>(null)
const tileLayer = ref<L.TileLayer | null>(null)
const marker = ref<L.Marker | null>(null)
const circle = ref<L.Circle | null>(null)
let resizeObserver: ResizeObserver | null = null
let intersectionObserver: IntersectionObserver | null = null
const mapReady = ref(false)
let idleTimeout: number | null = null
const loading = ref(false)
const error = ref<string | null>(null)

async function geocodeAndRender(query: string) {
    if (!map.value) return
  if (!query?.trim()) {
    error.value = 'Location is empty.'
    return
  }
  loading.value = true
  error.value = null
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(query)}`,
    )
    if (!res.ok) throw new Error('Geocoding failed')
    const data = (await res.json()) as Array<{ lat: string; lon: string }>
    if (!data.length) {
      error.value = 'No location match found.'
      return
    }
    const lat = Number(data[0].lat)
    const lon = Number(data[0].lon)
    const latLng: L.LatLngExpression = [lat, lon]
    map.value.setView(latLng, 14)

    if (marker.value) marker.value.setLatLng(latLng)
    else marker.value = L.marker(latLng).addTo(map.value)

    if (circle.value) circle.value.setLatLng(latLng)
    else circle.value = L.circle(latLng, { radius: 150, color: '#f97316' }).addTo(map.value)
    await nextTick()
    map.value.invalidateSize()
    map.value.once('idle', () => map.value?.invalidateSize())
    if (idleTimeout) window.clearTimeout(idleTimeout)
    idleTimeout = window.setTimeout(() => map.value?.invalidateSize(), 350)
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Geocoding failed'
  } finally {
    loading.value = false
  }
}

function initMap() {
  if (!mapEl.value || map.value) return
  map.value = L.map(mapEl.value, { zoomControl: true, attributionControl: true })
  tileLayer.value = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors',
  }).addTo(map.value)
  tileLayer.value.on('load', () => map.value?.invalidateSize())
  mapReady.value = true
  geocodeAndRender(props.location)
  nextTick(() => map.value?.invalidateSize())
  setTimeout(() => map.value?.invalidateSize(), 150)
  setTimeout(() => map.value?.invalidateSize(), 600)
  requestAnimationFrame(() => map.value?.invalidateSize())
  requestAnimationFrame(() => map.value?.invalidateSize())
  resizeObserver = new ResizeObserver(() => map.value?.invalidateSize())
  resizeObserver.observe(mapEl.value)
  window.addEventListener('resize', handleWindowResize)
}

function handleWindowResize() {
  map.value?.invalidateSize()
}

onMounted(() => {
  if (!mapEl.value) return
  intersectionObserver = new IntersectionObserver(
    (entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        initMap()
        intersectionObserver?.disconnect()
      }
    },
    { threshold: 0.1 },
  )
  intersectionObserver.observe(mapEl.value)
})

onBeforeUnmount(() => {
  if (resizeObserver && mapEl.value) resizeObserver.unobserve(mapEl.value)
  resizeObserver = null
  intersectionObserver?.disconnect()
  intersectionObserver = null
  window.removeEventListener('resize', handleWindowResize)
  if (idleTimeout) window.clearTimeout(idleTimeout)
  idleTimeout = null
})

watch(
  () => props.location,
  (next) => {
    if (!map.value || !mapReady.value) return
    geocodeAndRender(next)
  },
)
</script>

<template>
  <div class="space-y-2">
    <div ref="mapEl" class="h-72 w-full min-w-0 rounded-md overflow-hidden border border-base-700 bg-base-900" />
    <p v-if="loading" class="text-xs text-dim-500">Resolving location…</p>
    <p v-else-if="error" class="text-xs text-risk/80">{{ error }}</p>
    <p v-else class="text-[10px] text-dim-500">Approximate location (150m radius)</p>
  </div>
</template>
