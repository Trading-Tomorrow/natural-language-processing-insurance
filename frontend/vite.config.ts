import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiBase = env.VITE_API_BASE_URL || 'http://localhost:8000'

  return {
    plugins: [vue()],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
    server: {
      port: 5173,
      proxy: {
        '/cases': {
          target: apiBase,
          changeOrigin: true,
          bypass(req) {
            // Browser navigation (F5, direct URL) → serve the SPA
            // API calls from fetch() → proxy to backend
            if (req.headers.accept?.includes('text/html')) {
              return '/index.html'
            }
          },
        },
        '/static': {
          target: apiBase,
          changeOrigin: true,
        },
      },
    },
  }
})
