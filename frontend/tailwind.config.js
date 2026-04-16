/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Near-black monochrome system
        base: {
          950: '#08080a',
          900: '#0f0f12',
          800: '#161619',
          700: '#1e1e22',
          600: '#28282d',
          500: '#36363d',
        },
        // Text scale
        dim: {
          100: '#f4f4f5',
          300: '#a1a1aa',
          500: '#52525b',
          700: '#27272a',
        },
        // Functional colours only
        ok:   '#22c55e',
        risk: '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
