import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const api = `http://127.0.0.1:${process.env.FLEET_API_PORT ?? '8765'}`

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': api } },
  preview: { proxy: { '/api': api } },
  build: {
    chunkSizeWarningLimit: 1200,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            { name: 'three', test: /node_modules[\\/](three[\\/](build|src)|@react-three[\\/](fiber|drei))[\\/]/ },
            { name: 'effects', test: /node_modules[\\/](@react-three[\\/]postprocessing|postprocessing|n8ao)[\\/]/ },
          ],
        },
      },
    },
  },
})
