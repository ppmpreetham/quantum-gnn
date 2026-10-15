import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8765' } },
  preview: { proxy: { '/api': 'http://127.0.0.1:8765' } },
  build: { chunkSizeWarningLimit: 1200, rollupOptions: { output: { manualChunks: { three: ['three', '@react-three/fiber', '@react-three/drei'], effects: ['@react-three/postprocessing', 'postprocessing'] } } } },
})
