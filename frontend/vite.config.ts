import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  build: {
    rollupOptions: {
      // 多页：主应用 + 独立 shader playground。相对路径相对项目根解析，
      // 避免引入 node:path/__dirname（未安装 @types/node）。
      input: {
        main: 'index.html',
        playground: 'playground.html',
      },
    },
  },
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/png-shader': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
