import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Консоль обслуживается API по пути /console (app.main монтирует dist).
// В dev-режиме Vite проксирует REST и WebSocket на uvicorn - без CORS.
export default defineConfig({
  base: "/console/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", ws: true },
    },
  },
});
