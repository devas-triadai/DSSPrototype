/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        dss: {
          bg: "#030712",
          panel: "#111827",
          card: "#1f2937",
          border: "#374151",
          accent: "#d4a017",
          success: "#22c55e",
          danger: "#ef4444",
          info: "#60a5fa",
          muted: "#9ca3af",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
