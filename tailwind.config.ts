import type { Config } from "tailwindcss";

export default {
  content: [
    // Use broader patterns to include more utilities
    "./app/**/*.{html,js,ts,jsx,tsx}",
    "./src/**/*.{js,ts,jsx,tsx,html,css}",
    // This file will act as a "utilities manifest"
    "./src/utilities.txt",
  ],
  theme: { extend: {} },
  plugins: [],
} satisfies Config;
