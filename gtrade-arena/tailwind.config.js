/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Custom colors for strategies
        'signal-follower': '#9b59b6',
        'cooperator': '#27ae60',
        'defector': '#e74c3c',
        'tit-for-tat': '#3498db',
        'benchmark': '#2c3e50',
      }
    },
  },
  plugins: [],
}
