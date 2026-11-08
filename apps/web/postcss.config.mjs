/**
 * PostCSS config for Tailwind CSS v4.
 *
 * v4 replaced the `tailwind.config.js` + `@tailwindcss/postcss` pair with a
 * single plugin and CSS-first configuration (`@theme` in globals.css). There is
 * deliberately no tailwind.config.ts in this repo - that file would be dead
 * config that misleads a reader.
 */
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
