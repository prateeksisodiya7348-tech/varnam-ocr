import { defineConfig } from "vitest/config";
import path from "node:path";

/**
 * Vitest configuration.
 *
 * Deliberately does not use jsdom: every test in this package covers pure
 * functions (export formatting, box scaling). Browser behaviour is covered by
 * the Playwright suite instead, which is the only way to test a real upload.
 */
export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
