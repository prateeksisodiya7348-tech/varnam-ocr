import type { NextConfig } from "next";

/**
 * The app has two deployment targets, and they need slightly different output.
 *
 * - Default (Vercel, local `next dev`): the normal server build. Response headers
 *   are applied by Next itself.
 * - `NEXT_OUTPUT=export` (Render static site, GitHub Pages): a fully static
 *   `out/` folder. Static hosting cannot run Next middleware or its header
 *   pipeline, so the `headers()` block has to be dropped in that mode or the
 *   build fails with "headers is not supported with output: export".
 *
 * Everything else is identical, so the conditional stays tiny on purpose.
 */
const isStaticExport = process.env.NEXT_OUTPUT === "export";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,

  // Static export needs relative, folder-per-route URLs and no image optimizer.
  ...(isStaticExport
    ? {
        output: "export" as const,
        trailingSlash: true,
        images: { unoptimized: true },
      }
    : {}),

  // The upload API expects raw multipart bodies; Next's default 1MB body limit
  // only applies to its own route handlers, but setting it explicitly documents
  // the contract for anyone adding a proxy route later.
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },

  // Static hosts serve headers from their own config, so this only applies to
  // the server build.
  ...(isStaticExport
    ? {}
    : {
        async headers() {
          return [
            {
              // Applied to every route. These are cheap and stop an entire class
              // of findings in a security review.
              source: "/:path*",
              headers: [
                { key: "X-Content-Type-Options", value: "nosniff" },
                { key: "X-Frame-Options", value: "DENY" },
                { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
                {
                  key: "Permissions-Policy",
                  // The camera is used for document capture, so it must be allowed
                  // for this origin only; everything else is denied.
                  value: "camera=(self), microphone=(), geolocation=()",
                },
              ],
            },
          ];
        },
      }),
};

export default nextConfig;
