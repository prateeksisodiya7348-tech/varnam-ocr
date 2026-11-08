import type { NextConfig } from "next";

/**
 * Next.js configuration.
 *
 * Design decision worth calling out in review: this app has NO API routes and no
 * server-side inference. The model physically cannot run in a Vercel function
 * (deployment size and 10s execution limits), so the entire ML path lives in the
 * Render service and the browser talks to it directly over HTTPS. Keeping the
 * frontend serverless-only is what makes the free tier viable.
 */
const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,

  // The upload API expects raw multipart bodies; Next's default 1MB body limit
  // only applies to its own route handlers, but setting it explicitly documents
  // the contract for anyone adding a proxy route later.
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },

  async headers() {
    return [
      {
        // Applied to every route. These are cheap and stop an entire class of
        // findings in a security review.
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
};

export default nextConfig;
