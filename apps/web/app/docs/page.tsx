import type { Metadata } from "next";
import { SiteHeader } from "@/components/site-header";

export const metadata: Metadata = {
  title: "API reference",
  description: "Endpoints, request shapes and error codes for the Varnam OCR API.",
};

interface Endpoint {
  method: string;
  path: string;
  summary: string;
  detail: string;
}

const ENDPOINTS: Endpoint[] = [
  {
    method: "POST",
    path: "/v1/ocr",
    summary: "Recognize one image",
    detail:
      "multipart/form-data with a single `file` field (PNG, JPEG or WEBP, ≤10 MB). Returns text, per-line boxes and confidence. Add `?export=txt` or `?export=hocr` for a raw body.",
  },
  {
    method: "POST",
    path: "/v1/ocr/pdf",
    summary: "Queue a PDF",
    detail:
      "Returns 202 with a job id. Multipage documents exceed any sane request timeout, so pages are rendered and recognized in the background.",
  },
  {
    method: "GET",
    path: "/v1/jobs/{id}",
    summary: "Poll a job",
    detail: "Returns queued | processing | succeeded | failed, plus the result once complete.",
  },
  {
    method: "POST",
    path: "/v1/feedback",
    summary: "Submit a correction",
    detail:
      "JSON body `{ job_id, line_index, corrected_text }`. Stored alongside the model's prediction as a training sample.",
  },
  {
    method: "GET",
    path: "/health/ready",
    summary: "Readiness",
    detail: "503 until the model is loaded; use this, not /health/live, for traffic routing.",
  },
];

const ERRORS: Array<[string, string, string]> = [
  ["invalid_upload", "422", "Undecodable, empty or degenerate image."],
  ["payload_too_large", "413", "Exceeds the byte or page ceiling."],
  ["unsupported_media", "415", "Not a PNG/JPEG/WEBP/PDF by magic bytes."],
  ["rate_limited", "429", "Token bucket exhausted. Respect Retry-After."],
  ["engine_busy", "503", "Inference slot occupied. Retry shortly."],
  ["engine_unavailable", "503", "Model not loaded or runtime failure."],
  ["not_found", "404", "Unknown job id."],
];

/** Static API reference page. Kept in sync with the OpenAPI document by hand. */
export default function DocsPage() {
  return (
    <div className="min-h-dvh">
      <SiteHeader />
      <main className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
        <h1 className="text-3xl font-semibold tracking-tight">API reference</h1>
        <p className="mt-2 text-sm text-muted">
          Base URL is the deployed Render service. The interactive schema lives at{" "}
          <code className="rounded bg-surface px-1.5 py-0.5 text-xs">/docs</code> on the API
          host.
        </p>

        <h2 className="mt-10 text-lg font-semibold">Endpoints</h2>
        <div className="mt-4 space-y-3">
          {ENDPOINTS.map((endpoint) => (
            <article key={endpoint.path} className="rounded-[var(--radius-card)] border border-line bg-surface p-4">
              <div className="flex items-center gap-2">
                <span className="rounded bg-saffron-soft px-2 py-0.5 text-[11px] font-semibold text-saffron">
                  {endpoint.method}
                </span>
                <code className="text-sm">{endpoint.path}</code>
              </div>
              <p className="mt-2 text-sm font-medium">{endpoint.summary}</p>
              <p className="mt-1 text-sm text-muted">{endpoint.detail}</p>
            </article>
          ))}
        </div>

        <h2 className="mt-10 text-lg font-semibold">Error codes</h2>
        <div className="mt-4 overflow-hidden rounded-[var(--radius-card)] border border-line">
          <table className="w-full text-left text-sm">
            <thead className="bg-surface text-[11px] uppercase tracking-wide text-muted">
              <tr>
                <th className="px-4 py-2 font-medium">code</th>
                <th className="px-4 py-2 font-medium">status</th>
                <th className="px-4 py-2 font-medium">meaning</th>
              </tr>
            </thead>
            <tbody>
              {ERRORS.map(([code, status, meaning]) => (
                <tr key={code} className="border-t border-line">
                  <td className="px-4 py-2 font-mono text-xs">{code}</td>
                  <td className="px-4 py-2 tabular-nums text-muted">{status}</td>
                  <td className="px-4 py-2 text-muted">{meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h2 className="mt-10 text-lg font-semibold">Try it</h2>
        <pre className="mt-4 overflow-x-auto rounded-[var(--radius-card)] border border-line bg-surface p-4 text-xs leading-relaxed">
{`curl -s http://127.0.0.1:8000/v1/ocr \\
  -F "file=@page.png" \\
  | jq '.text, .mean_confidence'

# raw text instead of JSON
curl -s "http://127.0.0.1:8000/v1/ocr?export=txt" -F "file=@page.png"`}
        </pre>

        <p className="mt-10 text-xs text-muted">
          Every response carries <code>x-request-id</code> and <code>x-response-time-ms</code>{" "}
          headers; quote the request id when reporting a failure.
        </p>
      </main>
    </div>
  );
}
