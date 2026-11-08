/**
 * Typed client for the Varnam OCR API.
 *
 * Three behaviours here are deliberate and worth defending in review:
 *
 * 1. **Retry on 503.** The API runs on a free instance that sleeps after ~15
 *    minutes. The first request after idle fails while the container boots, so a
 *    single retry with backoff turns a scary failure into a ~20s wait.
 * 2. **Abort support.** A 10MB upload on a phone should be cancellable, and the
 *    component unmount must not leave a dangling fetch that setState's later.
 * 3. **Zod parse at the boundary.** See lib/types.ts.
 */
import { apiErrorSchema, ocrResultSchema, type OcrResult } from "./types";

/**
 * Base URL of the OCR service.
 *
 * On Vercel this is `NEXT_PUBLIC_API_BASE_URL`. The fallback points at the local
 * dev server so `pnpm dev` works with no env file at all.
 */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

/** Error carrying the API's stable machine-readable code. */
export class ApiRequestError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly requestId?: string,
    readonly retryable = false,
  ) {
    super(message);
    this.name = "ApiRequestError";
  }
}

/** Progress callback so the UI can explain a cold start instead of hanging. */
export type ProgressHandler = (message: string) => void;

const RETRYABLE_CODES = new Set(["engine_busy", "engine_unavailable", "rate_limited"]);

/** Sleep helper used only for the retry path. */
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Send an image to the OCR endpoint.
 *
 * @param file Image chosen by the user.
 * @param options.signal Abort signal for cancellation.
 * @param options.onProgress Human-readable progress updates.
 * @returns The validated OCR result.
 * @throws ApiRequestError for every non-2xx response.
 */
export async function runOcr(
  file: File,
  options: { signal?: AbortSignal; onProgress?: ProgressHandler } = {},
): Promise<OcrResult> {
  const { signal, onProgress } = options;
  const attempts = 3;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    onProgress?.(
      attempt === 1
        ? `Uploading ${(file.size / 1024).toFixed(0)} KB…`
        : `Waking the OCR service (attempt ${attempt}/${attempts})…`,
    );

    const body = new FormData();
    body.append("file", file, file.name);

    let response: Response;
    try {
      response = await fetch(`${API_BASE_URL}/v1/ocr`, {
        method: "POST",
        body,
        signal,
        headers: { Accept: "application/json" },
      });
    } catch (error) {
      if (signal?.aborted) throw error;
      // A network-level failure is indistinguishable from a sleeping instance
      // from the browser's perspective, so it takes the same retry path.
      if (attempt === attempts) {
        throw new ApiRequestError(
          "Could not reach the OCR service. Check your connection and try again.",
          "network_error",
          undefined,
          true,
        );
      }
      await delay(1500 * attempt);
      continue;
    }

    if (response.ok) {
      onProgress?.("Reading the page…");
      const json: unknown = await response.json();
      return ocrResultSchema.parse(json);
    }

    const payload: unknown = await response.json().catch(() => null);
    const parsed = apiErrorSchema.safeParse(payload);
    const code = parsed.success ? parsed.data.error.code : "unknown_error";
    const message = parsed.success
      ? parsed.data.error.message
      : `Request failed with status ${response.status}.`;
    const requestId = parsed.success ? (parsed.data.error.request_id ?? undefined) : undefined;

    if (RETRYABLE_CODES.has(code) && attempt < attempts) {
      await delay(2000 * attempt);
      continue;
    }
    throw new ApiRequestError(message, code, requestId, RETRYABLE_CODES.has(code));
  }

  throw new ApiRequestError("The OCR service did not respond. Try again shortly.", "timeout");
}

/**
 * Submit a user correction. Fire-and-forget by design: a failure here must never
 * interrupt the user's editing flow.
 *
 * @param jobId Job the correction belongs to.
 * @param lineIndex Zero-based line index.
 * @param correctedText What the user typed.
 */
export async function sendFeedback(
  jobId: string,
  lineIndex: number,
  correctedText: string,
): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/v1/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, line_index: lineIndex, corrected_text: correctedText }),
    });
    return response.ok;
  } catch {
    // Silently ignored: losing a training sample is preferable to showing an
    // error while someone is mid-sentence.
    return false;
  }
}

/**
 * Check whether the backend is awake, used to show an accurate banner.
 *
 * @returns True when `/health/ready` answers 200.
 */
export async function isServiceReady(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health/ready`, { cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}
