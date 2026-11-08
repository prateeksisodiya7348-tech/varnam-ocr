"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ExportMenu } from "@/components/export-menu";
import { ResultViewer } from "@/components/result-viewer";
import { SiteHeader } from "@/components/site-header";
import { TranscriptEditor } from "@/components/transcript-editor";
import { UploadDropzone } from "@/components/upload-dropzone";
import { ApiRequestError, runOcr, sendFeedback } from "@/lib/api";
import type { OcrResult, RequestStatus } from "@/lib/types";

/**
 * The OCR workspace.
 *
 * State machine: idle -> processing -> done | error. Keeping the status explicit
 * (rather than deriving it from `result`/`error` booleans) is what makes the
 * cold-start, retry and cancel cases expressible at all.
 */
export default function WorkspacePage() {
  const [status, setStatus] = useState<RequestStatus>("idle");
  const [progress, setProgress] = useState<string>("");
  const [result, setResult] = useState<OcrResult | null>(null);
  const [texts, setTexts] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [activeLine, setActiveLine] = useState<number | null>(null);

  // Kept in a ref so a re-render never orphans an in-flight request that then
  // tries to setState on an unmounted component.
  const abortRef = useRef<AbortController | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  /** Revoke the previous object URL so long sessions do not leak blobs. */
  const releaseObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  useEffect(() => () => {
    abortRef.current?.abort();
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
  }, []);

  const process = useCallback(
    async (file: File) => {
      releaseObjectUrl();
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;

      setImageUrl(url);
      setResult(null);
      setTexts([]);
      setError(null);
      setStatus("processing");
      setProgress("Preparing upload…");

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await runOcr(file, {
          signal: controller.signal,
          onProgress: setProgress,
        });
        setResult(response);
        setTexts(response.lines.map((line) => line.text));
        setStatus("done");
        setProgress("");
      } catch (caught) {
        if (controller.signal.aborted) {
          setStatus("idle");
          return;
        }
        setStatus("error");
        setError(
          caught instanceof ApiRequestError
            ? [caught.message, caught.requestId ? `Request id: ${caught.requestId}` : null]
                .filter(Boolean)
                .join(" ")
            : "Something went wrong while contacting the OCR service.",
        );
      } finally {
        abortRef.current = null;
      }
    },
    [releaseObjectUrl],
  );

  const reset = () => {
    abortRef.current?.abort();
    releaseObjectUrl();
    setImageUrl(null);
    setResult(null);
    setTexts([]);
    setError(null);
    setStatus("idle");
    setProgress("");
  };

  /** Persist a correction when a line is blurred and actually changed. */
  const commitCorrection = async (index: number) => {
    if (!result) return;
    const original = result.lines[index]?.text ?? "";
    const edited = texts[index] ?? "";
    if (edited.trim() === original.trim() || !edited.trim()) return;
    await sendFeedback(result.job_id, index, edited);
  };

  const lowConfidence = result
    ? result.lines.filter((line) => line.confidence < 0.75).length
    : 0;

  return (
    <div className="min-h-dvh">
      <SiteHeader />

      <main className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">OCR workspace</h1>
            <p className="mt-1 text-sm text-muted">
              One page at a time. Edit any line you disagree with — corrections are saved
              as future training samples.
            </p>
          </div>
          {result ? (
            <button
              type="button"
              onClick={reset}
              className="rounded-md border border-line px-3.5 py-2 text-sm transition-colors hover:border-saffron hover:text-saffron"
            >
              New page
            </button>
          ) : null}
        </div>

        {/* Screen-reader status announcements: OCR takes seconds, and a silent UI
            is unusable with a reader. */}
        <p aria-live="polite" className="sr-only">
          {status === "processing" ? progress : status === "done" ? "Recognition complete" : ""}
        </p>

        {!result ? (
          <div className="mt-8">
            <UploadDropzone onSelect={process} disabled={status === "processing"} />

            {status === "processing" ? (
              <div className="mt-5 rounded-[var(--radius-card)] border border-line bg-surface p-5">
                <div className="flex items-center gap-3">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-saffron border-t-transparent" />
                  <p className="text-sm font-medium">{progress || "Working…"}</p>
                </div>
                <p className="mt-2 text-xs text-muted">
                  The free-tier service sleeps when idle. The first request can take up to
                  30 seconds while the model loads; it retries automatically.
                </p>
                <button
                  type="button"
                  onClick={reset}
                  className="mt-3 text-xs text-muted underline hover:text-ink"
                >
                  Cancel
                </button>
              </div>
            ) : null}

            {status === "error" && error ? (
              <div
                role="alert"
                className="mt-5 rounded-[var(--radius-card)] border border-red-500/40 bg-red-500/5 p-5"
              >
                <p className="text-sm font-medium text-red-700 dark:text-red-400">
                  Could not process that page
                </p>
                <p className="mt-1 text-sm text-muted">{error}</p>
                <button
                  type="button"
                  onClick={reset}
                  className="mt-3 rounded-md border border-line px-3 py-1.5 text-xs hover:border-saffron"
                >
                  Try another file
                </button>
              </div>
            ) : null}
          </div>
        ) : (
          <div className="mt-8 grid gap-6 lg:grid-cols-2">
            <section aria-label="Source image with detected lines">
              {imageUrl ? (
                <ResultViewer
                  imageUrl={imageUrl}
                  page={result.page}
                  lines={result.lines}
                  activeLine={activeLine}
                  onHoverLine={setActiveLine}
                />
              ) : null}

              <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] text-muted">
                <span className="rounded bg-surface px-2 py-1">engine: {result.engine}</span>
                <span className="rounded bg-surface px-2 py-1 tabular-nums">
                  {result.duration_ms.toFixed(0)} ms inference
                </span>
                <span className="rounded bg-surface px-2 py-1 tabular-nums">
                  {result.page.line_count} lines
                </span>
                {lowConfidence > 0 ? (
                  <span className="rounded bg-amber-500/15 px-2 py-1 text-amber-700 dark:text-amber-400">
                    {lowConfidence} line{lowConfidence === 1 ? "" : "s"} need review
                  </span>
                ) : null}
              </div>

              {result.devanagari_ratio < 0.3 ? (
                <p
                  role="alert"
                  className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs text-amber-800 dark:text-amber-300"
                >
                  Only {(result.devanagari_ratio * 100).toFixed(0)}% of the output is
                  Devanagari. The page may be rotated, upside down, or not in Devanagari at
                  all.
                </p>
              ) : null}
            </section>

            <section aria-label="Editable transcript" className="flex flex-col">
              <div className="mb-3">
                <ExportMenu result={result} lines={result.lines} texts={texts} />
              </div>

              <div className="max-h-[62vh] overflow-y-auto rounded-[var(--radius-card)] border border-line bg-paper p-2">
                <TranscriptEditor
                  lines={result.lines}
                  texts={texts}
                  activeLine={activeLine}
                  onHoverLine={setActiveLine}
                  onChange={(index, value) =>
                    setTexts((previous) => {
                      const next = [...previous];
                      next[index] = value;
                      return next;
                    })
                  }
                  onCommit={commitCorrection}
                />
              </div>

              <p className="mt-3 text-[11px] text-muted">
                Edits are stored with the model&apos;s original prediction to build a
                domain-specific fine-tuning set. Nothing is uploaded beyond the page you
                chose.
              </p>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
