"use client";

import { useState } from "react";
import { confidenceLabel, downloadFile, toHocr, toJson, toPlainText } from "@/lib/export";
import { cn } from "@/lib/utils";
import type { OcrLine, OcrResult } from "@/lib/types";

/**
 * Props for {@link ExportMenu}.
 */
interface ExportMenuProps {
  /** Server result, used for metadata and page geometry. */
  result: OcrResult;
  /** Model lines, needed for confidence values in hOCR. */
  lines: OcrLine[];
  /** Current text after user edits, which is what should be exported. */
  texts: string[];
}

/**
 * Export controls for the current result.
 *
 * Edits are applied before export: shipping the model's raw output after a user
 * corrected it would be a data-loss bug, and it is the kind of thing a user only
 * notices after pasting into a document.
 */
export function ExportMenu({ result, lines, texts }: ExportMenuProps) {
  const [copied, setCopied] = useState(false);

  /** Build the line array with user edits applied. */
  const editedLines: OcrLine[] = lines.map((line, index) => ({
    ...line,
    text: texts[index] ?? line.text,
  }));

  const editedResult: OcrResult = { ...result, lines: editedLines, text: texts.join("\n") };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(editedResult.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard access can be denied; fall back to a download so the user is
      // never stuck without a way to get the text out.
      downloadFile("varnam.txt", toPlainText(editedLines));
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={copy}
        className="rounded-md bg-ink px-3.5 py-2 text-sm font-medium text-paper transition-opacity hover:opacity-90"
      >
        {copied ? "Copied ✓" : "Copy text"}
      </button>

      <button
        type="button"
        onClick={() => downloadFile("varnam.txt", toPlainText(editedLines))}
        className={cn(
          "rounded-md border border-line px-3.5 py-2 text-sm",
          "transition-colors hover:border-saffron hover:text-saffron",
        )}
      >
        .txt
      </button>

      <button
        type="button"
        onClick={() => downloadFile("varnam.json", toJson(editedResult), "application/json")}
        className="rounded-md border border-line px-3.5 py-2 text-sm transition-colors hover:border-saffron hover:text-saffron"
      >
        .json
      </button>

      <button
        type="button"
        onClick={() => downloadFile("varnam.hocr", toHocr(editedResult), "application/xhtml+xml")}
        className="rounded-md border border-line px-3.5 py-2 text-sm transition-colors hover:border-saffron hover:text-saffron"
        title="hOCR 1.2 — opens in Tesseract tooling, OCRmyPDF and ABBYY readers"
      >
        .hocr
      </button>

      <div className="ml-auto flex items-center gap-3 text-[11px] text-muted">
        <span className="tabular-nums" title="Mean model confidence across lines">
          confidence {(result.mean_confidence * 100).toFixed(0)}% ({confidenceLabel(result.mean_confidence)})
        </span>
        <span className="tabular-nums" title="Fraction of characters that are Devanagari">
          देवनागरी {(result.devanagari_ratio * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}
