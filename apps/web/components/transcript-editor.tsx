"use client";

import { cn } from "@/lib/utils";
import type { OcrLine } from "@/lib/types";

/**
 * Props for {@link TranscriptEditor}.
 */
interface TranscriptEditorProps {
  /** Original model output, kept for diffing against edits. */
  lines: OcrLine[];
  /** Current (possibly edited) text per line. */
  texts: string[];
  /** Currently highlighted line index. */
  activeLine: number | null;
  /** Called on every keystroke. */
  onChange: (index: number, value: string) => void;
  /** Called when a line loses focus, which is when a correction is submitted. */
  onCommit: (index: number) => void;
  /** Hover sync back to the image overlay. */
  onHoverLine: (index: number | null) => void;
}

/**
 * Editable per-line transcript.
 *
 * Why edit per line rather than one big textarea: a correction is only useful as
 * training data when it is attributable to a single line and its crop. A single
 * blob loses that alignment, which makes the active-learning loop worthless.
 */
export function TranscriptEditor({
  lines,
  texts,
  activeLine,
  onChange,
  onCommit,
  onHoverLine,
}: TranscriptEditorProps) {
  return (
    <div className="flex flex-col gap-1.5">
      {lines.map((line, index) => {
        const isEdited = texts[index] !== line.text;
        return (
          <div
            key={index}
            onMouseEnter={() => onHoverLine(index)}
            onMouseLeave={() => onHoverLine(null)}
            className={cn(
              "group flex items-start gap-2 rounded-md border px-2.5 py-2 transition-colors",
              activeLine === index
                ? "border-saffron bg-saffron-soft"
                : "border-transparent hover:border-line hover:bg-surface",
            )}
          >
            <span className="mt-1.5 w-7 shrink-0 text-right text-[11px] tabular-nums text-muted">
              {index + 1}
            </span>

            <textarea
              value={texts[index] ?? ""}
              onChange={(event) => onChange(index, event.target.value)}
              onBlur={() => onCommit(index)}
              rows={Math.max(1, Math.ceil((texts[index]?.length ?? 0) / 48))}
              aria-label={`Line ${index + 1} transcription`}
              spellCheck={false}
              className={cn(
                "devanagari min-h-9 w-full resize-none bg-transparent text-[15px] outline-none",
                "focus:bg-surface",
              )}
            />

            <span
              className={cn(
                "mt-1.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] tabular-nums",
                line.confidence >= 0.9
                  ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400"
                  : line.confidence >= 0.75
                    ? "bg-amber-500/15 text-amber-700 dark:text-amber-400"
                    : "bg-red-500/15 text-red-700 dark:text-red-400",
              )}
              title={`Model confidence ${(line.confidence * 100).toFixed(0)}%`}
            >
              {(line.confidence * 100).toFixed(0)}%
            </span>

            {isEdited ? (
              <span
                className="mt-1.5 shrink-0 text-[10px] font-medium text-saffron"
                title="Edited by you — saved as a training sample"
              >
                edited
              </span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
