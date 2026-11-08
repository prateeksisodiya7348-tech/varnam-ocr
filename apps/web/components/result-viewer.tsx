"use client";

import { cn } from "@/lib/utils";
import type { OcrLine, OcrPage } from "@/lib/types";

/**
 * Props for {@link ResultViewer}.
 */
interface ResultViewerProps {
  /** Object URL of the original upload. */
  imageUrl: string;
  /** Page geometry reported by the API, in original-upload pixels. */
  page: OcrPage;
  /** Recognized lines, already in reading order. */
  lines: OcrLine[];
  /** Index currently highlighted from the editor side. */
  activeLine: number | null;
  /** Called when the pointer enters a box. */
  onHoverLine: (index: number | null) => void;
}

/**
 * Colour for a confidence value.
 *
 * @param confidence Value in [0, 1].
 * @returns Tailwind classes for border and background.
 */
function confidenceClasses(confidence: number): string {
  if (confidence >= 0.9) return "border-emerald-500/70 bg-emerald-500/10";
  if (confidence >= 0.75) return "border-amber-500/70 bg-amber-500/10";
  return "border-red-500/70 bg-red-500/10";
}

/**
 * Original image with a bounding-box overlay.
 *
 * Why percentages instead of pixels: the image is rendered responsively, so
 * absolute pixel boxes would drift out of alignment at every viewport width.
 * Expressing each box as a percentage of the reported page size keeps it exact at
 * any size with no resize listener.
 */
export function ResultViewer({ imageUrl, page, lines, activeLine, onHoverLine }: ResultViewerProps) {
  return (
    <div className="relative overflow-hidden rounded-[var(--radius-card)] border border-line bg-surface">
      <div className="relative" style={{ aspectRatio: `${page.width} / ${page.height}` }}>
        {/* eslint-disable-next-line @next/next/no-img-element -- object URL of a local file; next/image cannot optimize it */}
        <img
          src={imageUrl}
          alt="Uploaded document"
          className="absolute inset-0 h-full w-full object-contain"
        />

        {lines.map((line, index) => {
          const left = (line.bbox.x0 / page.width) * 100;
          const top = (line.bbox.y0 / page.height) * 100;
          const width = ((line.bbox.x1 - line.bbox.x0) / page.width) * 100;
          const height = ((line.bbox.y1 - line.bbox.y0) / page.height) * 100;
          const isActive = activeLine === index;

          return (
            <button
              key={index}
              type="button"
              aria-label={`Line ${index + 1}: ${line.text}`}
              onMouseEnter={() => onHoverLine(index)}
              onMouseLeave={() => onHoverLine(null)}
              onFocus={() => onHoverLine(index)}
              onBlur={() => onHoverLine(null)}
              className={cn(
                "absolute rounded-[3px] border transition-all",
                confidenceClasses(line.confidence),
                isActive ? "opacity-100 ring-2 ring-saffron" : "opacity-45 hover:opacity-90",
              )}
              style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
            />
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-4 border-t border-line px-4 py-2.5 text-[11px] text-muted">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm border border-emerald-500 bg-emerald-500/20" />
          high confidence
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm border border-amber-500 bg-amber-500/20" />
          review
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm border border-red-500 bg-red-500/20" />
          low
        </span>
        <span className="ml-auto tabular-nums">
          {page.width}×{page.height}px · {lines.length} lines
        </span>
      </div>
    </div>
  );
}
