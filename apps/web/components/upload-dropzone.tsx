"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

/**
 * Props for {@link UploadDropzone}.
 */
interface UploadDropzoneProps {
  /** Called with the first accepted file. */
  onSelect: (file: File) => void;
  /** Disables interaction while a request is in flight. */
  disabled?: boolean;
  /** Hard client-side mirror of the API cap, in bytes. */
  maxBytes?: number;
}

const ACCEPTED = ["image/png", "image/jpeg", "image/webp"];

/**
 * Drag-and-drop, click, and paste upload target.
 *
 * Why three input paths: on a laptop people drag a file, on mobile they tap, and
 * when copying from a PDF viewer they paste. Supporting only one silently loses a
 * third of the audience.
 */
export function UploadDropzone({ onSelect, disabled = false, maxBytes = 10 * 1024 * 1024 }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  /**
   * Validate on the client too.
   *
   * @param file Candidate file.
   * @returns True when the file should be sent.
   *
   * The server re-validates by magic bytes; this check exists purely to give
   * instant feedback instead of a round trip and a 10MB upload penalty.
   */
  const accept = useCallback(
    (file: File): boolean => {
      if (!ACCEPTED.includes(file.type)) {
        setLocalError("Please choose a PNG, JPEG or WEBP image.");
        return false;
      }
      if (file.size > maxBytes) {
        setLocalError(`That file is ${(file.size / 1024 / 1024).toFixed(1)}MB; the limit is ${maxBytes / 1024 / 1024}MB.`);
        return false;
      }
      setLocalError(null);
      return true;
    },
    [maxBytes],
  );

  // Paste support: a screenshot copied from another app arrives as a clipboard
  // item, not a drag event.
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      if (disabled) return;
      const item = Array.from(event.clipboardData?.items ?? []).find((entry) =>
        entry.type.startsWith("image/"),
      );
      const file = item?.getAsFile();
      if (file && accept(file)) onSelect(file);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [accept, disabled, onSelect]);

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file && accept(file)) onSelect(file);
  };

  return (
    <div className="w-full">
      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        aria-label="Upload a document image"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(event) => {
          if ((event.key === "Enter" || event.key === " ") && !disabled) {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!disabled) handleFiles(event.dataTransfer.files);
        }}
        className={cn(
          "group relative flex cursor-pointer flex-col items-center justify-center gap-3 rounded-[var(--radius-card)]",
          "border-2 border-dashed px-6 py-14 text-center transition-colors",
          dragging ? "border-saffron bg-saffron-soft" : "border-line bg-surface hover:border-saffron",
          disabled && "pointer-events-none opacity-60",
        )}
      >
        <span
          aria-hidden
          className="grid h-14 w-14 place-items-center rounded-full bg-saffron-soft text-2xl"
        >
          ⬆
        </span>
        <div>
          <p className="text-base font-medium">Drop a page here, or click to browse</p>
          <p className="mt-1 text-sm text-muted">
            PNG, JPEG or WEBP · up to 10 MB · you can also paste from the clipboard
          </p>
        </div>
        <p className="mt-1 text-xs text-muted">
          Printed or handwritten Devanagari — Hindi, Marathi, Nepali, Sanskrit
        </p>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          className="sr-only"
          onChange={(event) => handleFiles(event.target.files)}
        />
      </div>

      {localError ? (
        <p role="alert" className="mt-3 text-sm text-red-600 dark:text-red-400">
          {localError}
        </p>
      ) : null}
    </div>
  );
}
