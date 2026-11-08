/**
 * Export formatters.
 *
 * hOCR is generated in the browser from the same result object the API returned,
 * which keeps the feature usable offline and avoids a second round trip. The
 * server also exposes `?export=hocr`, and both produce the same structure.
 */
import type { OcrLine, OcrResult } from "./types";

/** Escape a string for use in XML text content. */
export function escapeXml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

/**
 * Plain text export: one line per recognized line.
 *
 * @param lines Recognized lines in reading order.
 * @returns UTF-8 text with a trailing newline.
 */
export function toPlainText(lines: OcrLine[]): string {
  return `${lines.map((line) => line.text).join("\n")}\n`;
}

/**
 * JSON export with metadata, for downstream tooling.
 *
 * @param result Full OCR result.
 * @returns Pretty-printed JSON string.
 */
export function toJson(result: OcrResult): string {
  return `${JSON.stringify(
    {
      engine: result.engine,
      job_id: result.job_id,
      created_at: result.created_at,
      mean_confidence: result.mean_confidence,
      devanagari_ratio: result.devanagari_ratio,
      page: result.page,
      lines: result.lines,
    },
    null,
    2,
  )}\n`;
}

/**
 * hOCR 1.2 export.
 *
 * @param result Full OCR result.
 * @returns An hOCR XHTML document.
 *
 * Why: hOCR is the format OCR pipelines (Tesseract, OCRmyPDF) actually consume,
 * so this output can be chained into a PDF/A workflow without custom glue.
 */
export function toHocr(result: OcrResult): string {
  const spans = result.lines
    .map((line, index) => {
      const words = line.words
        .map(
          (word, wordIndex) =>
            `<span class="ocrx_word" id="word_1_${index + 1}_${wordIndex + 1}" ` +
            `title="bbox ${word.bbox.x0} ${word.bbox.y0} ${word.bbox.x1} ${word.bbox.y1}; ` +
            `x_wconf ${Math.round(word.confidence * 100)}">${escapeXml(word.text)}</span> `,
        )
        .join("");
      const content = words || escapeXml(line.text);
      return (
        `<span class="ocr_line" id="line_1_${index + 1}" ` +
        `title="bbox ${line.bbox.x0} ${line.bbox.y0} ${line.bbox.x1} ${line.bbox.y1}; ` +
        `x_wconf ${Math.round(line.confidence * 100)}">${content}</span><br />`
      );
    })
    .join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="hi">
<head>
  <title>Varnam OCR export</title>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <meta name="ocr-system" content="varnam-ocr" />
</head>
<body>
  <div class="ocr_page" id="page_1" title="bbox 0 0 ${result.page.width} ${result.page.height}; ppageno 0">
    <p class="ocr_par" id="par_1_1" lang="hi">
${spans}
    </p>
  </div>
</body>
</html>
`;
}

/**
 * Trigger a browser download for a generated string.
 *
 * @param filename Suggested filename.
 * @param content File contents.
 * @param mimeType MIME type; defaults to UTF-8 text.
 */
export function downloadFile(filename: string, content: string, mimeType = "text/plain;charset=utf-8"): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  // Revoking immediately can cancel the download in some browsers; one tick of
  // delay is the pragmatic fix.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Map a confidence value to a human label.
 *
 * @param confidence Number in [0, 1].
 * @returns Short qualitative label used in badges.
 */
export function confidenceLabel(confidence: number): string {
  if (confidence >= 0.9) return "high";
  if (confidence >= 0.75) return "medium";
  return "low";
}
