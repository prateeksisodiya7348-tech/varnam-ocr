import { describe, expect, it } from "vitest";
import { confidenceLabel, escapeXml, toHocr, toJson, toPlainText } from "@/lib/export";
import type { OcrResult } from "@/lib/types";

/**
 * Build a minimal valid result for the formatters.
 *
 * @returns An OcrResult with one line and one word.
 */
function fixture(): OcrResult {
  return {
    job_id: "job-1",
    status: "succeeded",
    engine: "fake",
    text: "नमस्ते दुनिया",
    lines: [
      {
        text: "नमस्ते दुनिया",
        bbox: { x0: 10, y0: 20, x1: 300, y1: 60 },
        confidence: 0.93,
        words: [{ text: "नमस्ते", bbox: { x0: 10, y0: 20, x1: 140, y1: 60 }, confidence: 0.93 }],
      },
    ],
    page: { width: 400, height: 100, scale: 1, line_count: 1 },
    mean_confidence: 0.93,
    devanagari_ratio: 1,
    duration_ms: 12.5,
    created_at: "2026-01-01T00:00:00Z",
  };
}

describe("plain text export", () => {
  it("joins lines with newlines and ends with one", () => {
    expect(toPlainText(fixture().lines)).toBe("नमस्ते दुनिया\n");
  });

  it("preserves combining marks exactly", () => {
    const lines = [{ ...fixture().lines[0]!, text: "क्षेत्रीय" }];
    expect(toPlainText(lines)).toBe("क्षेत्रीय\n");
  });
});

describe("json export", () => {
  it("round-trips through JSON.parse", () => {
    const parsed = JSON.parse(toJson(fixture())) as Record<string, unknown>;
    expect(parsed.engine).toBe("fake");
    expect(Array.isArray(parsed.lines)).toBe(true);
  });
});

describe("hocr export", () => {
  it("contains the required structural classes", () => {
    const document = toHocr(fixture());
    expect(document).toContain('class="ocr_page"');
    expect(document).toContain('class="ocr_line"');
    expect(document).toContain('class="ocrx_word"');
    expect(document).toContain("x_wconf 93");
  });

  it("escapes markup so output cannot be injected", () => {
    const result = fixture();
    result.lines[0]!.text = "<img src=x onerror=alert(1)>";
    // Word spans take precedence over the line text when both exist, so clear
    // them first - otherwise this test never exercises the line-level escaping
    // path and would pass even if that path stopped escaping.
    result.lines[0]!.words = [];
    const document = toHocr(result);
    expect(document).not.toContain("<img src=x");
    expect(document).toContain("&lt;img");
  });
});

describe("helpers", () => {
  it("escapes XML entities", () => {
    expect(escapeXml(`<&">'`)).toBe("&lt;&amp;&quot;&gt;&apos;");
  });

  it("maps confidence to a qualitative label", () => {
    expect(confidenceLabel(0.95)).toBe("high");
    expect(confidenceLabel(0.8)).toBe("medium");
    expect(confidenceLabel(0.5)).toBe("low");
  });
});
