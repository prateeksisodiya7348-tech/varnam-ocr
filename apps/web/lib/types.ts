/**
 * Runtime-validated API types.
 *
 * The API is a separate deployment, so its responses are untrusted input as far
 * as this app is concerned. Parsing with Zod at the boundary means a schema
 * change produces one clear error at the edge instead of `undefined` leaking into
 * a component three levels deep.
 */
import { z } from "zod";

export const bboxSchema = z.object({
  x0: z.number(),
  y0: z.number(),
  x1: z.number(),
  y1: z.number(),
});

export const wordSchema = z.object({
  text: z.string(),
  bbox: bboxSchema,
  confidence: z.number(),
});

export const lineSchema = z.object({
  text: z.string(),
  bbox: bboxSchema,
  confidence: z.number(),
  words: z.array(wordSchema),
});

export const pageSchema = z.object({
  width: z.number(),
  height: z.number(),
  scale: z.number(),
  line_count: z.number(),
});

export const ocrResultSchema = z.object({
  job_id: z.string(),
  status: z.literal("succeeded"),
  engine: z.string(),
  text: z.string(),
  lines: z.array(lineSchema),
  page: pageSchema,
  mean_confidence: z.number(),
  devanagari_ratio: z.number(),
  duration_ms: z.number(),
  created_at: z.string(),
});

export const apiErrorSchema = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    request_id: z.string().nullable().optional(),
  }),
});

export type BBox = z.infer<typeof bboxSchema>;
export type Word = z.infer<typeof wordSchema>;
export type OcrLine = z.infer<typeof lineSchema>;
export type OcrPage = z.infer<typeof pageSchema>;
export type OcrResult = z.infer<typeof ocrResultSchema>;
export type ApiError = z.infer<typeof apiErrorSchema>;

/** Client-side view of a single processing request. */
export type RequestStatus = "idle" | "uploading" | "processing" | "done" | "error";
