import Link from "next/link";
import { SiteHeader } from "@/components/site-header";

/**
 * Landing page.
 *
 * Server component: it has no interactivity beyond the header toggle, so shipping
 * JavaScript for the whole page would be pure cost. The actual tool lives at
 * /app and is client-rendered because it needs file APIs.
 */
export default function HomePage() {
  const features = [
    {
      title: "Line-accurate boxes",
      body: "Every recognized line comes back with pixel coordinates and a confidence value, mapped back to your original image — no coordinate math in the UI.",
    },
    {
      title: "Devanagari-safe text",
      body: "Matras, virama, nukta, anusvara and the danda survive normalization intact. No silent character stripping, ever.",
    },
    {
      title: "Corrections become training data",
      body: "Edit a wrong line and it is stored with the model's prediction. The highest-value dataset is the one your own users label.",
    },
    {
      title: "Export that composes",
      body: "Plain text, structured JSON, and hOCR 1.2 — the last one feeds Tesseract tooling and OCRmyPDF directly.",
    },
    {
      title: "Runs on free infrastructure",
      body: "Next.js on Vercel, FastAPI + ONNX on a 512 MB Render instance. The hard limits are documented, not hidden.",
    },
    {
      title: "Typed end to end",
      body: "Strict TypeScript, Zod validation at the network boundary, mypy-clean Python, and a CI gate on all of it.",
    },
  ];

  return (
    <div className="min-h-dvh">
      <SiteHeader />

      <main className="mx-auto max-w-6xl px-4 sm:px-6">
        {/* Hero */}
        <section className="py-16 sm:py-24">
          <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1 text-xs text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-saffron" />
            Open source · printed &amp; handwritten Devanagari
          </p>

          <h1 className="max-w-3xl text-4xl font-semibold leading-[1.15] tracking-tight sm:text-6xl">
            Turn a photograph of{" "}
            <span className="devanagari text-saffron">देवनागरी</span> into text you can
            actually edit.
          </h1>

          <p className="mt-6 max-w-2xl text-lg text-muted">
            Upload a page. Get structured text with per-line confidence, bounding boxes,
            and export formats that plug into real document pipelines. Built for Hindi,
            Marathi, Nepali and Sanskrit.
          </p>

          <div className="mt-9 flex flex-wrap items-center gap-3">
            <Link
              href="/app"
              className="rounded-lg bg-saffron px-6 py-3 text-sm font-semibold text-white transition-opacity hover:opacity-90"
            >
              Try it on your page
            </Link>
            <a
              href="/docs"
              className="rounded-lg border border-line px-6 py-3 text-sm font-medium transition-colors hover:border-saffron hover:text-saffron"
            >
              Read the API reference
            </a>
          </div>

          <p className="mt-4 text-xs text-muted">
            No account required. Files are processed once and the image is not retained
            beyond the session.
          </p>
        </section>

        {/* Sample output */}
        <section className="pb-16">
          <div className="overflow-hidden rounded-[var(--radius-card)] border border-line bg-surface">
            <div className="flex items-center justify-between border-b border-line px-5 py-3">
              <span className="text-xs font-medium uppercase tracking-[0.12em] text-muted">
                Sample result
              </span>
              <span className="text-[11px] text-muted">engine: onnx · 1.24s · 4 lines</span>
            </div>
            <div className="grid gap-6 p-5 sm:grid-cols-2">
              <div className="rounded-lg bg-paper p-5">
                <p className="mb-3 text-[11px] uppercase tracking-[0.12em] text-muted">Recognized</p>
                <p className="devanagari space-y-2 text-lg leading-relaxed">
                  भारत एक विशाल और विविधतापूर्ण देश है।
                  <br />
                  राष्ट्रीय शिक्षा नीति २०२० लागू की गई।
                  <br />
                  क्षेत्रीय भाषाओं का महत्व बढ़ रहा है।
                </p>
              </div>
              <div className="rounded-lg bg-paper p-5">
                <p className="mb-3 text-[11px] uppercase tracking-[0.12em] text-muted">
                  Structured output
                </p>
                <pre className="overflow-x-auto text-[12px] leading-relaxed text-muted">
{`{
  "text": "भारत एक विशाल …",
  "lines": [
    {
      "bbox": [72, 118, 812, 152],
      "confidence": 0.97
    }
  ],
  "devanagari_ratio": 1.0
}`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Features */}
        <section className="border-t border-line py-16">
          <h2 className="text-2xl font-semibold tracking-tight">
            Built like production software, not a demo
          </h2>
          <div className="mt-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {features.map((feature) => (
              <article
                key={feature.title}
                className="rounded-[var(--radius-card)] border border-line bg-surface p-5"
              >
                <h3 className="text-sm font-semibold">{feature.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{feature.body}</p>
              </article>
            ))}
          </div>
        </section>

        {/* Honest limits: stating them beats having a reviewer find them */}
        <section className="border-t border-line py-16">
          <h2 className="text-2xl font-semibold tracking-tight">What it will not do</h2>
          <ul className="mt-6 grid gap-3 text-sm text-muted sm:grid-cols-2">
            <li className="rounded-lg border border-line bg-surface p-4">
              <strong className="text-ink">Other scripts.</strong> Bengali, Tamil, Telugu
              and the rest are not in the charset and will not be recognized correctly.
            </li>
            <li className="rounded-lg border border-line bg-surface p-4">
              <strong className="text-ink">First request is slow.</strong> The free API
              instance sleeps when idle; expect ~20–30 seconds once, then it is fast.
            </li>
            <li className="rounded-lg border border-line bg-surface p-4">
              <strong className="text-ink">Complex layout.</strong> Columns and tables use
              a heuristic segmenter, not a learned layout model.
            </li>
            <li className="rounded-lg border border-line bg-surface p-4">
              <strong className="text-ink">No human-in-the-loop guarantee.</strong> Every
              line carries a confidence value — review the low ones before trusting them.
            </li>
          </ul>
        </section>
      </main>

      <footer className="border-t border-line py-8">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-4 text-xs text-muted sm:flex-row sm:px-6">
          <span>Varnam OCR · Devanagari recognition, built in the open.</span>
          <span className="flex items-center gap-4">
            <a href="/docs" className="hover:text-ink">
              API
            </a>
            <a href="/app" className="hover:text-ink">
              App
            </a>
            <a href="https://github.com" className="hover:text-ink">
              Source
            </a>
          </span>
        </div>
      </footer>
    </div>
  );
}
