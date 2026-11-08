import type { Metadata, Viewport } from "next";
import "./globals.css";

/**
 * Root layout.
 *
 * Fonts are resolved from the user's system (with Noto Sans Devanagari first)
 * rather than via next/font. Reason: next/font downloads at build time, which
 * makes an offline build fail, and the Devanagari subset is large enough that a
 * self-hosted copy hurts the first paint on mobile more than a system fallback.
 */
export const metadata: Metadata = {
  title: {
    default: "Varnam OCR — Devanagari text recognition",
    template: "%s · Varnam OCR",
  },
  description:
    "Open-source OCR for printed and handwritten Devanagari. Upload a page, get editable text with character-level confidence and bounding boxes.",
  keywords: ["Devanagari OCR", "Hindi OCR", "handwritten OCR", "Marathi", "Nepali", "Sanskrit"],
  authors: [{ name: "Varnam OCR" }],
  openGraph: {
    title: "Varnam OCR — Devanagari text recognition",
    description: "Printed and handwritten Devanagari to editable text, with confidence heatmaps.",
    type: "website",
  },
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fbf8f3" },
    { media: "(prefers-color-scheme: dark)", color: "#12120f" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/*
          Applies the stored theme before first paint. Without this inline script
          the page renders light, then flips - a visible flash on every navigation.
        */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(() => {
              try {
                const stored = localStorage.getItem("varnam-theme");
                const dark = stored ? stored === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
                if (dark) document.documentElement.classList.add("dark");
              } catch (_) {}
            })();`,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
