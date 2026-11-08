"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

/**
 * Site header with a theme toggle.
 *
 * The toggle is a client component while the rest of the page can stay server
 * rendered, which keeps the landing page fast.
 */
export function SiteHeader() {
  const [dark, setDark] = useState(false);

  // Read the class the inline boot script already applied, so the button glyph
  // matches reality on the very first render.
  useEffect(() => {
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  const toggle = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("varnam-theme", next ? "dark" : "light");
    } catch {
      // Private browsing can block storage; the toggle still works for the session.
    }
  };

  return (
    <header className="sticky top-0 z-20 border-b border-line bg-paper/85 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2.5">
          <span
            aria-hidden
            className="grid h-9 w-9 place-items-center rounded-lg bg-saffron text-lg font-bold text-white"
          >
            व
          </span>
          <span className="flex flex-col leading-none">
            <span className="text-base font-semibold tracking-tight">Varnam OCR</span>
            <span className="text-[11px] uppercase tracking-[0.14em] text-muted">
              Devanagari recognition
            </span>
          </span>
        </Link>

        <nav className="flex items-center gap-1 text-sm">
          <a
            href="/docs"
            className="hidden rounded-md px-3 py-2 text-muted transition-colors hover:text-ink sm:block"
          >
            API
          </a>
          <Link
            href="/app"
            className="rounded-md bg-ink px-4 py-2 font-medium text-paper transition-opacity hover:opacity-90"
          >
            Open app
          </Link>
          <button
            type="button"
            onClick={toggle}
            aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
            aria-pressed={dark}
            className={cn(
              "ml-1 grid h-9 w-9 place-items-center rounded-md border border-line text-muted",
              "transition-colors hover:text-ink",
            )}
          >
            {dark ? "☀" : "☾"}
          </button>
        </nav>
      </div>
    </header>
  );
}
