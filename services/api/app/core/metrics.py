"""Lightweight in-process metrics.

Why hand-rolled instead of prometheus_client: the exporter adds ~2MB of RSS and
a background thread for a service whose entire metric surface is five counters
and one latency histogram. The text format below is directly scrapeable by
Prometheus, so the observability story is intact without the dependency.
"""

from __future__ import annotations

import threading

# Latency buckets in milliseconds, chosen around the 3s p95 target for one page.
_BUCKETS_MS: tuple[float, ...] = (100, 250, 500, 1000, 2000, 3000, 5000, 10000, 25000)


class Metrics:
    """Thread-safe counters and a latency histogram.

    Attributes:
        _lock: Guards all mutations; sync routes run on a threadpool.
        _counters: Name -> integer value.
        _latency_buckets: Cumulative counts per bucket bound.
        _latency_sum_ms: Sum of observed latencies.
        _latency_count: Number of observations.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "ocr_requests_total": 0,
            "ocr_failures_total": 0,
            "ocr_rate_limited_total": 0,
            "ocr_pages_total": 0,
            "feedback_samples_total": 0,
        }
        self._latency_buckets: list[int] = [0] * len(_BUCKETS_MS)
        self._latency_sum_ms: float = 0.0
        self._latency_count: int = 0
        self._started_at: float = 0.0

    def inc(self, name: str, amount: int = 1) -> None:
        """Increment a counter.

        Args:
            name: Counter name; unknown names are created on demand.
            amount: Delta, defaults to 1.
        """
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe_latency(self, duration_ms: float) -> None:
        """Record one end-to-end request latency.

        Args:
            duration_ms: Measured wall-clock duration.
        """
        with self._lock:
            self._latency_sum_ms += duration_ms
            self._latency_count += 1
            for index, bound in enumerate(_BUCKETS_MS):
                if duration_ms <= bound:
                    self._latency_buckets[index] += 1

    def render(self) -> str:
        """Render the Prometheus text exposition format.

        Returns:
            A UTF-8 text payload with one line per metric.
        """
        with self._lock:
            lines = ["# HELP varnam_ocr_total OCR requests handled by this instance",
                     "# TYPE varnam_ocr_total counter"]
            lines.extend(f"{name} {value}" for name, value in sorted(self._counters.items()))

            lines.append("# TYPE ocr_latency_ms histogram")
            cumulative = 0
            for index, bound in enumerate(_BUCKETS_MS):
                # Prometheus histograms are cumulative by definition.
                cumulative += self._latency_buckets[index]
                lines.append(f'ocr_latency_ms_bucket{{le="{bound}"}} {cumulative}')
            lines.append(f'ocr_latency_ms_bucket{{le="+Inf"}} {self._latency_count}')
            lines.append(f"ocr_latency_ms_sum {round(self._latency_sum_ms, 2)}")
            lines.append(f"ocr_latency_ms_count {self._latency_count}")
        return "\n".join(lines) + "\n"
