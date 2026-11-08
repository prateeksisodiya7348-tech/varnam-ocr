"""In-process token-bucket rate limiting.

Why not slowapi/redis: the free Render instance is a single process, and an
in-process bucket with a lock is exact for that topology with zero extra
infrastructure or network hop. The tradeoff - it does not survive a restart and
is per-instance - is documented here and in docs/LIMITATIONS.md, with the
Redis-backed migration path spelled out. Choosing the boring thing and knowing
its boundary is the point.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    """One caller's bucket state.

    Attributes:
        tokens: Fractional tokens currently available.
        updated_at: Monotonic timestamp of the last refill.
    """

    tokens: float
    updated_at: float


@dataclass
class TokenBucketLimiter:
    """Fixed-capacity, continuously-refilling token bucket per key.

    Attributes:
        capacity: Maximum burst size (tokens).
        refill_per_second: Sustained rate.
        _buckets: Key -> bucket state.
        _lock: Guards ``_buckets``; the free tier is single-process but
            FastAPI runs sync routes in a threadpool, so mutation is concurrent.
    """

    capacity: int
    refill_per_second: float
    _buckets: dict[str, _Bucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_sweep: float = field(default=0.0)

    @classmethod
    def per_minute(cls, requests_per_minute: int) -> TokenBucketLimiter:
        """Build a limiter allowing ``requests_per_minute`` sustained calls.

        Args:
            requests_per_minute: Sustained quota; capacity equals the quota so a
                user may spend a full minute's allowance as one burst.

        Returns:
            A configured limiter.
        """
        return cls(
            capacity=max(1, requests_per_minute),
            refill_per_second=max(1, requests_per_minute) / 60.0,
        )

    def allow(self, key: str, *, cost: float = 1.0) -> tuple[bool, float]:
        """Attempt to spend ``cost`` tokens for ``key``.

        Args:
            key: Caller identity, typically the client IP.
            cost: Tokens to consume. Multi-page requests cost more than one so
                a single huge upload cannot consume a minute of CPU for free.

        Returns:
            ``(allowed, retry_after_seconds)``. ``retry_after`` is 0.0 when
            allowed, otherwise the wait needed to afford ``cost``.
        """
        now = time.monotonic()
        with self._lock:
            self._maybe_sweep(now)
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.capacity), updated_at=now)
                self._buckets[key] = bucket

            # Continuous refill, clamped so an idle bucket cannot bank more
            # than `capacity` tokens (otherwise a sleeper gets an unbounded burst).
            elapsed = now - bucket.updated_at
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
            bucket.updated_at = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, 0.0

            deficit = cost - bucket.tokens
            return False, round(deficit / self.refill_per_second, 2)

    def _maybe_sweep(self, now: float) -> None:
        """Drop buckets that have fully refilled and are therefore indistinguishable.

        Without this, one entry per IP accumulates forever - a memory leak that
        only shows up under a scanner hammering the endpoint.

        Args:
            now: Current monotonic time.
        """
        if now - self._last_sweep < 300:
            return
        self._last_sweep = now
        stale = [
            key
            for key, bucket in self._buckets.items()
            if now - bucket.updated_at > 600
        ]
        for key in stale:
            del self._buckets[key]
