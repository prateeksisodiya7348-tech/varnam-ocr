"""Rate limiter behaviour.

The limiter is stateful, so these tests build fresh instances rather than
touching the app-wide one - a shared bucket would make test order matter.
"""

from __future__ import annotations

import time

from app.core.ratelimit import TokenBucketLimiter


def test_allows_up_to_capacity_then_blocks() -> None:
    """Burst capacity is exact, and the next call is refused with a retry hint."""
    limiter = TokenBucketLimiter.per_minute(3)
    for _ in range(3):
        allowed, _ = limiter.allow("client-a")
        assert allowed

    allowed, retry_after = limiter.allow("client-a")
    assert not allowed
    assert retry_after > 0


def test_clients_are_isolated() -> None:
    """One noisy client must not consume another client's budget."""
    limiter = TokenBucketLimiter.per_minute(1)
    assert limiter.allow("client-a")[0]
    assert limiter.allow("client-b")[0]
    assert not limiter.allow("client-a")[0]


def test_refills_over_time() -> None:
    """Tokens come back as time passes; a full minute restores the bucket."""
    # 60 requests/minute => 1 token per second, so a 1.1s sleep refunds one.
    limiter = TokenBucketLimiter.per_minute(60)
    assert limiter.allow("client")[0]
    assert limiter.allow("client")[0]
    time.sleep(1.1)
    assert limiter.allow("client")[0]


def test_cost_weights_expensive_requests() -> None:
    """A multi-page job costs more than a single page, as designed."""
    limiter = TokenBucketLimiter.per_minute(10)
    assert limiter.allow("client", cost=4.0)[0]
    assert limiter.allow("client", cost=4.0)[0]
    # 8 of 10 tokens spent; a 4-token request cannot fit.
    allowed, retry_after = limiter.allow("client", cost=4.0)
    assert not allowed
    assert retry_after > 0
