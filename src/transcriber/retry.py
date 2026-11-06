"""Bounded retry for transient speech-to-text failures.

Permanent failures and programming errors propagate immediately. Sleep is
injected so tests can assert the backoff schedule without waiting.
"""

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

from transcriber.exceptions import TransientEngineError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _default_jitter(base_delay: float) -> float:
    """Return a uniform jitter in ``[0, base_delay]`` so retries do not align."""
    return random.uniform(0, base_delay)


def retry(
    fn: Callable[[], T],
    *,
    attempts: int,
    base_delay: float,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float], float] | None = None,
) -> T:
    """Call ``fn`` until it succeeds or transient retries are exhausted.

    The delay after attempt ``n`` (starting at 0) is ``base_delay * 2^n`` plus
    jitter. Only ``TransientEngineError`` is retried. The last transient error
    is re-raised so callers see the real failure, not a generic wrapper.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    jitter_fn = jitter or _default_jitter
    last_error: TransientEngineError | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except TransientEngineError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            delay = base_delay * (2**attempt) + jitter_fn(base_delay)
            logger.warning(
                "Transient engine error on attempt %s/%s: %s; retrying in %.2fs",
                attempt + 1,
                attempts,
                exc,
                delay,
            )
            sleep(delay)
    if last_error is None:
        raise RuntimeError("retry loop exited without a result")
    raise last_error
