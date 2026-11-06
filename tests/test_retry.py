"""Retry only transient engine failures, and record the backoff."""

import pytest

from transcriber.exceptions import PermanentEngineError, TransientEngineError
from transcriber.retry import retry


def test_retries_transient_errors_then_returns() -> None:
    """Two failures then success sleep for base * 2^attempt with the injected jitter."""
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientEngineError("busy")
        return "ok"

    delays: list[float] = []
    result = retry(
        flaky,
        attempts=3,
        base_delay=1.0,
        sleep=delays.append,
        jitter=lambda _base: 0.25,
    )
    assert result == "ok"
    assert delays == [1.25, 2.25]


def test_permanent_errors_are_not_retried() -> None:
    """A 4xx-style failure must not burn the remaining attempts."""
    calls = {"n": 0}

    def fail() -> None:
        calls["n"] += 1
        raise PermanentEngineError("denied")

    delays: list[float] = []
    with pytest.raises(PermanentEngineError):
        retry(fail, attempts=5, base_delay=1.0, sleep=delays.append, jitter=lambda _base: 0)
    assert calls["n"] == 1
    assert delays == []


def test_other_exceptions_propagate_immediately() -> None:
    """Bugs are not retried; only TransientEngineError is."""

    def fail() -> None:
        raise ValueError("bug")

    delays: list[float] = []
    with pytest.raises(ValueError, match="bug"):
        retry(fail, attempts=4, base_delay=1.0, sleep=delays.append, jitter=lambda _base: 0)
    assert delays == []


def test_exhausted_attempts_reraise_the_last_transient_error() -> None:
    """The final attempt raises the last transient error and sleeps one fewer time."""

    def fail() -> None:
        raise TransientEngineError("still busy")

    delays: list[float] = []
    with pytest.raises(TransientEngineError, match="still busy"):
        retry(fail, attempts=3, base_delay=1.0, sleep=delays.append, jitter=lambda _base: 0)
    assert delays == [1.0, 2.0]


def test_permanent_after_a_transient_stops() -> None:
    """A later permanent failure cancels the rest of the schedule."""
    calls = {"n": 0}

    def fail() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientEngineError("later")
        raise PermanentEngineError("no")

    delays: list[float] = []
    with pytest.raises(PermanentEngineError):
        retry(fail, attempts=5, base_delay=1.0, sleep=delays.append, jitter=lambda _base: 0)
    assert calls["n"] == 2
    assert delays == [1.0]
