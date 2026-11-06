"""Pure planning of silence-aware chunk boundaries.

Cuts sit on silences so a word is never split. That removes any need to overlap
chunks or to de-duplicate text at the joins. When no silence is close enough,
the planner hard-cuts at the target time.
"""

import itertools
from collections.abc import Sequence

from transcriber.models import ChunkSpec

# A cut that leaves less than a second on either side is not worth a chunk.
MIN_CHUNK_SECONDS = 1.0


def find_cuts(
    silences: Sequence[float],
    total: float,
    target: float,
    window: float,
) -> list[float]:
    """Return interior cut points, strictly increasing, none at 0 or ``total``.

    Walking stops once the remaining audio is at most ``target + window``, so
    the last chunk stays bounded and short files produce no cuts at all.
    """
    if total <= target + window or target <= 0:
        return []
    cuts: list[float] = []
    previous = 0.0
    while total - previous > target + window:
        chosen = _choose_cut(silences, previous + target, window, previous, total)
        if chosen is None:
            break
        cuts.append(chosen)
        previous = chosen
    return cuts


def build_plan(cuts: Sequence[float], total: float) -> list[ChunkSpec]:
    """Turn cut points into contiguous chunks covering ``[0, total]``."""
    points = [0.0]
    for cut in cuts:
        if points[-1] < cut < total:
            points.append(float(cut))
    points.append(max(total, 0.0))
    return [
        ChunkSpec(index=index, start=start, end=end)
        for index, (start, end) in enumerate(itertools.pairwise(points))
    ]


def _choose_cut(
    silences: Sequence[float],
    goal: float,
    window: float,
    previous: float,
    total: float,
) -> float | None:
    earliest = previous + MIN_CHUNK_SECONDS
    latest = total - MIN_CHUNK_SECONDS
    if goal < earliest or goal > latest:
        return None
    candidates = [
        silence
        for silence in silences
        if earliest <= silence <= latest and abs(silence - goal) <= window
    ]
    if not candidates:
        return goal
    return min(candidates, key=lambda silence: (abs(silence - goal), silence))
