"""Chunk planning stays deterministic and never emits a sliver."""

import itertools

from transcriber.audio.chunking import build_plan, find_cuts
from transcriber.models import ChunkSpec


def test_snaps_to_the_nearest_silence_inside_the_window() -> None:
    """A silence nearer the goal wins over a farther one that is still in range."""
    cuts = find_cuts([12.0, 31.0, 33.0, 58.0], total=100, target=30, window=5)
    # Next goal is one target after the snapped cut, so the third cut is 88 not 90.
    assert cuts == [31.0, 58.0, 88.0]


def test_hard_cut_when_nothing_is_near_the_goal() -> None:
    """Missing silences fall back to the target time instead of drifting."""
    assert find_cuts([], total=100, target=30, window=5) == [30.0, 60.0, 90.0]
    assert find_cuts([20.0], total=100, target=30, window=5)[0] == 30.0


def test_short_audio_is_a_single_chunk() -> None:
    """Files no longer than target + window are not split and skip silence search."""
    plan = build_plan(find_cuts([5.0], total=20, target=30, window=5), 20)
    assert plan == [ChunkSpec(index=0, start=0, end=20)]


def test_cuts_are_strictly_increasing_and_chunks_cover_the_file() -> None:
    """Every chunk is at least one second and the plan starts at 0 and ends at total."""
    silences = [5.0, 5.0, 12.0, 19.0, 41.0, 48.0, 73.0, 90.0, 99.0]
    cuts = find_cuts(silences, total=120, target=25, window=8)
    assert all(later > earlier for earlier, later in itertools.pairwise(cuts))
    plan = build_plan(cuts, 120)
    assert plan[0].start == 0
    assert plan[-1].end == 120
    assert all(chunk.end - chunk.start >= 1 for chunk in plan)


def test_last_chunk_stays_within_the_window_bound() -> None:
    """Planning stops once the tail is no longer than target + window."""
    plan = build_plan(find_cuts([], total=250, target=100, window=10), 250)
    assert plan[-1].end - plan[-1].start <= 110
    assert [chunk.start for chunk in plan] == [0, 100, 200]
