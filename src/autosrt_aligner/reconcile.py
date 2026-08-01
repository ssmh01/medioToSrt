"""Merge overlapping chunk alignment evidence without inventing timestamps."""

from __future__ import annotations

import math
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass

from .models import AlignmentToken, ChunkAlignment, SourceDocument
from .text import normalize_for_alignment


@dataclass(frozen=True)
class ReconcileReport:
    source_token_coverage: float
    duplicate_token_count: int
    overlap_count: int
    time_reversal_count: int
    low_confidence_count: int
    issues: tuple[str, ...]
    missing_confidence_count: int = 0
    resolved_overlap_count: int = 0
    uncovered_source_ranges: tuple[tuple[int, int], ...] = ()
    confidence_available: bool = True

    @property
    def publishable(self) -> bool:
        return not self.issues and self.source_token_coverage >= 1.0


@dataclass(frozen=True)
class _Candidate:
    token: AlignmentToken
    required_coverage: int
    core_coverage: int
    span_length: int

    @property
    def start_char(self) -> int:
        return self.token.start_char or 0

    @property
    def end_char(self) -> int:
        return self.token.end_char or self.start_char

    @property
    def confidence(self) -> float:
        return self.token.confidence if self.token.confidence is not None else 0.0

    @property
    def score(self) -> tuple[float, ...]:
        """Score additive interval selections without changing timestamps."""

        return (
            float(self.required_coverage),
            float(self.core_coverage),
            self.confidence,
            self.time_stability,
            -self.token.duration,
            float(self.span_length),
            1.0,
        )

    @property
    def time_stability(self) -> float:
        if not math.isfinite(self.token.start) or not math.isfinite(self.token.end):
            return -1.0
        return 1.0 if self.token.end >= self.token.start else -1.0


def reconcile_chunk_alignments(
    chunks: list[ChunkAlignment],
    source: SourceDocument,
    *,
    min_confidence: float = 0.20,
) -> tuple[list[AlignmentToken], ReconcileReport]:
    """Select a non-overlapping set of real source intervals.

    Exact duplicates are deduplicated by confidence and core ownership. For
    non-identical overlaps, weighted interval selection prefers complete source
    coverage and core-owned evidence. It never splits a token or manufactures a
    timestamp; unresolved conflicts remain visible and block publication.
    """

    required_positions = _required_positions(source)
    candidates: list[_Candidate] = []
    for chunk_alignment in chunks:
        chunk = chunk_alignment.chunk
        core_start = (
            chunk.core_source_start
            if chunk.core_source_start is not None
            else chunk.source_start
        )
        core_end = (
            chunk.core_source_end
            if chunk.core_source_end is not None
            else chunk.source_end
        )
        for token in chunk_alignment.tokens:
            if token.start_char is None or token.end_char is None:
                continue
            if token.end_char <= token.start_char:
                continue
            if (
                not math.isfinite(token.start)
                or not math.isfinite(token.end)
                or token.start < 0
                or token.end < token.start
            ):
                continue
            start = token.start_char
            end = token.end_char
            candidates.append(
                _Candidate(
                    token=token,
                    required_coverage=_count_positions(required_positions, start, end),
                    core_coverage=_count_positions(
                        required_positions,
                        max(start, core_start),
                        min(end, core_end),
                    ),
                    span_length=end - start,
                )
            )

    unique_candidates, duplicate_count = _deduplicate_exact_candidates(candidates)
    selected_candidates = _select_non_overlapping_candidates(unique_candidates)
    selected_tokens = [candidate.token for candidate in selected_candidates]
    if _count_time_reversals(selected_tokens):
        monotonic_candidates = _select_monotonic_candidates(unique_candidates)
        current_coverage = len(
            _covered_required_positions(required_positions, selected_tokens)
        )
        monotonic_coverage = len(
            _covered_required_positions(
                required_positions,
                [candidate.token for candidate in monotonic_candidates],
            )
        )
        if (
            monotonic_coverage >= current_coverage
            and _count_time_reversals(
                [candidate.token for candidate in monotonic_candidates]
            )
            < _count_time_reversals(selected_tokens)
        ):
            selected_candidates = monotonic_candidates
    selected = [candidate.token for candidate in selected_candidates]
    selected.sort(key=lambda token: (token.start_char or 0, token.start, token.end))

    selected_ids = {id(candidate) for candidate in selected_candidates}
    selected_required = _covered_required_positions(required_positions, selected)
    overlap_count, resolved_overlap_count = _count_overlaps(
        unique_candidates,
        selected_candidates,
        selected_ids,
        required_positions,
        selected_required,
    )
    time_reversal_count = sum(
        1
        for previous, current in zip(selected, selected[1:])
        if current.start < previous.start - 0.05
    )
    low_confidence_count = sum(
        1
        for token in selected
        if token.confidence is not None and token.confidence < min_confidence
    )
    missing_confidence_count = sum(1 for token in selected if token.confidence is None)
    confidence_available = any(
        chunk_alignment.confidence_available for chunk_alignment in chunks
    )
    uncovered = tuple(
        _contiguous_ranges(sorted(required_positions - selected_required))
    )
    coverage = len(selected_required) / len(required_positions) if required_positions else 1.0

    issues: list[str] = []
    if not selected:
        issues.append("没有可用的对齐 token")
    if overlap_count:
        issues.append(f"存在 {overlap_count} 个未解决的 source token 重叠")
    if time_reversal_count:
        issues.append(f"存在 {time_reversal_count} 个时间倒退")
    if coverage < 1.0:
        issues.append(f"source token 覆盖率不足: {coverage:.6f}")
    if low_confidence_count:
        issues.append(f"存在 {low_confidence_count} 个低置信 token")
    if missing_confidence_count and confidence_available:
        issues.append(f"存在 {missing_confidence_count} 个缺少置信度 token")

    report = ReconcileReport(
        source_token_coverage=round(coverage, 6),
        duplicate_token_count=duplicate_count,
        overlap_count=overlap_count,
        time_reversal_count=time_reversal_count,
        low_confidence_count=low_confidence_count,
        issues=tuple(issues),
        missing_confidence_count=missing_confidence_count,
        resolved_overlap_count=resolved_overlap_count,
        uncovered_source_ranges=uncovered,
        confidence_available=confidence_available,
    )
    return selected, report


def _deduplicate_exact_candidates(
    candidates: list[_Candidate],
) -> tuple[list[_Candidate], int]:
    groups: dict[tuple[int, int, str], list[_Candidate]] = {}
    for candidate in candidates:
        key = (
            candidate.start_char,
            candidate.end_char,
            normalize_for_alignment(candidate.token.text),
        )
        groups.setdefault(key, []).append(candidate)

    unique: list[_Candidate] = []
    duplicate_count = 0
    for group in groups.values():
        duplicate_count += max(0, len(group) - 1)
        # A deterministic forced aligner such as Qwen does not emit a scalar
        # confidence. Keep its exact duplicate alternatives until the global
        # time-order pass; otherwise a core-owned boundary token can discard
        # the adjacent chunk's more stable real timestamp too early.
        if all(candidate.token.confidence is None for candidate in group):
            unique.extend(group)
        else:
            unique.append(max(group, key=_exact_candidate_rank))
    return unique, duplicate_count


def _exact_candidate_rank(candidate: _Candidate) -> tuple[float, ...]:
    return (
        candidate.confidence,
        float(candidate.core_coverage),
        float(candidate.required_coverage),
        -candidate.token.duration,
        -candidate.token.start,
    )


def _select_non_overlapping_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.end_char,
            candidate.start_char,
            -candidate.required_coverage,
            -candidate.core_coverage,
            -candidate.confidence,
            candidate.token.start,
        ),
    )
    predecessors: list[int] = []
    for index, candidate in enumerate(ordered):
        predecessor = -1
        for previous_index in range(index - 1, -1, -1):
            if ordered[previous_index].end_char <= candidate.start_char:
                predecessor = previous_index
                break
        predecessors.append(predecessor)

    zero_score = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    scores: list[tuple[float, ...]] = [zero_score]
    paths: list[list[int]] = [[]]
    for index, candidate in enumerate(ordered):
        skip_score = scores[index]
        skip_path = paths[index]
        predecessor = predecessors[index]
        base_score = scores[predecessor + 1]
        base_path = paths[predecessor + 1]
        take_score = _add_scores(base_score, candidate.score)
        take_path = [*base_path, index]
        if take_score > skip_score:
            scores.append(take_score)
            paths.append(take_path)
        else:
            scores.append(skip_score)
            paths.append(skip_path)
    return [ordered[index] for index in paths[-1]]


def _select_monotonic_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    """Find a real-token path that also preserves source-time order.

    This is a recovery choice between overlapping model evidence. It never
    changes a timestamp; when no monotonic path keeps the same source
    coverage, the normal selector remains in place and the hard gate reports
    the reversal.
    """

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.start_char,
            candidate.end_char,
            -candidate.required_coverage,
            -candidate.core_coverage,
            -candidate.confidence,
            candidate.token.start,
        ),
    )

    time_values = sorted(
        {
            candidate.token.start
            for candidate in ordered
            if math.isfinite(candidate.token.start)
        }
    )
    if not time_values:
        return []

    State = tuple[int, tuple[float, ...], list[int]]
    tree: list[State | None] = [None] * (len(time_values) + 1)
    states: list[State | None] = [None] * len(ordered)
    by_end = sorted(range(len(ordered)), key=lambda index: ordered[index].end_char)
    eligible_end = 0

    for index, candidate in enumerate(ordered):
        while (
            eligible_end < len(by_end)
            and ordered[by_end[eligible_end]].end_char <= candidate.start_char
        ):
            previous_index = by_end[eligible_end]
            previous_state = states[previous_index]
            if previous_state is not None:
                _fenwick_update(
                    tree,
                    bisect_right(time_values, ordered[previous_index].token.start),
                    previous_state,
                )
            eligible_end += 1

        prefix_end = bisect_right(time_values, candidate.token.start + 0.05)
        previous_state = _fenwick_query(tree, prefix_end)
        if previous_state is None:
            state: State = (candidate.required_coverage, candidate.score, [index])
        else:
            coverage, score, path = previous_state
            state = (
                coverage + candidate.required_coverage,
                _add_scores(score, candidate.score),
                [*path, index],
            )
        states[index] = state

    completed = [state for state in states if state is not None]
    if not completed:
        return []
    _, _, path = max(completed, key=lambda state: (state[0], state[1]))
    return [ordered[index] for index in path]


def _fenwick_update(
    tree: list[tuple[int, tuple[float, ...], list[int]] | None],
    index: int,
    state: tuple[int, tuple[float, ...], list[int]],
) -> None:
    while index < len(tree):
        current = tree[index]
        if current is None or (state[0], state[1]) > (current[0], current[1]):
            tree[index] = state
        index += index & -index


def _fenwick_query(
    tree: list[tuple[int, tuple[float, ...], list[int]] | None],
    index: int,
) -> tuple[int, tuple[float, ...], list[int]] | None:
    best = None
    while index > 0:
        current = tree[index]
        if current is not None and (
            best is None or (current[0], current[1]) > (best[0], best[1])
        ):
            best = current
        index -= index & -index
    return best


def _count_time_reversals(tokens: list[AlignmentToken]) -> int:
    ordered = sorted(tokens, key=lambda token: (token.start_char or 0, token.start))
    return sum(
        1
        for previous, current in zip(ordered, ordered[1:])
        if current.start < previous.start - 0.05
    )


def _add_scores(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(a + b for a, b in zip(left, right))


def _count_overlaps(
    candidates: list[_Candidate],
    selected: list[_Candidate],
    selected_ids: set[int],
    required_positions: set[int],
    selected_required: set[int],
) -> tuple[int, int]:
    unresolved = 0
    resolved = 0
    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        overlaps = [
            chosen
            for chosen in selected
            if _spans_overlap(candidate.start_char, candidate.end_char, chosen.start_char, chosen.end_char)
        ]
        if not overlaps:
            continue
        same_span_text_conflict = any(
            candidate.start_char == chosen.start_char
            and candidate.end_char == chosen.end_char
            and normalize_for_alignment(candidate.token.text)
            != normalize_for_alignment(chosen.token.text)
            for chosen in overlaps
        )
        candidate_required = {
            position
            for position in required_positions
            if candidate.start_char <= position < candidate.end_char
        }
        uncovered_by_selection = candidate_required - selected_required
        if same_span_text_conflict or uncovered_by_selection:
            unresolved += 1
        else:
            resolved += 1
    return unresolved, resolved


def _spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def _required_positions(source: SourceDocument) -> set[int]:
    return {
        index
        for index, char in enumerate(source.display_text)
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    }


def _count_positions(positions: set[int], start: int, end: int) -> int:
    if end <= start:
        return 0
    return sum(1 for position in positions if start <= position < end)


def _covered_required_positions(
    required_positions: set[int],
    tokens: list[AlignmentToken],
) -> set[int]:
    covered: set[int] = set()
    for token in tokens:
        start = max(0, token.start_char or 0)
        end = min(max(start, token.end_char or start), max(required_positions, default=0) + 1)
        covered.update(position for position in required_positions if start <= position < end)
    return covered


def _contiguous_ranges(positions: list[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            ranges.append((start, previous + 1))
            start = position
        previous = position
    ranges.append((start, previous + 1))
    return ranges


def _source_token_coverage(source: SourceDocument, tokens: list[AlignmentToken]) -> float:
    required = _required_positions(source)
    if not required:
        return 1.0
    covered = _covered_required_positions(required, tokens)
    return len(covered) / len(required)
