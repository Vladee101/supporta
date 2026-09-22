"""Слияние выдач по RRF (ADR-010): чистая функция, без базы."""

from __future__ import annotations

import uuid

from app.services.retrieval import RRF_K, rrf_fuse

A, B, C, D = (uuid.UUID(int=n) for n in range(1, 5))


def test_agreement_of_both_branches_beats_a_single_first_place():
    """Документ из обеих выдач выше, чем первый ранг только одной: в этом смысл гибрида."""
    fused = rrf_fuse([[A, B], [C, B]])
    assert fused[0] == B


def test_score_is_sum_of_reciprocal_ranks():
    fused = rrf_fuse([[A, B, C], [C]])
    # C: 1/(60+3) + 1/(60+1) > A: 1/(60+1) > B: 1/(60+2)
    assert fused == [C, A, B]
    assert 1 / (RRF_K + 3) + 1 / (RRF_K + 1) > 1 / (RRF_K + 1)


def test_single_ranking_keeps_its_order():
    """Лексическая ветка пуста (запрос из стоп-слов) - гибрид равен векторному поиску."""
    assert rrf_fuse([[A, B, C], []]) == [A, B, C]


def test_ties_are_broken_deterministically():
    """Порядок выдачи пишется в трейс - он не должен зависеть от порядка обхода."""
    assert rrf_fuse([[A, B], [B, A]]) == rrf_fuse([[B, A], [A, B]])
    assert rrf_fuse([[D], [C]]) == [C, D]


def test_empty_input_gives_empty_output():
    assert rrf_fuse([[], []]) == []
