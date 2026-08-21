"""Tests for retrieval and operational evaluation metrics."""

from exocortex.eval.metrics import (
    calculate_hit_rate_and_mrr,
    compute_chunk_statistics,
    parse_chunk_page_numbers,
)
from exocortex.ingestion import Chunk
from exocortex.vectorstore import SearchResult


def test_hit_rate_and_mrr():
    ref_pages = [4, 5]
    results = [
        SearchResult(
            text="c1", metadata={"page_numbers": "2,3"}, distance=0.1, chunk_id="1"
        ),
        SearchResult(
            text="c2", metadata={"page_numbers": "4,5"}, distance=0.2, chunk_id="2"
        ),
        SearchResult(
            text="c3", metadata={"page_numbers": "5"}, distance=0.3, chunk_id="3"
        ),
    ]
    hit, mrr = calculate_hit_rate_and_mrr(results, ref_pages, top_k=3)
    assert hit == 1.0
    assert mrr == 0.5  # First match at rank 2 (1/2)


def test_hit_rate_and_mrr_rank_1():
    ref_pages = [2]
    results = [
        SearchResult(
            text="c1", metadata={"page_numbers": "2,3"}, distance=0.1, chunk_id="1"
        ),
        SearchResult(
            text="c2", metadata={"page_numbers": "4,5"}, distance=0.2, chunk_id="2"
        ),
    ]
    hit, mrr = calculate_hit_rate_and_mrr(results, ref_pages, top_k=2)
    assert hit == 1.0
    assert mrr == 1.0


def test_hit_rate_and_mrr_no_hit():
    ref_pages = [10]
    results = [
        SearchResult(
            text="c1", metadata={"page_numbers": "2,3"}, distance=0.1, chunk_id="1"
        ),
        SearchResult(
            text="c2", metadata={"page_numbers": "4,5"}, distance=0.2, chunk_id="2"
        ),
    ]
    hit, mrr = calculate_hit_rate_and_mrr(results, ref_pages, top_k=5)
    assert hit == 0.0
    assert mrr == 0.0


def test_hit_rate_and_mrr_empty_inputs():
    assert calculate_hit_rate_and_mrr([], [1, 2]) == (0.0, 0.0)
    assert calculate_hit_rate_and_mrr(
        [
            SearchResult(
                text="c", metadata={"page_numbers": "1"}, distance=0.1, chunk_id="1"
            )
        ],
        [],
    ) == (0.0, 0.0)
    assert calculate_hit_rate_and_mrr([], []) == (0.0, 0.0)


def test_hit_rate_and_mrr_top_k():
    ref_pages = [5]
    results = [
        SearchResult(
            text="c1", metadata={"page_numbers": "1"}, distance=0.1, chunk_id="1"
        ),
        SearchResult(
            text="c2", metadata={"page_numbers": "2"}, distance=0.2, chunk_id="2"
        ),
        SearchResult(
            text="c3", metadata={"page_numbers": "3"}, distance=0.3, chunk_id="3"
        ),
        SearchResult(
            text="c4", metadata={"page_numbers": "5"}, distance=0.4, chunk_id="4"
        ),
    ]
    # top_k=3 should not reach rank 4
    hit_3, mrr_3 = calculate_hit_rate_and_mrr(results, ref_pages, top_k=3)
    assert hit_3 == 0.0
    assert mrr_3 == 0.0

    # top_k=5 should reach rank 4
    hit_5, mrr_5 = calculate_hit_rate_and_mrr(results, ref_pages, top_k=5)
    assert hit_5 == 1.0
    assert mrr_5 == 0.25


def test_parse_chunk_page_numbers():
    assert parse_chunk_page_numbers({"page_numbers": [1, 2, 3]}) == [1, 2, 3]
    assert parse_chunk_page_numbers({"page_numbers": "1, 2, 3"}) == [1, 2, 3]
    assert parse_chunk_page_numbers({"page_numbers": "4,5"}) == [4, 5]
    assert parse_chunk_page_numbers({"page_numbers": "?"}) == []
    assert parse_chunk_page_numbers({"page_numbers": ""}) == []
    assert parse_chunk_page_numbers({}) == []
    assert parse_chunk_page_numbers({"page_numbers": "invalid, none"}) == []


def test_chunk_statistics():
    chunks = [
        Chunk(
            text="Hello world",
            document_id="1",
            filename="f",
            page_numbers=[1],
            chunk_index=0,
        ),
        Chunk(
            text="Another test chunk with more tokens",
            document_id="1",
            filename="f",
            page_numbers=[1],
            chunk_index=1,
        ),
    ]
    stats = compute_chunk_statistics(chunks)
    assert stats["chunk_count"] == 2
    assert stats["mean_chars"] > 0
    assert stats["median_chars"] > 0
    assert stats["min_chars"] == 11
    assert stats["max_chars"] == 35


def test_chunk_statistics_empty():
    stats = compute_chunk_statistics([])
    assert stats == {
        "chunk_count": 0,
        "mean_chars": 0.0,
        "median_chars": 0.0,
        "min_chars": 0,
        "max_chars": 0,
    }


def test_chunk_statistics_single():
    chunks = [
        Chunk(
            text="Test", document_id="1", filename="f", page_numbers=[1], chunk_index=0
        ),
    ]
    stats = compute_chunk_statistics(chunks)
    assert stats == {
        "chunk_count": 1,
        "mean_chars": 4.0,
        "median_chars": 4.0,
        "min_chars": 4,
        "max_chars": 4,
    }
