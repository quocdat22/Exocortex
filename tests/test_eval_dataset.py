from pathlib import Path

import pytest

from exocortex.eval.dataset import (
    GoldenSample,
    load_golden_dataset,
    parse_pages_from_reference,
)


def test_load_golden_dataset():
    path = Path("data/GoldenDatset.md")
    dataset = load_golden_dataset(path)
    assert len(dataset) == 22

    sample1 = dataset[0]
    assert sample1.entry_id == 1
    assert "production machine learning system" in sample1.question.lower()
    assert len(sample1.ground_truth) > 20
    assert 1 in sample1.reference_pages
    assert len(sample1.excerpt_context) > 0

    # Test all samples
    for i, sample in enumerate(dataset, start=1):
        assert sample.entry_id == i
        assert len(sample.question) > 0
        assert len(sample.ground_truth) > 0
        assert len(sample.reference_pages) > 0
        assert len(sample.reference_location) > 0
        assert len(sample.excerpt_context) > 0


def test_load_golden_dataset_default_path():
    dataset = load_golden_dataset()
    assert len(dataset) == 22


def test_load_golden_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_golden_dataset("non_existent_dataset.md")


def test_parse_pages_from_reference():
    assert parse_pages_from_reference("Page 1, Section: Overview") == [1]
    assert parse_pages_from_reference("Page 4–5, Section: Point 2") == [4, 5]
    assert parse_pages_from_reference("Pages 15-16, Section: Priorities") == [15, 16]
    assert parse_pages_from_reference("Page 6, Section: (Point 5), Paragraph 1–2.") == [
        6
    ]
    assert parse_pages_from_reference("Unknown location with no numbers") == []
    assert parse_pages_from_reference("Section 12, Paragraph 3") == [12, 3]


def test_golden_sample_dataclass():
    sample = GoldenSample(
        entry_id=1,
        question="What is ML?",
        ground_truth="Machine Learning is...",
        reference_pages=[1, 2],
        reference_location="Page 1–2",
        excerpt_context="ML is...",
    )
    assert sample.entry_id == 1
    assert sample.question == "What is ML?"
    assert sample.reference_pages == [1, 2]
