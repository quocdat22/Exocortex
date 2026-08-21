"""Golden dataset parser for RAG chunking benchmark evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldenSample:
    """A single evaluation sample from the Golden Dataset."""

    entry_id: int
    question: str
    ground_truth: str
    reference_pages: list[int]
    reference_location: str
    excerpt_context: str


def parse_pages_from_reference(ref_str: str) -> list[int]:
    """Parse page numbers like 'Page 1', 'Page 4–5', 'Pages 15–16' into a list of ints."""
    # Match patterns like Page 1, Pages 1-23, Page 4–5
    match_range = re.search(r"Pages?\s+(\d+)\s*[–\-]\s*(\d+)", ref_str, re.IGNORECASE)
    if match_range:
        start, end = int(match_range.group(1)), int(match_range.group(2))
        return list(range(start, end + 1))

    match_single = re.search(r"Pages?\s+(\d+)", ref_str, re.IGNORECASE)
    if match_single:
        return [int(match_single.group(1))]

    # Fallback to any numbers found
    numbers = re.findall(r"\b\d+\b", ref_str)
    return [int(n) for n in numbers] if numbers else []


def _clean_excerpt(excerpt: str) -> str:
    """Clean blockquote markdown markers and quotes from excerpt."""
    lines = [re.sub(r"^\s*>\s*", "", line) for line in excerpt.splitlines()]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n*---+\s*$", "", text).strip()
    return text.strip('"').strip()


def load_golden_dataset(
    path: Path | str = "data/GoldenDatset.md",
) -> list[GoldenSample]:
    """Load and parse Golden Dataset from markdown file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {path}")

    content = path.read_text(encoding="utf-8")

    # Regex matching entries with optional markdown styling
    pattern = re.compile(
        r"Entry\s+(\d+).*?"
        r"(?:\*?\s*\*\*Question:\*\*|Question:)\s*(.*?)"
        r"(?:\*?\s*\*\*Ground Truth Answer:\*\*|Ground Truth Answer:)\s*(.*?)"
        r"(?:\*?\s*\*\*Reference Location:\*\*|Reference Location:)\s*(.*?)"
        r"(?:\*?\s*\*\*Excerpt Context:\*\*|Excerpt Context:)\s*(.*?)"
        r"(?=(?:(?:###\s*)?Entry\s+\d+|\Z))",
        re.DOTALL,
    )

    matches = pattern.findall(content)
    samples: list[GoldenSample] = []

    for entry_id_str, q, gt, ref_loc, excerpt in matches:
        entry_id = int(entry_id_str)
        ref_pages = parse_pages_from_reference(ref_loc)
        excerpt_clean = _clean_excerpt(excerpt)
        samples.append(
            GoldenSample(
                entry_id=entry_id,
                question=q.strip(),
                ground_truth=gt.strip(),
                reference_pages=ref_pages,
                reference_location=ref_loc.strip(),
                excerpt_context=excerpt_clean,
            )
        )

    return samples
