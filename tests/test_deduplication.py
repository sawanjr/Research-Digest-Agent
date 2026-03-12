from __future__ import annotations

from pathlib import Path

from app.ingestion import ingest_sources


def test_duplicate_content_is_deduplicated(tmp_path: Path) -> None:
    duplicate_text = (
        "Mandatory reporting helped investigators identify the root cause of model incidents, "
        "and the same policy recommendation appears in two mirrored files."
    )
    (tmp_path / "first.txt").write_text(duplicate_text, encoding="utf-8")
    (tmp_path / "second.txt").write_text(duplicate_text, encoding="utf-8")

    sources, errors = ingest_sources(urls=[], folder_path=str(tmp_path))

    assert len(sources) == 1
    assert any(error.reason.startswith("duplicate_content_of_") for error in errors)
