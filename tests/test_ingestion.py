from __future__ import annotations

from pathlib import Path

from app.ingestion import ingest_sources


def test_empty_and_unreachable_sources_are_handled(tmp_path: Path) -> None:
    good_file = tmp_path / "good.txt"
    good_file.write_text(
        (
            "Audit logs improved oversight quality in pilot programs, "
            "and teams resolved incidents faster when evidence was available."
        ),
        encoding="utf-8",
    )
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   ", encoding="utf-8")

    def fake_reader(url: str) -> tuple[str, str]:
        if "good.example" in url:
            return (
                "Good URL Source",
                (
                    "Independent reviewers found that model cards helped regulators compare systems "
                    "and made incident triage easier in practice."
                ),
            )
        raise RuntimeError("network failure")

    sources, errors = ingest_sources(
        urls=["https://good.example/report", "https://bad.example/report"],
        folder_path=str(tmp_path),
        timeout_seconds=1,
        url_reader=fake_reader,
    )

    assert len(sources) == 2
    reasons = [error.reason for error in errors]
    assert any(reason.startswith("fetch_failed") for reason in reasons)
    assert any(reason == "empty_or_unclear_content" for reason in reasons)
