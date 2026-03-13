from __future__ import annotations

import argparse
import asyncio
import sys

from research_agent.core.graph import run_research_digest
from research_agent.models import RuntimeConfig
from research_agent.config import configure_langsmith, get_settings


def _parse_urls(raw_values: list[str] | None) -> list[str]:
    urls: list[str] = []
    for value in raw_values or []:
        for candidate in value.split(","):
            cleaned = candidate.strip()
            if cleaned:
                urls.append(cleaned)
    return urls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the semantic research digest pipeline.")
    parser.add_argument("--topic", required=True, help="Research topic.")
    parser.add_argument("--urls", nargs="*", help="One or more URLs.")
    parser.add_argument("--folder-path", help="Folder containing .txt/.html files.")
    parser.add_argument("--output-dir", default="outputs/latest", help="Output directory.")
    parser.add_argument(
        "--grouping-threshold",
        type=float,
        default=None,
        help="Distance threshold for semantic clustering.",
    )
    parser.add_argument(
        "--api-provider",
        choices=["openai", "groq", "anthropic"],
        default="groq",
        help="LLM provider used for optional claim extraction.",
    )
    parser.add_argument("--api-key", default=None, help="Runtime API key (optional).")
    parser.add_argument(
        "--use-vector-store",
        action="store_true",
        help="Enable optional Chroma persistence for clusters.",
    )
    return parser


async def _run_async(args: argparse.Namespace) -> int:
    urls = _parse_urls(args.urls)
    runtime = RuntimeConfig(
        api_provider=args.api_provider,
        api_key=args.api_key,
        grouping_distance_threshold=args.grouping_threshold,
        use_vector_store=args.use_vector_store,
    )

    try:
        result = await run_research_digest(
            topic=args.topic,
            urls=urls,
            folder_path=args.folder_path,
            uploaded_sources=None,
            output_dir=args.output_dir,
            runtime=runtime,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    print("Research digest completed.")
    print(f"Sources processed: {len(result.sources)}")
    print(f"Claims extracted: {len(result.claims)}")
    print(f"Semantic clusters: {len(result.clusters)}")
    print(f"Ingestion issues: {len(result.ingestion_errors)}")
    print(f"Processing issues: {len(result.processing_errors)}")
    print(f"Digest file: {result.digest_path}")
    print(f"Sources file: {result.sources_json_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_langsmith(get_settings())
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
