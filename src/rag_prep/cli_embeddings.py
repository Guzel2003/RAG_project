"""CLI для embeddings."""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline_embeddings import run_embeddings_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-embed", description="RAG Embeddings Pipeline")
    parser.add_argument("--config", default="config/embeddings.yaml")
    args = parser.parse_args()

    try:
        result = run_embeddings_pipeline(args.config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()