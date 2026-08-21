"""CLI для vector store."""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline_vector_store import run_vector_store_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-vector-store", description="RAG Vector Store Pipeline")
    parser.add_argument("--config", default="config/vector_store.yaml")
    args = parser.parse_args()

    try:
        result = run_vector_store_pipeline(args.config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()