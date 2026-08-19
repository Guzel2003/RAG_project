"""CLI для чанкинга."""

from __future__ import annotations

import argparse
import json

from .config_chunking import load_chunking_config
from .pipeline_chunking import run_chunking_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-chunk", description="RAG Chunking Pipeline")
    parser.add_argument("--config", default="config/chunking.yaml")
    args = parser.parse_args()

    config = load_chunking_config(args.config)
    result = run_chunking_pipeline(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()