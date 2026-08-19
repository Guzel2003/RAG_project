"""CLI точка входа."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-prep", description="RAG Data Preparation Pipeline")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_pipeline(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()