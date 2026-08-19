"""Точка входа для python -m rag_prep."""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "chunk":
    from .cli_chunking import main
    sys.argv.pop(1)
elif len(sys.argv) > 1 and sys.argv[1] == "embed":
    from .cli_embeddings import main
    sys.argv.pop(1)
else:
    from .cli import main

if __name__ == "__main__":
    main()