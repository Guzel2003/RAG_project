"""Точка входа для python -m rag_prep."""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "chunk":
    from .cli_chunking import main
    sys.argv.pop(1)  # убираем "chunk" из аргументов
else:
    from .cli_prepare import main

if __name__ == "__main__":
    main()