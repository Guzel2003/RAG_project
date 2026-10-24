# agent/tools.py
import os
import sys
import math
import re

# ==========================================
# 1. НАСТРОЙКА ПУТЕЙ
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Импортируем саму функцию запроса из твоего RAG
try:
    from src.rag_prep.rag_chain import run_rag_query

    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Не удалось импортировать run_rag_query: {e}")
    RAG_AVAILABLE = False


# ==========================================
# 2. ИНИЦИАЛИЗАЦИЯ ЗАВИСИМОСТЕЙ RAG
# ==========================================
def get_rag_dependencies():
    """
    Создает и возвращает qdrant_client, collection_name и embedding_func
    на основе настроек из твоих config-файлов.
    """
    try:
        # 1. Инициализация Qdrant (локально, путь из paths.storage_dir)
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(path="data/qdrant_storage")

        # 2. Инициализация модели эмбеддингов (модель из embedding.model)
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        def embedding_func(text: str):
            return embed_model.encode(text).tolist()

        # 3. Имя коллекции (из vector_store.collection_name)
        collection_name = "construction_docs"

        return qdrant_client, collection_name, embedding_func

    except Exception as e:
        raise RuntimeError(f"Не удалось инициализировать RAG зависимости: {e}")


# ==========================================
# 3. TOOL: ПОИСК В RAG
# ==========================================
def search_rag(query: str) -> dict:
    """
    Ищет формулы и нормативы в твоей реальной базе Qdrant.
    """
    if not RAG_AVAILABLE:
        return {"status": "error", "error": "Модуль rag_chain не найден. Проверь пути."}

    try:
        # Получаем живые объекты для работы
        qdrant_client, collection_name, embedding_func = get_rag_dependencies()

        # Вызываем твою функцию поиска
        result = run_rag_query(
            question=query,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            embedding_func=embedding_func,
            top_k=3
        )

        # Извлекаем контекст и источники
        context = result.get("context", "Контекст не найден.")
        sources = [
            chunk.get("metadata", {}).get("file_name", "Неизвестный источник")
            for chunk in result.get("found_chunks", [])
        ]

        return {
            "status": "success",
            "context": context,
            "sources": list(set(sources))  # Убираем дубликаты
        }

    except Exception as e:
        return {
            "status": "error",
            "error": f"Ошибка при поиске в базе знаний: {str(e)}"
        }


# ==========================================
# 4. TOOL: КАЛЬКУЛЯТОР ВЫРАЖЕНИЙ
# ==========================================
def calculate_expression(expression: str) -> dict:
    """Безопасно вычисляет математическое выражение."""
    try:
        if not re.match(r'^[\d\.\+\-\*\/\(\)\s\w]+$', expression):
            return {"status": "error", "error": "Выражение содержит недопустимые символы"}

        safe_dict = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e
        }

        result = eval(expression, safe_dict)

        return {
            "status": "success",
            "result": round(result, 4),
            "expression": expression
        }
    except ZeroDivisionError:
        return {"status": "error", "error": "Деление на ноль"}
    except Exception as e:
        return {"status": "error", "error": f"Ошибка вычисления: {str(e)}"}