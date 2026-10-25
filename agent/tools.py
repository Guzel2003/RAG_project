# agent/tools.py
import os
import sys
import ast
import operator
import re

# ==========================================
# 1. НАСТРОЙКА ПУТЕЙ
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Импортируем саму функцию запроса из RAG
try:
    from src.rag_prep.rag_chain import run_rag_query

    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Не удалось импортировать run_rag_query: {e}")
    RAG_AVAILABLE = False


# ==========================================
# 2. ИНИЦИАЛИЗАЦИЯ ЗАВИСИМОСТЕЙ RAG (FastEmbed)
# ==========================================
def get_rag_dependencies():
    """
    Создает и возвращает qdrant_client, collection_name и embedding_func.
    Используем FastEmbed с той же моделью, что и при индексации базы!
    """
    try:
        from qdrant_client import QdrantClient
        from fastembed import TextEmbedding

        # 1. Инициализация Qdrant
        qdrant_client = QdrantClient(path="data/qdrant_storage")

        # 2. Инициализация FastEmbed (ТОЧНО ТА ЖЕ МОДЕЛЬ, ЧТО В ВАШЕМ МАППИНГЕ!)
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embed_model = TextEmbedding(model_name)

        def embedding_func(text: str):
            # FastEmbed возвращает генератор. Берем первый элемент и преобразуем в список float
            return list(embed_model.embed(text))[0].tolist()

        # 3. Имя коллекции
        collection_name = "construction_docs"

        return qdrant_client, collection_name, embedding_func

    except Exception as e:
        raise RuntimeError(f"Не удалось инициализировать RAG зависимости: {e}")


# ==========================================
# 3. TOOL: ПОИСК В RAG
# ==========================================
def search_rag(query: str) -> dict:
    """
    Ищет формулы и нормативы в базе Qdrant.
    """
    if not RAG_AVAILABLE:
        return {"status": "error", "error": "Модуль rag_chain не найден. Проверь пути."}

    try:
        qdrant_client, collection_name, embedding_func = get_rag_dependencies()

        # Вызываем обновленную функцию поиска
        result = run_rag_query(
            question=query,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            embedding_func=embedding_func,
            top_k=3
        )

        # Просто возвращаем результат
        return result

    except Exception as e:
        return {
            "status": "error",
            "error": f"Ошибка при поиске в базе знаний: {str(e)}"
        }


# ==========================================
# 4. TOOL: БЕЗОПАСНЫЙ КАЛЬКУЛЯТОР (Без eval!)
# ==========================================
def calculate_expression(expression: str) -> dict:
    """
    Безопасно вычисляет математическое выражение через AST (абстрактное синтаксическое дерево).
    Полностью исключает использование eval() и выполнение произвольного кода.
    """
    # Разрешенные математические операторы
    allowed_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }

    def _safe_eval(node):
        """Рекурсивно вычисляет узлы AST."""
        if isinstance(node, ast.Constant):  # Для Python 3.8+ (числа)
            return node.value
        elif isinstance(node, ast.BinOp):  # Бинарные операции (2 + 2)
            return allowed_ops[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp):  # Унарные операции (-5)
            return allowed_ops[type(node.op)](_safe_eval(node.operand))
        else:
            raise ValueError("Выражение содержит недопустимые символы или функции")

    try:
        # Парсим строку в AST. Если там будет код вроде "import os", парсинг упадет или мы его отсечем
        tree = ast.parse(expression, mode='eval')

        # Вычисляем
        result = _safe_eval(tree.body)

        return {
            "status": "success",
            "result": round(result, 4),
            "expression": expression
        }
    except ZeroDivisionError:
        return {"status": "error", "error": "Деление на ноль"}
    except Exception as e:
        return {"status": "error", "error": f"Ошибка вычисления: {str(e)}"}