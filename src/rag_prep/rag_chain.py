import os
from typing import List, Dict, Any


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Формирует единый текстовый контекст из найденных чанков.
    """
    if not chunks:
        return "Контекст не найден."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get('metadata', {}).get('document_name', 'Неизвестный источник')
        text = chunk.get('text', '')
        part = f"[Фрагмент {i}]\nИсточник: {source}\nТекст: {text}"
        context_parts.append(part)

    return "\n\n".join(context_parts)


def run_rag_query(
        question: str,
        qdrant_client,
        collection_name: str,
        embedding_func,
        top_k: int = 3
) -> Dict[str, Any]:
    """
    Полная цепочка RAG
    """
    try:
        # 1. Векторизуем вопрос
        query_vector = embedding_func(question)

        # 2. Поиск в Vector Store (Qdrant)
        search_result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        found_chunks = []
        sources = []

        for point in search_result.points:
            found_chunks.append({
                "text": point.payload.get("text", ""),
                "metadata": point.payload,
                "score": point.score
            })
            # Собираем уникальные имена источников
            source = point.payload.get("document_name", "Неизвестный источник")
            if source not in sources:
                sources.append(source)

        # 3. Формируем контекст
        context = build_context(found_chunks)

        # 4. ВОЗВРАЩАЕМ ДАННЫЕ АГЕНТУ
        return {
            "status": "success",
            "context": context,
            "sources": sources,
            "found_chunks": found_chunks
        }

    except Exception as e:
        return {
            "status": "error",
            "error": f"Ошибка при поиске в базе знаний: {str(e)}"
        }