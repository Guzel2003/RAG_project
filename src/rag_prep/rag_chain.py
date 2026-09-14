import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

# Загружаем переменные из .env файла
load_dotenv()

# Инициализация клиента OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Пункт 6: Формирует единый текстовый контекст из найденных чанков.
    Каждый фрагмент отделяется разделителем и содержит метаданные.
    """
    if not chunks:
        return "Контекст не найден."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # Берем название документа из metadata
        source = chunk.get('metadata', {}).get('document_name', 'Неизвестный источник')
        text = chunk.get('text', '')


        part = f"[Фрагмент {i}]\nИсточник: {source}\nТекст: {text}"
        context_parts.append(part)

    return "\n\n".join(context_parts)


def generate_answer(question: str, context: str) -> str:
    """
    Пункты 7 и 8: Составляет промпт и получает ответ от LLM.
    """

    # Пункт 7: Строгий системный промпт
    system_prompt = """
    Ты полезный ассистент для работы со строительной базой знаний.
    Твоя задача: отвечать на вопрос пользователя ТОЛЬКО на основе предоставленного ниже КОНТЕКСТА.

    ПРАВИЛА:
    1. Если в контексте есть точный ответ — дай его кратко и понятно.
    2. Если в контексте НЕТ информации для ответа, честно напиши: 
       "В предоставленных документах нет информации для ответа на этот вопрос".
    3. ЗАПРЕЩЕНО выдумывать факты или использовать знания вне контекста.
    4. Не упоминай, что ты используешь контекст, просто отвечай на вопрос.
    """

    user_message = f"КОНТЕКСТ:\n{context}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}"

    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning:free",  # модель
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1  # Низкая температура для более точных ответов
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка при обращении к LLM: {str(e)}"


def run_rag_query(
        question: str,
        qdrant_client,
        collection_name: str,
        embedding_func,
        top_k: int = 3
) -> Dict[str, Any]:
    """
    Пункт 5: Полная цепочка RAG.
    Принимает вопрос, ищет чанки в Qdrant, формирует контекст и получает ответ.
    """

    # 1. Векторизуем вопрос
    query_vector = embedding_func(question)

    # 2. Поиск в Vector Store (Qdrant)
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    # Преобразуем результат в список словарей
    found_chunks = []
    for point in search_result.points:
        found_chunks.append({
            "text": point.payload.get("text", ""),
            "metadata": point.payload,
            "score": point.score
        })

    # 3. Формируем контекст
    context = build_context(found_chunks)

    # 4. Получаем ответ от LLM
    llm_answer = generate_answer(question, context)

    return {
        "question": question,
        "found_chunks": found_chunks,
        "context": context,
        "llm_answer": llm_answer
    }