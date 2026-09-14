"""
Простое тестирование RAG-пайплайна на 5 вопросах.
Без автоматического определения метаданных.
Запуск: python test_rag_questions.py
"""

from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from rag_prep.rag_chain import run_rag_query
import json
import os

os.makedirs("results", exist_ok=True)

client = QdrantClient(path="data/qdrant_storage")
embedding_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_embedding(text: str):
    return list(embedding_model.embed(text))[0]

# === ТЕСТОВЫЕ ВОПРОСЫ ===
questions = [
    {
        "question": "Какие документы прилагаются к заявлению о выдаче разрешения на ввод объекта в эксплуатацию?",
        "type": "has_answer"
    },
    {
        "question": "Что должно быть указано на информационном щите при въезде на строительную площадку?",
        "type": "has_answer"
    },
    {
        "question": "Какие графические параметры должны содержать исполнительные геодезические схемы?",
        "type": "has_answer"
    },
    {
        "question": "В какие сроки лицо, осуществляющее строительство, должно оградить строительную площадку?",
        "type": "specific_section"
    },
    {
        "question": "Как оформить кредит на строительство частного дома?",
        "type": "no_answer"
    }
]

results = []
for q in questions:
    print(f"\n{'='*60}")
    print(f"ВОПРОС: {q['question']}")

    # Запуск RAG-цепочки
    result = run_rag_query(
        question=q["question"],
        qdrant_client=client,
        collection_name="construction_docs",
        embedding_func=get_embedding,
        top_k=3
    )

    print(f"\nНАЙДЕННЫЕ ЧАНКИ:")
    for i, chunk in enumerate(result['found_chunks'], 1):
        score = chunk.get('score', 0)
        text_preview = chunk['text'][:200]
        print(f"  {i}. [{score:.4f}] {text_preview}...")

    print(f"\nОТВЕТ LLM:\n{result['llm_answer']}")

    # Простая оценка
    if q['type'] == 'no_answer':
        is_correct = ("нет информации" in result['llm_answer'].lower() or
                      "не найдено" in result['llm_answer'].lower())
        evaluation = "Корректно" if is_correct else "Некорректно"
    else:
        # Просто проверяем, что ответ не пустой и похож на правду
        evaluation = "Корректно" if len(result['llm_answer']) > 20 else "Частично"

    print(f"ОЦЕНКА: {evaluation}")

    # Сохранение для отчета
    results.append({
        "question": q["question"],
        "type": q["type"],
        "found_chunks_text": [c['text'][:300] for c in result['found_chunks']],
        "llm_answer": result['llm_answer'],
        "evaluation": evaluation
    })

# Сохраняем итоговый JSON
output_path = "results/rag_test_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Результаты сохранены в {output_path}")