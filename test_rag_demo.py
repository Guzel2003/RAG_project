from qdrant_client import QdrantClient
from rag_prep.rag_chain import run_rag_query
from fastembed import TextEmbedding # Используем ту же модель, что и при индексации

# 1. Подключаемся к существующей базе
client = QdrantClient(path="data/qdrant_storage")

# 2. Инициализируем модель эмбеддингов (та же, что была при создании индекса)
embedding_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_embedding(text: str):
    """Обертка для получения вектора из строки"""
    return list(embedding_model.embed(text))[0]

# 3. Задаем тестовый вопрос
question = "Какие требования к доставке документов?"

print(f"Ищу ответ на вопрос: {question}\n")

# 4. Запускаем RAG
result = run_rag_query(
    question=question,
    qdrant_client=client,
    collection_name="construction_docs",
    embedding_func=get_embedding,
    top_k=3
)

# 5. Выводим результаты
print("--- НАЙДЕННЫЕ ЧАНКИ ---")
for i, chunk in enumerate(result['found_chunks'], 1):
    print(f"{i}. [Score: {chunk['score']:.4f}] {chunk['text'][:100]}...")

print("\n--- ОТВЕТ LLM ---")
print(result['llm_answer'])