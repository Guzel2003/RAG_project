# RAG Data Preparation Pipeline

Production-ready пайплайн подготовки данных для RAG-систем в сфере строительства и нормативной документации. Проект реализует полный цикл обработки документов: от сырых файлов до векторного индекса, готового для семантического поиска.

## 📋 Назначение проекта

Проект решает задачу подготовки корпоративной базы знаний для RAG (Retrieval-Augmented Generation) системы. На вход подаются строительные нормативы, ГОСТы, СП и другие документы в форматах PDF, DOCX, TXT, CSV, HTML. На выходе — векторный индекс в Qdrant, готовый для подключения LLM и генерации ответов на вопросы пользователей.

**Ключевые возможности:**

- Поддержка 5+ форматов документов (PDF, DOCX, TXT, CSV, HTML)
- Автоматическое определение кодировки (UTF-8, CP1251 для русских документов)
- Semantic-aware chunking с сохранением границ абзацев
- Дедупликация через MinHash LSH
- Векторизация через локальную multilingual модель
- Векторное хранилище Qdrant в local mode (без Docker)
- Полная трассируемость данных (lineage metadata)
- Строгая валидация на каждом этапе

### Что входит в репозиторий

В репозиторий включены:

- ✅ Полный исходный код всех 4 этапов
- ✅ Конфигурационные файлы
- ✅ Артефакты проверки (`data/vector_store/`)
- ✅ Инструкция по запуску

### Что НЕ входит в репозиторий

Следующие папки исключены через `.gitignore` (слишком большие для git):

- `data/raw/` — исходные PDF/DOCX документы
- `data/prepared/` — результат этапа 1
- `data/chunks/` — результат этапа 2
- `data/embeddings/` — результат этапа 3
- `data/qdrant_storage/` — бинарные данные Qdrant
  
🏗️ Структура проекта
RAG_project/
├── config/                     # Конфигурационные файлы
│   ├── default.yaml            # Этап 1: подготовка данных
│   ├── chunking.yaml           # Этап 2: чанкинг
│   ├── embeddings.yaml         # Этап 3: векторизация
│   └── vector_store.yaml       # Этап 4: индексация
├── data/                       # Данные (не входят в архив, кроме vector_store)
│   ├── raw/                    # Исходные документы
│   ├── prepared/               # Результат этапа 1
│   ├── chunks/                 # Результат этапа 2
│   ├── embeddings/             # Результат этапа 3
│   ├── vector_store/           # Артефакты проверки этапа 4 (JSON)
│   │    ├── manifest.json
│   │    ├── serch_result.json
│   │    ├── validation.json
│   └── qdrant_storage/         # Данные Qdrant
├── logs/                       # Логи выполнения
├── src/
│   └── rag_prep/               # Основной пакет
│       ├── __init__.py         # Инициализация пакета
│       ├── __main__.py         # Точка входа CLI
│       ├── utils.py            # Общие утилиты
│       │
│       ├── cli.py              # CLI этапа 1
│       ├── config.py           # Конфиги этапа 1
│       ├── models.py           # Модели этапа 1
│       ├── pipeline.py         # Пайплайн этапа 1
│       │
│       ├── cli_chunking.py     # CLI этапа 2
│       ├── config_chunking.py  # Конфиги этапа 2
│       ├── models_chunking.py  # Модели этапа 2
│       ├── pipeline_chunking.py# Пайплайн этапа 2
│       │
│       ├── cli_embeddings.py   # CLI этапа 3
│       ├── config_embeddings.py# Конфиги этапа 3
│       ├── models_embeddings.py# Модели этапа 3
│       ├── pipeline_embeddings.py # Пайплайн этапа 3
│       │
│       ├── cli_vector_store.py # CLI этапа 4
│       ├── config_vector_store.py # Конфиги этапа 4
│       ├── models_vector_store.py # Модели этапа 4
│       ├── pipeline_vector_store.py # Пайплайн этапа 4
│       │
│       ├── stages/             # Этапы подготовки (этап 1)
│       │   ├── __init__.py
│       │   ├── loading.py      # Загрузка файлов
│       │   ├── parsing.py      # Парсинг документов
│       │   ├── cleaning.py     # Очистка текста
│       │   ├── normalization.py# Нормализация
│       │   ├── deduplication.py# Дедупликация
│       │   ├── structuring.py  # Структурирование
│       │   └── exporting.py    # Экспорт
│       │
│       ├── chunking_stages/    # Этапы чанкинга (этап 2)
│       │   ├── __init__.py
│       │   ├── loading.py      # Загрузка документов
│       │   ├── splitting.py    # Разбиение на чанки
│       │   ├── validation.py   # Валидация чанков
│       │   └── exporting.py    # Экспорт чанков
│       │
│       ├── embedding_stages/   # Этапы embeddings (этап 3)
│       │   ├── __init__.py
│       │   ├── loading.py      # Загрузка чанков
│       │   ├── embedding.py    # Расчёт векторов
│       │   ├── validation.py   # Валидация векторов
│       │   ├── metrics.py      # Сбор метрик
│       │   └── exporting.py    # Экспорт embeddings
│       │
│       └── vector_store_stages/# Этапы vector store (этап 4)
│           ├── __init__.py
│           ├── loading.py      # Загрузка и валидация embeddings
│           ├── indexing.py     # Создание индекса и загрузка
│           ├── validation.py   # Валидация состояния БД
│           ├── searching.py    # Тестовый поиск
│           └── exporting.py    # Экспорт артефактов
│
├── .gitignore
├── pyproject.toml              # Описание пакета
├── requirements.txt            # Зависимости
└── README.md                   # Этот файл

### Как воспроизвести проект полностью

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/Guzel2003/RAG_project.git
   cd RAG_project
2. Добавьте свои документы в data/raw/

 🚀 Быстрый старт
Требования
Python 3.12+
Windows 10/11 или Linux/macOS
~2 ГБ свободного места для моделей и данных

Установка
  # 1. Установите зависимости
  pip install -r requirements.txt
  
  # 2. Установите пакет в режиме разработки
  pip install -e .


📊 Этапы пайплайна
Этап 1: Подготовка данных
Назначение: Загрузка, парсинг, очистка, нормализация и дедупликация исходных документов.
Что делает:
1. Находит все поддерживаемые файлы в data/raw/
2. Парсит PDF через PyMuPDF, DOCX через python-docx, TXT с автоопределением кодировки
3. Очищает текст от boilerplate (куки, навигация, копирайты)
4. Нормализует Unicode и собирает статистику предложений
5. Удаляет точные и близкие дубликаты через MinHash LSH
6. Структурирует элементы в документы с metadata
7. Экспортирует в JSON/JSONL + manifest
   
Запуск:
    python -m rag_prep --config config/default.yaml


Результат:
data/prepared/documents.json — массив документов
data/prepared/documents.jsonl — построчный формат
data/prepared/manifest.json — метаданные запуска
Этап 2: Chunking (чанкинг)
Назначение: Разбиение подготовленных документов на чанки, пригодные для embeddings и retrieval.
Что делает:
1. Загружает документы из data/prepared/documents.jsonl
2. Разбивает текст на семантические блоки
3. Применяет sentence-aware splitting через LlamaIndex
4. Сохраняет overlap между чанками для контекста
5. Валидирует чанки (пустые, слишком маленькие/большие)
6. Собирает метрики качества
7. Экспортирует в JSON/JSONL + manifest
Стратегия: sentence — разбиение по предложениям с сохранением границ абзацев. Не режет текст посередине слова или предложения.
Параметры:
1. chunk_size: 256 токенов
2. chunk_overlap: 50 токенов
3. tokenizer_model: text-embedding-3-small

Запуск:
    python -m rag_prep chunk --config config/chunking.yaml


Результат:
data/chunks/chunks.json — массив чанков
data/chunks/chunks.jsonl — построчный формат
data/chunks/manifest.json — метаданные запуска

Этап 3: Embeddings (векторизация)
Назначение: Расчёт векторных представлений для каждого чанка.
Что делает:
1. Загружает чанки из data/chunks/chunks.jsonl
2. Рассчитывает embeddings через локальную модель FastEmbed
3. Нормализует векторы (L2 normalization)
4. Валидирует векторы (NaN, Infinity, размерность)
5. Собирает метрики (средняя норма, скорость обработки)
6. Экспортирует в JSON/JSONL + manifest
Модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Запуск:
    python -m rag_prep embed --config config/embeddings.yaml
    
Результат:
data/embeddings/embeddings.json — массив embeddings
data/embeddings/embeddings.jsonl — построчный формат
data/embeddings/manifest.json — метаданные запуска

Этап 4: Vector Store (индексация)
Назначение: Создание векторного индекса, загрузка embeddings, строгая валидация и тестовый поиск.
Vector Store: Qdrant в local mode
Хранение: файловая система (data/qdrant_storage/)
Без Docker, без сервера
Метрика расстояния: Cosine similarity
Payload: текст + полная metadata чанка
Что делает:
1. Загружает embeddings из data/embeddings/embeddings.jsonl
2. Валидирует данные (структура, типы, размерность из конфига, NaN/Infinity)
3. Создает коллекцию construction_docs в Qdrant (с проверкой параметров существующей коллекции)
4. Загружает векторы батчами по 100 штук
5. Проверяет количество точек, размерность и структуру payload в выборке
6. Выполняет тестовые поисковые запросы с фиксированным seed
7. Сохраняет детальные артефакты проверки

Запуск:
    python -m rag_prep vector-store --config config/vector_store.yaml

Результат (Артефакты):
data/vector_store/validation.json — результаты валидации индекса (размерность, метрика, наличие обязательных полей)
data/vector_store/search_results.json — результаты тестового поиска с полной metadata найденных документов
data/vector_store/manifest.json — снимок конфигурации, статистика запуска и статусы валидации
data/qdrant_storage/ — данные Qdrant (бинарные файлы)
      
