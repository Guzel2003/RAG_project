\# RAG Data Preparation Pipeline



Production-ready пайплайн подготовки данных для RAG-систем в сфере строительства и нормативной документации. Проект реализует полный цикл обработки документов: от сырых файлов до векторного индекса, готового для семантического поиска.



\## 📋 Назначение проекта



Проект решает задачу подготовки корпоративной базы знаний для RAG (Retrieval-Augmented Generation) системы. На вход подаются строительные нормативы, ГОСТы, СП и другие документы в форматах PDF, DOCX, TXT, CSV, HTML. На выходе — векторный индекс в Qdrant, готовый для подключения LLM и генерации ответов на вопросы пользователей.



\*\*Ключевые возможности:\*\*

\- Поддержка 5+ форматов документов (PDF, DOCX, TXT, CSV, HTML)

\- Автоматическое определение кодировки (UTF-8, CP1251 для русских документов)

\- Semantic-aware chunking с сохранением границ абзацев

\- Дедупликация через MinHash LSH

\- Векторизация через локальную multilingual модель

\- Векторное хранилище Qdrant в local mode (без Docker)

\- Полная трассируемость данных (lineage metadata)

\- Валидация на каждом этапе





\### Что входит в репозиторий



В репозиторий включены:

\- ✅ Полный исходный код всех 4 этапов

\- ✅ Конфигурационные файлы

\- ✅ Артефакты проверки (`data/vector\_store/`)

\- ✅ Инструкция по запуску



\### Что НЕ входит в репозиторий



Следующие папки исключены через `.gitignore` (слишком большие для git):

\- `data/raw/` — исходные PDF/DOCX документы

\- `data/prepared/` — результат этапа 1

\- `data/chunks/` — результат этапа 2

\- `data/embeddings/` — результат этапа 3

\- `data/qdrant\_storage/` — бинарные данные Qdrant



\### Как воспроизвести проект полностью



1\. Клонируйте репозиторий:

&#x20;  ```bash

&#x20;  git clone https://github.com/Guzel2003/RAG\_project.git

&#x20;  cd RAG\_project



2\. Добавьте свои документы в data/raw/ 



\## 🏗️ Структура проекта

RAG\_project/

├── config/ # Конфигурационные файлы

│ ├── default.yaml # Этап 1: подготовка данных

│ ├── chunking.yaml # Этап 2: чанкинг

│ ├── embeddings.yaml # Этап 3: векторизация

│ └── vector\_store.yaml # Этап 4: индексация

├── data/ # Данные (не входят в архив)

│ ├── raw/ # Исходные документы

│ ├── prepared/ # Результат этапа 1

│ ├── chunks/ # Результат этапа 2

│ ├── embeddings/ # Результат этапа 3

│ ├── vector\_store/ # Артефакты проверки этапа 4

│ └── qdrant\_storage/ # Данные Qdrant

├── logs/ # Логи выполнения

├── src/

│ └── rag\_prep/ # Основной пакет

│ ├── init.py

│ ├── main.py # Точка входа

│ ├── utils.py # Общие утилиты

│ │

│ ├── cli.py # CLI этапа 1

│ ├── config.py # Конфиги этапа 1

│ ├── models.py # Модели этапа 1

│ ├── pipeline.py # Пайплайн этапа 1

│ │

│ ├── cli\_chunking.py # CLI этапа 2

│ ├── config\_chunking.py # Конфиги этапа 2

│ ├── models\_chunking.py # Модели этапа 2

│ ├── pipeline\_chunking.py # Пайплайн этапа 2

│ │

│ ├── cli\_embeddings.py # CLI этапа 3

│ ├── config\_embeddings.py # Конфиги этапа 3

│ ├── models\_embeddings.py # Модели этапа 3

│ ├── pipeline\_embeddings.py # Пайплайн этапа 3

│ │

│ ├── cli\_vector\_store.py # CLI этапа 4

│ ├── config\_vector\_store.py # Конфиги этапа 4

│ ├── models\_vector\_store.py # Модели этапа 4

│ ├── pipeline\_vector\_store.py # Пайплайн этапа 4

│ │

│ ├── stages/ # Этапы подготовки (этап 1)

│ │ ├── loading.py # Загрузка файлов

│ │ ├── parsing.py # Парсинг документов

│ │ ├── cleaning.py # Очистка текста

│ │ ├── normalization.py # Нормализация

│ │ ├── deduplication.py # Дедупликация

│ │ ├── structuring.py # Структурирование

│ │ └── exporting.py # Экспорт

│ │

│ ├── chunking\_stages/ # Этапы чанкинга (этап 2)

│ │ ├── loading.py # Загрузка документов

│ │ ├── splitting.py # Разбиение на чанки

│ │ ├── validation.py # Валидация чанков

│ │ └── exporting.py # Экспорт чанков

│ │

│ ├── embedding\_stages/ # Этапы embeddings (этап 3)

│ │ ├── loading.py # Загрузка чанков

│ │ ├── embedding.py # Расчёт векторов

│ │ ├── validation.py # Валидация векторов

│ │ ├── metrics.py # Сбор метрик

│ │ └── exporting.py # Экспорт embeddings

│ │

│ └── vector\_store\_stages/ # Этапы vector store (этап 4)

│ ├── loading.py # Загрузка embeddings

│ ├── indexing.py # Создание индекса

│ ├── validation.py # Валидация индекса

│ ├── searching.py # Тестовый поиск

│ └── exporting.py # Экспорт артефактов

│

├── pyproject.toml # Описание пакета

├── requirements.txt # Зависимости

└── README.md # Этот файл



\## 🚀 Быстрый старт



\### Требования



\- Python 3.12+

\- Windows 10/11 или Linux/macOS

\- \~2 ГБ свободного места для моделей и данных



\### Установка



\# 1. Установите зависимости

pip install -r requirements.txt



\# 2. Установите пакет в режиме разработки

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

python -m rag\_prep --config config/default.yaml



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

chunk\_size: 256 токенов

chunk\_overlap: 50 токенов 

tokenizer\_model: text-embedding-3-small



Запуск:

python -m rag\_prep chunk --config config/chunking.yaml



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
7. Модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2



Запуск:

python -m rag\_prep embed --config config/embeddings.yaml



Результат:

data/embeddings/embeddings.json — массив embeddings

data/embeddings/embeddings.jsonl — построчный формат

data/embeddings/manifest.json — метаданные запуска



Этап 4: Vector Store (индексация)

Назначение: Создание векторного индекса, загрузка embeddings, валидация и тестовый поиск.



Vector Store: Qdrant в local mode

1. Хранение: файловая система (data/qdrant\_storage/)
2. Без Docker, без сервера
3. Метрика расстояния: Cosine similarity
4. Payload: текст + metadata чанка



Что делает:

1. Загружает embeddings из data/embeddings/embeddings.jsonl
2. Валидирует данные (структура, размерность, NaN/Infinity)
3. Создаёт коллекцию construction\_docs в Qdrant
4. Загружает векторы батчами по 100 штук
5. Проверяет количество точек и структуру payload
6. Выполняет тестовые поисковые запросы
7. Сохраняет артефакты проверки



Как создать индекс:

Индекс создаётся автоматически при запуске пайплайна. Параметр recreate\_collection: true в конфиге гарантирует, что при каждом запуске старая коллекция удаляется и создаётся заново.



Как загрузить embeddings:

Загрузка происходит автоматически после создания коллекции. Векторы загружаются батчами для оптимизации производительности.



Как выполнить тестовый поиск:

Пайплайн автоматически берёт случайные чанки из входных данных и ищет их в индексе. Первый результат должен быть тем же самым чанком с score ≈ 1.0.



Запуск:

python -m rag\_prep vector-store --config config/vector\_store.yaml



Результат:

data/vector\_store/validation.json — результаты валидации индекса

data/vector\_store/search\_results.json — результаты тестового поиска

data/vector\_store/manifest.json — метаданные запуска

data/qdrant\_storage/ — данные Qdrant (бинарные файлы)



