"""
Базовое тематическое моделирование (BERTopic)
реплик информанта с сохранением отрицаний.
Используются биграммы и триграммы.
Код предназначен для Google Colab.
"""

import re
import random
import numpy as np
import torch
import os
import glob
import pandas as pd
import pymorphy3
import nltk

from docx import Document
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

# Переменные окружения
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Фиксация seed для воспроизводимости
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Загрузка стоп-слов
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

morph = pymorphy3.MorphAnalyzer()

# Пользовательские стоп-слова (леммы)
CUSTOM_STOP_LEMMAS = {
    'весь', 'тот', 'этот', 'такой', 'который', 'свой', 'наш', 'ваш', 'мой', 'твой',
    'ну', 'вот', 'так', 'типа', 'вроде', 'значить', 'короче', 'наверное', 'возможно',
    'очень', 'просто', 'уже', 'ещё', 'только', 'лишь', 'даже', 'же', 'бы', 'ли',
    'там', 'тут', 'здесь', 'сейчас', 'потом', 'тогда', 'где', 'куда', 'откуда',
    'и', 'в', 'на', 'с', 'к', 'по', 'о', 'об', 'из', 'за', 'под', 'над', 'для',
    'он', 'она', 'оно', 'они', 'мы', 'вы', 'я', 'ты', 'пример', 'например',
}

STOP_WORDS = set(stopwords.words('russian')).union(CUSTOM_STOP_LEMMAS)

# Сохраняем отрицание
STOP_WORDS.discard('не')


def read_docx(filepath):
    """Читает все параграфы из .docx файла."""
    doc = Document(filepath)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def extract_informant_utterances(lines):
    """Извлекает реплики, начинающиеся с 'И:'."""
    utterances = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if re.match(r'^\s*И:', line):
            text = re.sub(r'^\s*И:\s*', '', line).strip()

            if text:
                utterances.append(text)

    return utterances


def lemmatize_with_negation(text):
    """Лемматизация с сохранением токена 'не'."""
    tokens = re.findall(r'[а-яё]+', text.lower())
    lemmas = []

    for token in tokens:
        if len(token) >= 2:
            lemma = morph.parse(token)[0].normal_form

            if lemma not in STOP_WORDS or lemma == 'не':
                lemmas.append(lemma)

    return ' '.join(lemmas)


# Поиск всех файлов с "Транскрипт" в названии
docx_files = glob.glob('/content/*.docx')

transcript_files = [
    f for f in docx_files
    if 'Транскрипт' in os.path.basename(f)
]

if not transcript_files:
    raise FileNotFoundError(
        "Не найдено файлов с 'Транскрипт' в /content/"
    )

# Удаление дубликатов по имени
seen = set()
unique_files = []

for f in transcript_files:
    name = os.path.basename(f)

    if name not in seen:
        seen.add(name)
        unique_files.append(f)

transcript_files = unique_files

print(f"Найдено файлов: {len(transcript_files)}")

# Извлечение всех реплик информанта
all_utterances = []

for fp in sorted(transcript_files):
    lines = read_docx(fp)
    utterances = extract_informant_utterances(lines)

    all_utterances.extend(utterances)

    print(
        f"{os.path.basename(fp)}: "
        f"{len(utterances)} реплик"
    )

print(f"Всего реплик: {len(all_utterances)}")

if not all_utterances:
    raise ValueError("Не удалось извлечь реплики информантов.")

# ЛЕММАТИЗАЦИЯ
processed = [
    lemmatize_with_negation(text)
    for text in all_utterances
]

# Фильтр: длина исходной реплики >= 12 символов
# и наличие хотя бы одного лемматизированного токена
valid_idx = [
    i for i, doc in enumerate(processed)
    if len(doc.split()) >= 1
    and len(all_utterances[i].strip()) >= 12
]

valid_docs = [processed[i] for i in valid_idx]
valid_orig = [all_utterances[i] for i in valid_idx]

print(f"После фильтрации: {len(valid_docs)} реплик")

# Проверка корпуса
if not valid_docs:
    raise ValueError("После фильтрации не осталось реплик.")

if len(valid_orig) != len(valid_docs):
    raise ValueError("Нарушена синхронизация исходных и обработанных реплик.")

if any(not isinstance(doc, str) or not doc.strip() for doc in valid_docs):
    raise ValueError("В корпусе обнаружены пустые или некорректные тексты.")

# ПАРАМЕТРЫ МОДЕЛИ
NGRAM_RANGE = (2, 3)

# Предварительная проверка словаря CountVectorizer
vectorizer = CountVectorizer(
    stop_words=list(STOP_WORDS),
    ngram_range=NGRAM_RANGE,
    min_df=2,
    max_features=2000,
    token_pattern=r'(?u)\b\w+\b'
)

try:
    vectorizer.fit(valid_docs)
except ValueError as e:
    raise ValueError(
        "Не удалось сформировать словарь CountVectorizer. "
        "Проверьте корпус, стоп-слова и min_df."
    ) from e

if not vectorizer.vocabulary_:
    raise ValueError("CountVectorizer сформировал пустой словарь.")

print(
    f"Количество признаков в словаре: "
    f"{len(vectorizer.vocabulary_)}"
)

# Расчёт эмбеддингов MiniLM
print("\nРасчёт эмбеддингов MiniLM...")

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = embedding_model.encode(
    valid_docs,
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
)

# Настройка UMAP
umap_model = UMAP(
    n_neighbors=16,
    n_components=5,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)

# Настройка HDBSCAN
hdbscan_model = HDBSCAN(
    min_cluster_size=12,
    min_samples=2,
    metric='euclidean',
    cluster_selection_method='eom'
)

# Создание BERTopic
topic_model = BERTopic(
    language="multilingual",
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    embedding_model=embedding_model,
    calculate_probabilities=True,
    verbose=True
)

# ОБУЧЕНИЕ

topics, probs = topic_model.fit_transform(
    valid_docs,
    embeddings=embeddings
)

# Пост-фильтрация ключевых слов
POST_FILTER = {
    'весь', 'тот', 'этот', 'такой', 'который', 'свой', 'наш', 'ваш',
    'просто', 'очень', 'уже', 'ещё', 'только', 'лишь', 'даже',
    'например', 'вообще', 'вопрос', 'дело', 'время', 'место', 'раз',
    'врач врач', 'обычно не', 'муж муж'
}

# Вывод информации о темах
topic_info = topic_model.get_topic_info()

print("\nБазовые темы (биграммы + триграммы):")

for idx, row in topic_info.iterrows():
    tid = row['Topic']

    if tid == -1:
        continue

    raw_words = topic_model.get_topic(tid)

    clean_words = [
        (w, s)
        for w, s in raw_words
        if w not in POST_FILTER and len(w) > 1
    ]

    if not clean_words:
        continue

    count = row['Count']
    top_words = [w for w, _ in clean_words[:12]]

    print(f"\nТема {tid} ({count} реплик):")
    print(' | '.join(top_words))

# Сохранение результатов
results = pd.DataFrame({
    'original': valid_orig,
    'processed': valid_docs,
    'topic': topics
})

results.to_csv(
    'results_bigram_trigram.csv',
    index=False,
    encoding='utf-8-sig'
)

topic_info.to_csv(
    'topics_bigram_trigram.csv',
    index=False,
    encoding='utf-8-sig'
)

# Сохранение эмбеддингов 
np.save('embeddings_bigram_trigram.npy', embeddings)

# Сохранение модели
topic_model.save("bertopic_model_bigram_trigram")

