"""
Guided BERTopic с предвычислением эмбеддингов.
Используются биграммы и триграммы.

- valid_orig — исходные реплики;
- valid_docs — соответствующие им лемматизированные реплики;
"""

# ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# ИМПОРТЫ
import random
import numpy as np
import torch
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer

# ПРОВЕРКА ИСХОДНЫХ ДАННЫХ 
# valid_orig и valid_docs должны быть созданы на предыдущем этапе.

if not valid_docs:
    raise ValueError("Список valid_docs пуст")

if len(valid_orig) != len(valid_docs):
    raise ValueError(
        "Количество исходных и лемматизированных реплик не совпадает. "
    )

if any(not isinstance(doc, str) or not doc.strip() for doc in valid_docs):
    raise ValueError(
        "В valid_docs обнаружены пустые тексты или значения, "
        "не являющиеся строками."
    )

if any(not isinstance(doc, str) or not doc.strip() for doc in valid_orig):
    raise ValueError(
        "В valid_orig обнаружены пустые тексты или значения, "
        "не являющиеся строками."
    )

print(f"Количество реплик для моделирования: {len(valid_docs)}")

# ФИКСАЦИЯ SEED
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ЛЕММАТИЗАЦИЯ СТОП-СЛОВ
nltk.download('stopwords', quiet=True)

morph = MorphAnalyzer()

CUSTOM_STOP = {
    'весь', 'тот', 'этот', 'такой', 'свой', 'наш', 'ваш',
    'ну', 'вот', 'так', 'типа',
    'очень', 'просто', 'уже', 'ещё', 'только', 'лишь', 'даже',
    'например', 'вообще', 'раз'
}

# Получаем стандартные русские стоп-слова NLTK
RAW_STOP_WORDS = set(stopwords.words('russian')).union(CUSTOM_STOP)


STOP_WORDS = {
    morph.parse(word)[0].normal_form
    for word in RAW_STOP_WORDS
}

# Сохраняем отрицание.
STOP_WORDS.discard('не')

# ФИЛЬТР КЛЮЧЕВЫХ СЛОВ
POST_FILTER = {
    'весь', 'тот', 'этот', 'время', 'человек', 'например',
    'врач врач', 'обычно не', 'муж муж'
}

def is_valid_ngram(ngram):
    """Отбрасываем биграммы вида 'слово не'."""
    parts = ngram.split()
    return not (len(parts) == 2 and parts[1] == 'не')


# SEED-ТЕМЫ
# Впишите свои списки ключевых слов и фраз.
seed_topics = [
    # ['здоровье', 'врач', 'лечение'],
    # ['спорт', 'бег', 'тренировка'],
]

# ПРЕДВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = embedding_model.encode(
    valid_docs,
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
)

# НАСТРОЙКА COUNT VECTORIZER 
vectorizer = CountVectorizer(
    stop_words=list(STOP_WORDS),
    ngram_range=(2, 3),
    min_df=2,
    max_features=2000,
    token_pattern=r'(?u)\b\w+\b'
)

# НАСТРОЙКА UMAP
umap_model = UMAP(
    n_neighbors=16,
    n_components=5,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
    unique=True
)

# НАСТРОЙКА HDBSCAN
hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    min_samples=2,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# СОЗДАНИЕ BERTopic
topic_model = BERTopic(
    language="multilingual",
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    seed_topic_list=seed_topics,
    embedding_model=embedding_model,
    calculate_probabilities=False,
    verbose=True
)

# ОБУЧЕНИЕ
print("Обучение Guided BERTopic...")

topics, probs = topic_model.fit_transform(
    valid_docs,
    embeddings=embeddings
)

# ПРОВЕРКА СЛОВАРЯ 
# Проверяем, что CountVectorizer сформировал непустой словарь.
vocabulary = topic_model.vectorizer_model.vocabulary_

if not vocabulary:
    raise ValueError(
        "CountVectorizer сформировал пустой словарь. "
    )

print(f"Количество признаков в словаре: {len(vocabulary)}")

# ВЫВОД ТЕМ
topic_info = topic_model.get_topic_info()

print("\nGuided topics (биграммы + триграммы):")

for idx, row in topic_info.iterrows():
    tid = row['Topic']

    if tid == -1:
        continue

    raw = topic_model.get_topic(tid)

    clean = [
        (w, s)
        for w, s in raw
        if w not in POST_FILTER
        and len(w) > 2
        and is_valid_ngram(w)
    ]

    if not clean:
        continue

    words = [w for w, _ in clean[:12]]

    print(f"\nTopic {tid} ({row['Count']} utterances):")
    print(' | '.join(words))

# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
results = pd.DataFrame({
    'original': valid_orig,
    'processed': valid_docs,
    'topic': topics
})

results.to_csv(
    'guided_results.csv',
    index=False,
    encoding='utf-8-sig'
)

topic_info.to_csv(
    'guided_topics.csv',
    index=False,
    encoding='utf-8-sig'
)

topic_model.save("guided_bertopic_model")
