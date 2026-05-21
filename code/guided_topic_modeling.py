"""
Guided BERTopic с предвычислением эмбеддингов (биграммы + триграммы).
Ожидается переменная valid_docs (список лемматизированных текстов).
"""

import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Фиксация seed
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------------------- СТОП-СЛОВА (краткий вариант) ----------------------
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

CUSTOM_STOP = {
    'весь', 'тот', 'этот', 'такой', 'который', 'свой', 'наш', 'ваш',
    'мочь', 'делать', 'говорить', 'знать', 'ну', 'вот', 'так', 'типа',
    'очень', 'просто', 'уже', 'ещё', 'только', 'лишь', 'даже',
    'например', 'вообще', 'человек', 'люди', 'время', 'место', 'раз'
}
STOP_WORDS = set(stopwords.words('russian')).union(CUSTOM_STOP)
STOP_WORDS.discard('не')   # сохраняем отрицания

# Фильтр мусорных n-грамм (можно дополнить)
POST_FILTER = {
    'весь', 'тот', 'этот', 'время', 'человек', 'например',
    'врач врач', 'обычно не', 'муж муж'
}

def is_valid_ngram(ngram):
    """Отбрасываем биграммы вида 'слово не'."""
    parts = ngram.split()
    return not (len(parts) == 2 and parts[1] == 'не')

# ---------------------- ЗАГОТОВКА ДЛЯ SEED-ТЕМ ----------------------
# ВПИШИТЕ СВОИ СПИСКИ КЛЮЧЕВЫХ СЛОВ (униграммы/фразы)
seed_topics = [
    # Пример: ['здоровье', 'врач', 'лечение'],
    #          ['спорт', 'бег', 'тренировка'],
]

if not seed_topics:
    raise ValueError("Заполните seed_topics своими ключевыми словами.")

# ---------------------- ПРЕДВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ ----------------------
print("Вычисление эмбеддингов...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = embedding_model.encode(valid_docs, show_progress_bar=True, batch_size=32)

# ---------------------- НАСТРОЙКА МОДЕЛИ ----------------------
vectorizer = CountVectorizer(
    stop_words=list(STOP_WORDS),
    ngram_range=(2, 3),          # только биграммы и триграммы
    min_df=2,
    max_features=2000,
    token_pattern=r'(?u)\b\w+\b'
)

umap_model = UMAP(n_neighbors=16, n_components=5, min_dist=0.1,
                  metric='cosine', random_state=42, unique=True)
hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=2,
                        metric='euclidean', cluster_selection_method='eom',
                        prediction_data=True)

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

# ---------------------- ОБУЧЕНИЕ ----------------------
topics, probs = topic_model.fit_transform(valid_docs, embeddings=embeddings)

# ---------------------- ВЫВОД ТЕМ (с фильтрацией) ----------------------
topic_info = topic_model.get_topic_info()
print("\nGuided topics (биграммы+триграммы):")
for idx, row in topic_info.iterrows():
    tid = row['Topic']
    if tid == -1:
        continue
    raw = topic_model.get_topic(tid)
    clean = [(w, s) for w, s in raw
             if w not in POST_FILTER and len(w) > 2 and is_valid_ngram(w)]
    if not clean:
        continue
    words = [w for w, _ in clean[:12]]
    print(f"\nTopic {tid} ({row['Count']} utterances):")
    print(' | '.join(words))

# ---------------------- СОХРАНЕНИЕ ----------------------
results = pd.DataFrame({
    'original': valid_orig,        # должна быть определена
    'processed': valid_docs,
    'topic': topics
})
results.to_csv('guided_results.csv', index=False, encoding='utf-8-sig')
topic_info.to_csv('guided_topics.csv', index=False, encoding='utf-8-sig')
topic_model.save("guided_bertopic_model")
print("\nМодель и результаты сохранены: guided_bertopic_model, guided_results.csv")
