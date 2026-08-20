from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_index(texts):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.9)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def score_query(vectorizer, matrix, query_text):
    query_vec = vectorizer.transform([query_text])
    return cosine_similarity(query_vec, matrix)[0]
