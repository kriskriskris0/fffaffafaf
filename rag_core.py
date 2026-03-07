import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Загружаем модель эмбеддингов (один раз)
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def load_pdf(path):
    """Читает текст из PDF-файла."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=800, overlap=150):
    """Разбивает текст на перекрывающиеся куски."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    """Создаёт эмбеддинги для каждого куска."""
    return embed_model.encode(chunks)

def create_index(embeddings):
    """Для совместимости возвращаем эмбеддинги (индекс не нужен)."""
    return embeddings

def retrieve(question, embeddings, chunks, k=3):
    """Ищет k самых похожих кусков на вопрос (по косинусной близости)."""
    query_vec = embed_model.encode([question])
    similarities = cosine_similarity(query_vec, embeddings)[0]
    top_k_idx = np.argsort(similarities)[-k:][::-1]
    return [chunks[i] for i in top_k_idx]