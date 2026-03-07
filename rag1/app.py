import streamlit as st
from rag_core import load_pdf, chunk_text, embed_chunks, create_index, retrieve
import torch  # может не понадобиться, но оставим для совместимости

# Для AirLLM
try:
    from airllm import AirLLMLlama2

    AIRLLM_AVAILABLE = True
except ImportError:
    AIRLLM_AVAILABLE = False
    st.error("AirLLM не установлен. Установите: pip install airllm")
    st.stop()

# ------------------- НАСТРОЙКИ -------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # или другая поддерживаемая модель
COMPRESSION = '4bit'  # или '8bit', None для полной точности


# ------------------------------------------------

@st.cache_resource
def load_model():
    """Загружает модель через AirLLM (кэшируется)."""
    with st.spinner(
            f"🔄 Загружаем Qwen через AirLLM (компрессия: {COMPRESSION})... Это может занять несколько минут при первом запуске."):
        model = AirLLMLlama2(
            MODEL_NAME,
            compression=COMPRESSION,  # квантизация
            device_map="auto",  # auto использует CPU+диск при необходимости
            max_seq_len=2048  # максимальная длина контекста
        )
        return model


def generate_answer(question, context_chunks, model):
    """Генерирует ответ, используя AirLLM."""
    if not context_chunks:
        return "Нет контекста."

    context = "\n\n".join(context_chunks[:2])  # ограничим двумя кусками

    # Формируем prompt (AirLLM ожидает просто текст)
    prompt = f"""Ты — полезный ассистент. Отвечай кратко и только на основе контекста.
Контекст:
{context}
Вопрос: {question}
Ответ:"""

    # Генерация через AirLLM (метод generate возвращает список токенов или текст)
    # Настраиваем параметры генерации
    output = model.generate(
        prompt,
        max_new_tokens=200,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        repetition_penalty=1.0
    )

    # AirLLM может вернуть список токенов или строку в зависимости от версии
    if isinstance(output, list):
        # если список токенов, нужно декодировать
        # но обычно есть свой токенизатор, проще использовать модель с токенизатором
        # В текущей реализации AirLLM возвращает строку
        return output[0] if output else ""
    else:
        return output.strip()


# ------------------- Интерфейс Streamlit -------------------
st.set_page_config(page_title="📚 Qwen RAG (AirLLM)", layout="wide")
st.title("📚 Qwen RAG — на AirLLM (экономия памяти)")

# Загрузка модели
if not AIRLLM_AVAILABLE:
    st.stop()

model = load_model()
st.sidebar.success("✅ Модель загружена через AirLLM")

# Загрузка PDF (точно такая же, как в предыдущих версиях)
uploaded_file = st.file_uploader("Загрузите PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF загружен")

    if "embeddings" not in st.session_state:
        with st.spinner("Обрабатываю документ..."):
            text = load_pdf("temp.pdf")
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)
            st.session_state.embeddings = embeddings
            st.session_state.chunks = chunks
        st.success(f"Готово! {len(chunks)} кусков")

    question = st.text_input("Введите вопрос:")

    if question:
        with st.spinner("Генерирую ответ..."):
            context = retrieve(question, st.session_state.embeddings, st.session_state.chunks)
            answer = generate_answer(question, context, model)
        st.markdown("### 🤖 Ответ:")
        st.write(answer)