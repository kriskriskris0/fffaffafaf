import streamlit as st
from rag_core import load_pdf, chunk_text, embed_chunks, create_index, retrieve
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Конфигурация
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

@st.cache_resource
def load_model():
    """Загружает модель Qwen (исправлено для новых версий transformers)"""
    with st.spinner("🔄 Загружаю Qwen..."):
        # Явно указываем use_fast=False, это часто помогает
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="cpu"  # явно CPU, чтобы избежать проблем с ускорением
        )
        return model, tokenizer

def generate_answer(question, context_chunks, model, tokenizer):
    if not context_chunks:
        return "Нет контекста."
    context = "\n\n".join(context_chunks[:2])
    messages = [
        {"role": "system", "content": "Ты — полезный ассистент. Отвечай кратко по контексту."},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# Интерфейс
st.set_page_config(page_title="📚 Qwen RAG", layout="wide")
st.title("📚 Qwen RAG (ручная установка)")

# Загрузка модели
try:
    model, tokenizer = load_model()
    st.sidebar.success("✅ Qwen загружена")
except Exception as e:
    st.sidebar.error(f"❌ Ошибка: {str(e)}")
    st.stop()

# Загрузка PDF
uploaded_file = st.file_uploader("Загрузите PDF", type="pdf")

if uploaded_file:
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
            answer = generate_answer(question, context, model, tokenizer)
        st.markdown("### 🤖 Ответ:")
        st.write(answer)