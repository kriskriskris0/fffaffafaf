import streamlit as st
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import numpy as np
import gc

# Добавляем путь к подмодулю визуального эмбеддера
sys.path.append(os.path.abspath("visual-language-ui-embedder"))
from config import UIEmbedderConfig
from main import UIEmbedderPipeline

from rag_core import extract_pdf_data, chunk_text, embed_chunks, create_index, retrieve, retrieve_image
from db import init_db, clear_db, insert_text_chunks, insert_image_chunks

# Конфигурация по умолчанию
DEFAULT_TEXT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # 2B модель для векторизации текста

def free_memory():
    """Очищает VRAM и RAM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(model_name):
    """Загружает модель (4-bit). Скачивает её, если нет в кэше."""
    with st.spinner(f"🔄 Загружаю модель {model_name}..."):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device.")
        
        if config.model_type == "qwen2_vl":
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map=device,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map=device,
            )
            
        model.eval()
        return model, tokenizer

def load_embedder_pipeline():
    """Загружает визуальный эмбеддер"""
    with st.spinner("🔄 Загружаю визуальный эмбеддер (Qwen2.5-VL)..."):
        config = UIEmbedderConfig.from_model_name("2B") # Можно указать другую 3B или 2B модель тут
        config.debug_decode_embeddings = False
        pipeline = UIEmbedderPipeline(config)
        return pipeline

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
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

# Интерфейс
st.set_page_config(page_title="📚 Qwen RAG + Images", layout="wide")
st.title("📚 Qwen RAG + Images")

try:
    init_db()
except Exception as e:
    st.error(f"Не удалось подключиться к базе данных PostgreSQL. Убедитесь, что она запущена: {e}")

# Настройки моделей
text_model_name = st.sidebar.text_input("Модель для текста:", value=DEFAULT_TEXT_MODEL)

# Загружаем сессионную модель для чата, если документ уже обработан или мы только запустили приложение
if "chat_model" not in st.session_state and "chat_tokenizer" not in st.session_state:
    try:
        model, tokenizer = load_model(text_model_name)
        st.session_state.chat_model = model
        st.session_state.chat_tokenizer = tokenizer
        st.sidebar.success(f"✅ Модель {text_model_name} загружена (режим чата)")
    except Exception as e:
        st.sidebar.error(f"❌ Ошибка загрузки текстовой модели: {str(e)}")
        st.stop()
        
# Настройки пропуска изображений
skip_first = st.sidebar.number_input("Пропустить N первых изображений:", min_value=0, value=0)
skip_last = st.sidebar.number_input("Пропустить M последних изображений:", min_value=0, value=0)

# Загрузка PDF
uploaded_file = st.file_uploader("Загрузите PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF загружен")

    if "pdf_processed" not in st.session_state:
        # ---- ЭТАП 1: Векторизация текста ----
        with st.spinner("Обрабатываю текст документа..."):
            clear_db()
            all_text, image_text_pairs = extract_pdf_data(
                "temp.pdf", 
                skip_first_images=skip_first, 
                skip_last_images=skip_last
            )
            
            # Векторизуем текст с помощью загруженной LLM
            text_chunks = chunk_text(all_text)
            text_embeddings = embed_chunks(text_chunks, st.session_state.chat_model, st.session_state.chat_tokenizer)
            
            # Сохраняем текстовые эмбеддинги в БД
            insert_text_chunks(text_chunks, text_embeddings)
            
        # ---- ОЧИСТКА LLM ДЛЯ ОСВОБОЖДЕНИЯ VRAM ПЕРЕД ВИЗУАЛЬНЫМ ЭМБЕДДЕРОМ ----
        with st.spinner("Выгружаю текстовую модель для освобождения VRAM..."):
            del st.session_state.chat_model
            del st.session_state.chat_tokenizer
            free_memory()

        # ---- ЭТАП 2: Визуальный Эмбеддер ----
        with st.spinner("Загружаю визуальный эмбеддер и векторизую изображения..."):
            vis_pipeline = load_embedder_pipeline()
            
            image_embeddings = []
            image_paths = []
            
            snippets_dir = "extracted_data/snippets"
            os.makedirs(snippets_dir, exist_ok=True)
            
            for idx, pair in enumerate(image_text_pairs):
                img = Image.open(pair["image_path"]).convert("RGB")
                context = pair["context_text"]
                
                try:
                    emb_dict = vis_pipeline.process(img, context)
                except IndexError:
                    continue
                
                original_width, original_height = img.size
                
                for bbox, emb_list in emb_dict.items():
                    emb_arr = np.array(emb_list).reshape(1, -1)
                    if emb_arr.shape[1] == text_embeddings.shape[1]:
                        # Вырезаем фрагмент из картинки
                        # Координаты bbox нормированы (от 0 до 1)
                        x1, y1, x2, y2 = bbox
                        left = int(x1 * original_width)
                        upper = int(y1 * original_height)
                        right = int(x2 * original_width)
                        lower = int(y2 * original_height)
                        
                        snippet = img.crop((left, upper, right, lower))
                        snippet_path = os.path.join(snippets_dir, f"snippet_{idx}_{left}_{upper}.png")
                        snippet.save(snippet_path)
                        
                        image_embeddings.append(emb_arr)
                        image_paths.append(snippet_path)
                    else:
                        st.warning(f"Пропуск изображения {idx+1}: размерность эмбеддинга {emb_arr.shape[1]} != {text_embeddings.shape[1]}")

            if image_embeddings:
                insert_image_chunks(image_paths, image_embeddings)
                
            st.session_state.pdf_processed = True
            
        # ---- ОЧИСТКА ЭМБЕДДЕРА И ВОЗВРАТ LLM ----
        with st.spinner("Выгружаю эмбеддер и возвращаю LLM для чата..."):
            del vis_pipeline
            free_memory()
            model, tokenizer = load_model(text_model_name)
            st.session_state.chat_model = model
            st.session_state.chat_tokenizer = tokenizer

        st.success(f"Готово! Обработано фрагментов текста: {len(text_chunks)}, визуальных компонентов: {len(image_embeddings)}")

    question = st.text_input("Введите вопрос:")
    if question:
        with st.spinner("Генерирую ответ..."):
            context = retrieve(
                question, 
                st.session_state.chat_model, 
                st.session_state.chat_tokenizer
            )
            
            best_snippet_path = retrieve_image(
                question,
                st.session_state.chat_model,
                st.session_state.chat_tokenizer
            )
            
            answer = generate_answer(question, context, st.session_state.chat_model, st.session_state.chat_tokenizer)
            
        st.markdown("### 🤖 Ответ:")
        st.write(answer)
        if best_snippet_path:
            st.markdown("### 🖼️ Наиболее подходящий фрагмент UI:")
            st.image(best_snippet_path)