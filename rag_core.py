import numpy as np
import torch
import fitz  # PyMuPDF
import os
import math
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

def extract_pdf_data(path, skip_first_images=0, skip_last_images=0, out_dir="extracted_data"):
    """
    Извлекает изображения и ближайший к ним текст из PDF.
    Сохраняет в out_dir/images и out_dir/texts.
    Сначала собирает все изображения, затем отбрасывает skip_first_images и skip_last_images.
    Возвращает (all_text_chunks, image_text_pairs)
    """
    img_dir = os.path.join(out_dir, "images")
    txt_dir = os.path.join(out_dir, "texts")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    doc = fitz.open(path)
    all_text = ""
    
    # Собираем данные обо всех изображениях: (page_num, xref, bbox, pixmap_data, closest_text)
    images_info = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        all_text += page.get_text() + "\n"
        
        # Получаем текстовые блоки: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0] # block_type == 0 is text
        
        # Получаем изображения на странице
        image_list = page.get_images(full=True)
        
        for img_info in image_list:
            xref = img_info[0]
            # Чтобы найти координаты изображения, нужно получить bbox
            # fitz.Page.get_image_rects возвращает список rects для данного xref
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            
            img_rect = rects[0] # берём первый (обычно один)
            img_cx = (img_rect.x0 + img_rect.x1) / 2
            img_cy = (img_rect.y0 + img_rect.y1) / 2
            
            closest_text = ""
            min_dist = float('inf')
            
            # Ищем ближайший текстовый блок
            for tb in text_blocks:
                tb_cx = (tb[0] + tb[2]) / 2
                tb_cy = (tb[1] + tb[3]) / 2
                
                # Эвклидово расстояние между центрами
                dist = math.hypot(img_cx - tb_cx, img_cy - tb_cy)
                if dist < min_dist:
                    min_dist = dist
                    closest_text = tb[4]
            
            # Извлекаем само изображение
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            images_info.append({
                'page': page_num,
                'xref': xref,
                'ext': image_ext,
                'bytes': image_bytes,
                'text': closest_text.strip()
            })
            
    # Применяем фильтр пропусков (skip_first_images, skip_last_images)
    total_imgs = len(images_info)
    start_idx = skip_first_images
    end_idx = total_imgs - skip_last_images
    
    if start_idx >= end_idx or end_idx <= 0:
        filtered_images = []
    else:
        filtered_images = images_info[start_idx:end_idx]
        
    image_text_pairs = []
    
    for i, img_data in enumerate(filtered_images):
        img_filename = f"image_{i}.{img_data['ext']}"
        txt_filename = f"image_{i}.txt"
        
        img_path = os.path.join(img_dir, img_filename)
        txt_path = os.path.join(txt_dir, txt_filename)
        
        with open(img_path, "wb") as f:
            f.write(img_data['bytes'])
            
        with open(txt_path, "w", encoding='utf-8') as f:
            f.write(img_data['text'])
            
        image_text_pairs.append({
            "image_path": img_path,
            "text_path": txt_path,
            "context_text": img_data['text']
        })

    return all_text, image_text_pairs



def chunk_text(text, chunk_size=800, overlap=150):
    """Разбивает текст на перекрывающиеся куски."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _get_embedding(text_or_texts, model, tokenizer, max_length=512):
    """
    Получает эмбеддинг текста (или списка) через Qwen2.5 (mean-pool последнего hidden state).
    """
    if isinstance(text_or_texts, str):
        text_or_texts = [text_or_texts]

    inputs = tokenizer(
        text_or_texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

    # Берём последний слой hidden states: (batch_size, seq_len, hidden_dim)
    last_hidden = outputs.hidden_states[-1]

    # Mean pool по токенам (игнорируем padding)
    mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch_size, seq_len, 1)
    embeddings = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, hidden_dim)

    return embeddings.cpu().float().numpy()


def embed_chunks(chunks, model, tokenizer, batch_size=4):
    """Создаёт эмбеддинги для кусков через Qwen2.5 (батчами для скорости)."""
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        emb_batch = _get_embedding(batch, model, tokenizer)
        embeddings.append(emb_batch)
    return np.vstack(embeddings)


def create_index(embeddings):
    """Для совместимости возвращаем эмбеддинги (индекс не нужен)."""
    return embeddings


def retrieve(question, embeddings, chunks, model, tokenizer, k=3):
    """Ищет k самых похожих кусков на вопрос (по косинусной близости)."""
    query_vec = _get_embedding(question, model, tokenizer)
    similarities = cosine_similarity(query_vec.reshape(1, -1), embeddings)[0]
    top_k_idx = np.argsort(similarities)[-k:][::-1]
    return [chunks[i] for i in top_k_idx]
def retrieve_image(question, image_embeddings, image_paths, model, tokenizer):
    """Ищет 1 самое похожее изображение на вопрос (по косинусной близости). Возвращает путь."""
    if not len(image_embeddings):
        return None
        
    query_vec = _get_embedding(question, model, tokenizer)
    similarities = cosine_similarity(query_vec.reshape(1, -1), image_embeddings)[0]
    best_idx = np.argmax(similarities)
    
    # Можно добавить порог отсечения (similarity > threshold),
    # чтобы не возвращать мусор, если совпадения слабые.
    # if similarities[best_idx] < 0.2: return None
    
    return image_paths[best_idx]