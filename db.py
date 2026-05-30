import os
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "vectordb")

def get_connection():
    return psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME
    )

def init_db():
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()
    
    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create tables
    # using a simple vector without strict dimensions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS text_chunks (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS image_chunks (
        id SERIAL PRIMARY KEY,
        image_path TEXT,
        bbox TEXT,
        embedding vector,
        text_chunk_id BIGINT REFERENCES text_chunks (id)
    );
    """)


    cur.close()
    conn.close()

def is_db_empty():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM text_chunks;")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count == 0
    except Exception:
        return True

def insert_text_chunks(chunks, embeddings):
    if len(chunks) == 0:
        return
        
    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor()
    for chunk, emb in zip(chunks, embeddings):
        # Flatten array just in case
        emb_arr = np.array(emb).flatten()
        cur.execute(
            "INSERT INTO text_chunks (content, embedding) VALUES (%s, %s)",
            (chunk, emb_arr)
        )
    conn.commit()
    cur.close()
    conn.close()

def insert_image_chunks(context_chunk, context_embedding, image_paths, bboxes, image_embeddings):
    if len(image_paths) == 0:
        return
        
    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor()
    context_emb_arr = np.array(context_embedding).flatten()
    cur.execute(
        "INSERT INTO text_chunks (content, embedding) VALUES (%s, %s) RETURNING id",
        (context_chunk, context_emb_arr)
    )
    text_chunk_id = cur.fetchone()[0]
    for path, bbox, emb in zip(image_paths, bboxes, image_embeddings):
        emb_arr = np.array(emb).flatten()

        cur.execute(
            "INSERT INTO image_chunks (image_path, bbox, embedding, text_chunk_id) VALUES (%s, %s, %s, %s)",
            (path, str(bbox), emb_arr, text_chunk_id)
        )
    conn.commit()
    cur.close()
    print(f"Загружен текстовый чанк {text_chunk_id}")
    conn.close()

def clear_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE text_chunks;")
    cur.execute("TRUNCATE TABLE image_chunks;")
    conn.commit()
    cur.close()
    conn.close()

def search_text_chunks(query_embedding, k=3):
    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor()
    
    query_arr = np.array(query_embedding).flatten()
    
    cur.execute("""
        SELECT id, content FROM text_chunks
        ORDER BY embedding <-> %s
        LIMIT %s;
    """, (query_arr, k))
    
    results = [(row[0], row[1]) for row in cur.fetchall()]
    cur.close()
    conn.close()
    
    return results

def search_image_chunks(text_chunk_ids, query_embedding):
    conn = get_connection()
    register_vector(conn)
    cur = conn.cursor()
    
    query_arr = np.array(query_embedding).flatten()
    
    cur.execute("""
        SELECT image_path, bbox FROM image_chunks
        WHERE text_chunk_id IN %s
        ORDER BY embedding <=> %s
        LIMIT 1;
    """, (tuple(text_chunk_ids), query_arr))
    
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    if rows:
        return rows
    return []
