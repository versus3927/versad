import psycopg2
from . import config
from .db import get_connection

def find_relevant(query: str, top_k: int | None = None) -> str:
    """Ищет релевантные документы в PostgreSQL и возвращает их как контекст."""
    top_k = top_k or config.RAG_TOP_K
    query_words = [f"'{w}'" for w in query.lower().split() if len(w) > 2]
    
    if not query_words:
        # Если запрос слишком короткий, просто берем последние документы
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT topic, url, text FROM corpus ORDER BY id DESC LIMIT %s", (top_k,))
                rows = cur.fetchall()
            conn.close()
            docs = [{"topic": r[0], "url": r[1], "text": r[2]} for r in rows]
        except Exception as e:
            return f"Ошибка поиска в БД: {e}"
    else:
        # Простой поиск по вхождениям слов через ILIKE
        # Строим запрос вида: text ILIKE '%слово1%' OR text ILIKE '%слово2%'...
        conditions = " OR ".join([f"text ILIKE '%{w}%'" for w in query_words])
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                # Сортируем по количеству совпадений (очень грубо)
                cur.execute(f"SELECT topic, url, text FROM corpus WHERE {conditions} LIMIT %s", (top_k,))
                rows = cur.fetchall()
            conn.close()
            docs = [{"topic": r[0], "url": r[1], "text": r[2]} for r in rows]
        except Exception as e:
            return f"Ошибка поиска в БД: {e}"

    if not docs:
        return ""

    snippets = []
    for d in docs:
        snippet = d["text"][:config.RAG_SNIPPET_CHARS]
        snippets.append(f"[тема: {d['topic']}, источник: {d['url']}]\n{snippet}")
    return "\n\n---\n\n".join(snippets)
