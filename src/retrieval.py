"""
Простой keyword-based поиск по собранному корпусу (без embeddings — быстро
и без лишних зависимостей). Находит документы, пересекающиеся по словам
с вопросом пользователя, и отдаёт их как контекст для Ollama.
"""
import json

from . import config


def _load_corpus() -> list[dict]:
    if not config.RAW_CORPUS_PATH.exists():
        return []
    docs = []
    with open(config.RAW_CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except Exception:
                continue
    return docs


def find_relevant(query: str, top_k: int | None = None) -> str:
    """Возвращает склеенные фрагменты наиболее релевантных документов, или '' если корпус пуст."""
    top_k = top_k or config.RAG_TOP_K
    docs = _load_corpus()
    if not docs:
        return ""

    query_words = {w for w in query.lower().split() if len(w) > 2}
    scored = []
    for d in docs:
        text_words = set(d["text"].lower().split())
        overlap = len(query_words & text_words)
        scored.append((overlap, d))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = [d for score, d in scored[:top_k] if score > 0]
    if not top:
        top = docs[-top_k:]  # если по словам ничего не нашлось — берём последнее собранное

    snippets = []
    for d in top:
        snippet = d["text"][:config.RAG_SNIPPET_CHARS]
        snippets.append(f"[тема: {d['topic']}, источник: {d['url']}]\n{snippet}")
    return "\n\n---\n\n".join(snippets)
