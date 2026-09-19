"""
Использует твой ключ с ai.starimg.ru (прокси в формате OpenAI API, модель Gemini)
как редактора данных: отбирает только связный, осмысленный текст и приводит его
к чистому виду перед тем как это попадёт в обучающий корпус собственной модели.

Прокси OpenAI-совместимый (Base URL .../v1), поэтому запрос идёт в формате
chat/completions, а не через нативный SDK google-generativeai.
"""
import logging

import requests

from . import config

logger = logging.getLogger("gemini_filter")

CLEAN_PROMPT = """Ты — фильтр данных для обучения языковой модели.
Тебе дан сырой текст, вытащенный со страницы сайта. Твоя задача:
1. Убрать мусор: меню, рекламу, куки-баннеры, копирайты, навигацию, повторы.
2. Оставить только связный, осмысленный текст на русском или английском языке.
3. Если текст в основном мусорный или бессвязный — верни пустую строку.
4. Не добавляй ничего от себя, не суммаризируй, только очищай.

Верни ЧИСТЫЙ текст без комментариев и без markdown-разметки.

Текст:
---
{text}
---
"""


def _chat_completion(prompt: str) -> str:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY не задан в переменных окружения")

    url = f"{config.GEMINI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.GEMINI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def clean_document(raw_text: str) -> str:
    """Прогоняет один документ через прокси, возвращает очищенный текст (может быть пустым)."""
    prompt = CLEAN_PROMPT.format(text=raw_text[:8000])
    try:
        return _chat_completion(prompt)
    except Exception as e:
        logger.warning(f"gemini clean failed: {e}")
        return ""


def clean_batch(docs: list[dict]) -> list[dict]:
    """Чистит список документов, отбрасывая пустые и слишком короткие результаты."""
    out = []
    for d in docs:
        cleaned = clean_document(d["text"])
        if cleaned and len(cleaned) >= config.MIN_CHARS_PER_PAGE // 2:
            out.append({"topic": d["topic"], "url": d["url"], "text": cleaned})
    return out


def append_to_corpus(docs: list[dict]):
    """Дописывает очищенные документы в jsonl-корпус для обучения."""
    import json
    with open(config.RAW_CORPUS_PATH, "a", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info(f"добавлено {len(docs)} документов в корпус")
