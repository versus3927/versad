"""
Общение с Ollama — уже предобученной моделью (она "знает" куда больше,
чем TinyGPT с случайными весами). Собранные и очищенные через Gemini данные
подмешиваются ей в контекст запроса (RAG), а не заново учат модель с нуля —
так свежесобранное сразу становится частью того, что она "знает" в ответе.
"""
import logging

import requests

from . import config

logger = logging.getLogger("ollama_client")

SYSTEM_PROMPT = (
    "Ты — ассистент, который постоянно сканирует интернет и пополняет свои знания. "
    "Если ниже дан раздел 'Свежие данные из интернета' — используй его как дополнительный, "
    "более актуальный источник, наравне со своими знаниями. Если он не относится к вопросу — "
    "просто отвечай как обычно."
)


def is_reachable() -> bool:
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=5)
        return r.ok
    except Exception:
        return False


def chat(prompt: str, context: str = "") -> str:
    user_content = prompt
    if context:
        user_content = f"Свежие данные из интернета:\n{context}\n\nВопрос: {prompt}"

    url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message", {}).get("content") or "").strip()
