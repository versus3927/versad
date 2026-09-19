"""
Лёгкий слой команд поверх чата. Сама модель (TinyGPT) не понимает команды —
она только предсказывает текст. Поэтому фразы вида "ищи X" или "найди
информацию про X" перехватываются ЗДЕСЬ, до генерации, и превращаются
в разовый запуск пайплайна сбора данных по теме X.
"""
import re

TRIGGERS = [
    r"^ищи\s+(.+)$",
    r"^найди\s+(?:информацию\s+(?:про|о)\s+)?(.+)$",
    r"^поищи\s+(.+)$",
    r"^собери\s+данные\s+(?:про|о)\s+(.+)$",
    r"^изучи\s+(.+)$",
    r"^search\s+(.+)$",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in TRIGGERS]


def extract_search_topic(message: str) -> str | None:
    """Если сообщение — команда на поиск, возвращает тему. Иначе None."""
    text = message.strip()
    for pattern in _COMPILED:
        m = pattern.match(text)
        if m:
            topic = m.group(1).strip(" .!?")
            if topic:
                return topic
    return None
