"""
Полный цикл самообучения: сканирует интернет, чистит найденное через Gemini,
дописывает в корпус, и если накопилось достаточно нового — дообучает модель.
Вызывается и вручную (кнопка /scan), и по расписанию (scheduler.py).
"""
import json
import logging

from . import config
from .scraper import collect_documents
from .gemini_filter import clean_batch, append_to_corpus
from .train import run_training_cycle

logger = logging.getLogger("pipeline")


def _load_state() -> dict:
    if config.STATE_PATH.exists():
        return json.loads(config.STATE_PATH.read_text(encoding="utf-8"))
    return {"docs_since_last_train": 0, "total_docs": 0, "cycles": 0, "last_error": None, "last_run_log": []}


def _save_state(state: dict):
    config.STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_full_cycle(topics: list[str] | None = None) -> dict:
    """scrape -> gemini clean -> append -> (retrain if enough new docs)
    topics: если задано — сканирует только эти темы вместо config.SEARCH_TOPICS
    (используется для разовых запросов из чата вида "ищи X")."""
    state = _load_state()
    state.setdefault("last_error", None)
    log_steps = []

    try:
        logger.info("скан интернета...")
        raw_docs = collect_documents(topics=topics)
        log_steps.append(f"scraper: найдено {len(raw_docs)} сырых страниц")
        logger.info(log_steps[-1])
    except Exception as e:
        state["last_error"] = f"scraper упал: {e!r}"
        state["last_run_log"] = log_steps
        _save_state(state)
        logger.exception("scraper упал")
        raise

    if not raw_docs:
        state["last_error"] = "поиск не вернул ни одной страницы (0 raw_docs) — см. last_run_log"
        state["last_run_log"] = log_steps
        _save_state(state)
        return {"new_docs": 0, "total_docs": state["total_docs"], "trained": False, "note": state["last_error"]}

    try:
        # Gemini фильтрация отключена: все найденные документы проходят напрямую
        cleaned = raw_docs
        log_steps.append(f"filter: из {len(raw_docs)} страниц все {len(cleaned)} прошли (фильтр Gemini отключен)")
        logger.info(log_steps[-1])
    except Exception as e:
        state["last_error"] = f"filter упал: {e!r}"
        state["last_run_log"] = log_steps
        _save_state(state)
        logger.exception("filter упал")
        raise

    if cleaned:
        append_to_corpus(cleaned)

    state["docs_since_last_train"] += len(cleaned)
    state["total_docs"] += len(cleaned)
    state["last_error"] = None if cleaned else "все документы отсеялись при очистке Gemini (0 прошло)"
    state["last_run_log"] = log_steps

    result = {"new_docs": len(cleaned), "total_docs": state["total_docs"], "trained": False}

    if state["docs_since_last_train"] >= config.MIN_NEW_DOCS_TO_RETRAIN:
        try:
            train_result = run_training_cycle(retrain_tokenizer=True)
            state["docs_since_last_train"] = 0
            state["cycles"] += 1
            result["trained"] = True
            result["train_result"] = train_result
        except Exception as e:
            state["last_error"] = f"обучение пропущено: {e!r}"
            logger.warning(state["last_error"])

    _save_state(state)
    return result
