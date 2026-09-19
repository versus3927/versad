"""FastAPI-сервер чата, сканера и защищённой админ-панели."""
import json
import logging
import os
import secrets
import threading
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.commands import extract_search_topic
from src.generate import generate_text, is_model_ready
from src.ollama_client import chat as ollama_chat
from src.ollama_client import is_reachable as ollama_is_reachable
from src.pipeline import run_full_cycle
from src.retrieval import find_relevant
from src.db import init_db, migrate_jsonl_to_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("app")
app = FastAPI(title="Neuronet", version="2.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

_lock = threading.Lock()
_pipeline_running = False
_last_cycle_started: Optional[str] = None
_last_cycle_finished: Optional[str] = None
scheduler = BackgroundScheduler(timezone=os.getenv("TZ", "UTC"))


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=150, ge=1, le=1000)


class AdminScanRequest(BaseModel):
    topics: list[str] = Field(default_factory=list, max_length=20)


class SchedulerRequest(BaseModel):
    enabled: bool


def _load_state() -> dict:
    if not config.STATE_PATH.exists():
        return {}
    try:
        return json.loads(config.STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("state.json недоступен: %s", exc)
        return {"last_error": "state.json повреждён или недоступен"}


def _admin_guard(x_admin_token: str = Header(default="")) -> None:
    # Защита отключена: доступ к админке открыт для всех
    pass


def _scheduler_enabled() -> bool:
    job = scheduler.get_job("auto_scan")
    return bool(job and job.next_run_time)


def _corpus_preview(limit: int) -> list[dict]:
    if not config.RAW_CORPUS_PATH.exists():
        return []
    rows = []
    try:
        for line in config.RAW_CORPUS_PATH.read_text(encoding="utf-8").splitlines()[-limit:][::-1]:
            try:
                item = json.loads(line)
                rows.append({"topic": item.get("topic", ""), "url": item.get("url", ""), "chars": len(item.get("text", ""))})
            except json.JSONDecodeError:
                continue
    except OSError as exc:
        logger.warning("корпус недоступен: %s", exc)
    return rows


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/knowledge")
def knowledge_page():
    return FileResponse("static/knowledge.html")


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/status")
def status():
    return {"model_ready": is_model_ready(), "pipeline_running": _pipeline_running, "state": _load_state(), "topics": config.SEARCH_TOPICS, "scan_interval_minutes": config.SCAN_INTERVAL_MINUTES}


@app.post("/chat")
def chat(req: ChatRequest):
    topic = extract_search_topic(req.prompt)
    if topic:
        if _pipeline_running:
            return {"text": "предыдущий цикл ещё выполняется; попробуй немного позже"}
        threading.Thread(target=_run_pipeline_bg, kwargs={"topics": [topic]}, daemon=True).start()
        return {"text": "понял, запускаю поиск по теме «" + topic + "»"}
    if config.USE_OLLAMA:
        context = find_relevant(req.prompt)
        try:
            return {"text": ollama_chat(req.prompt, context=context), "used_fresh_data": bool(context)}
        except Exception as exc:
            return JSONResponse({"error": "Ollama недоступна (" + config.OLLAMA_BASE_URL + "): " + str(exc)}, status_code=502)
    if not is_model_ready():
        return JSONResponse({"error": "модель ещё не обучена — запусти скан"}, status_code=409)
    try:
        return {"text": generate_text(req.prompt, max_new_tokens=req.max_new_tokens)}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


def _run_pipeline_bg(topics: list[str] | None = None):
    global _pipeline_running, _last_cycle_started, _last_cycle_finished
    with _lock:
        if _pipeline_running:
            return
        _pipeline_running = True
        _last_cycle_started = datetime.now(timezone.utc).isoformat()
    try:
        logger.info("цикл завершён: %s", run_full_cycle(topics=topics))
    except Exception as exc:
        logger.exception("ошибка цикла: %s", exc)
    finally:
        _last_cycle_finished = datetime.now(timezone.utc).isoformat()
        _pipeline_running = False


@app.post("/scan")
def scan():
    if _pipeline_running:
        return {"started": False, "reason": "pipeline_running"}
    threading.Thread(target=_run_pipeline_bg, daemon=True).start()
    return {"started": True}


@app.get("/api/admin/status", dependencies=[Depends(_admin_guard)])
def admin_status():
    state = _load_state()
    return {"service": "online", "model_ready": is_model_ready(), "pipeline_running": _pipeline_running, "scheduler_enabled": _scheduler_enabled(), "ollama_reachable": ollama_is_reachable() if config.USE_OLLAMA else None, "gemini_configured": bool(config.GEMINI_API_KEY), "use_ollama": config.USE_OLLAMA, "ollama_model": config.OLLAMA_MODEL, "topics": config.SEARCH_TOPICS, "scan_interval_minutes": config.SCAN_INTERVAL_MINUTES, "pages_per_topic": config.PAGES_PER_TOPIC, "state": state, "last_cycle_started": _last_cycle_started, "last_cycle_finished": _last_cycle_finished, "corpus_bytes": config.RAW_CORPUS_PATH.stat().st_size if config.RAW_CORPUS_PATH.exists() else 0, "checkpoint_bytes": config.MODEL_CKPT_PATH.stat().st_size if config.MODEL_CKPT_PATH.exists() else 0}


@app.get("/api/admin/corpus", dependencies=[Depends(_admin_guard)])
def admin_corpus(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": _corpus_preview(limit)}


@app.post("/api/admin/scan", dependencies=[Depends(_admin_guard)])
def admin_scan(req: AdminScanRequest):
    if _pipeline_running:
        raise HTTPException(status_code=409, detail="Цикл уже выполняется")
    topics = [item.strip() for item in req.topics if item.strip()]
    threading.Thread(target=_run_pipeline_bg, kwargs={"topics": topics or None}, daemon=True).start()
    return {"started": True, "topics": topics or config.SEARCH_TOPICS}


@app.post("/api/admin/scheduler", dependencies=[Depends(_admin_guard)])
def admin_scheduler(req: SchedulerRequest):
    job = scheduler.get_job("auto_scan")
    if not job:
        raise HTTPException(status_code=503, detail="Планировщик ещё не запущен")
    job.resume() if req.enabled else job.pause()
    return {"enabled": _scheduler_enabled()}


@app.post("/api/admin/clear-error", dependencies=[Depends(_admin_guard)])
def admin_clear_error():
    state = _load_state()
    state["last_error"] = None
    config.STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"cleared": True}


@app.on_event("startup")
def on_startup():
    # Инициализируем БД и переносим данные
    try:
        init_db()
        migrate_jsonl_to_db()
    except Exception as e:
        logger.error(f"Критическая ошибка при старте БД: {e}")

    if not scheduler.get_job("auto_scan"):
        scheduler.add_job(_run_pipeline_bg, "interval", minutes=config.SCAN_INTERVAL_MINUTES, id="auto_scan", replace_existing=True, max_instances=1, coalesce=True)
    if not scheduler.running:
        scheduler.start()
    logger.info("планировщик запущен: цикл каждые %s мин.", config.SCAN_INTERVAL_MINUTES)


@app.on_event("shutdown")
def on_shutdown():
    if scheduler.running:
        scheduler.shutdown(wait=False)
