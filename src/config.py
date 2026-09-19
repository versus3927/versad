"""
Центральная конфигурация. Всё, что меняется между запусками, живёт тут
или в переменных окружения (Railway -> Variables).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CKPT_DIR = DATA_DIR / "checkpoints"   # внутри data/, чтобы хватило ОДНОГО Volume на Railway
DATA_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

RAW_CORPUS_PATH = DATA_DIR / "raw_corpus.jsonl"       # сырые чистые тексты после фильтра Gemini
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
MODEL_CKPT_PATH = CKPT_DIR / "model.pt"
STATE_PATH = DATA_DIR / "state.json"                  # прогресс: сколько строк уже видела модель и т.д.

# --- Поиск и сбор данных ---
SEARCH_TOPICS = [t.strip() for t in os.getenv(
    "SEARCH_TOPICS",
    "математика,алгебра,геометрия,физика,химия,биология,астрономия,история,география,литература,обществознание,философия,информатика,программирование,технологии,наука,искусственный интеллект,робототехника,энциклопедия знаний"
).split(",") if t.strip()]
PAGES_PER_TOPIC = int(os.getenv("PAGES_PER_TOPIC", "5"))
SCAN_INTERVAL_MINUTES = 1  # Жестко ставим 1 минуту, игнорируя переменную окружения
MIN_CHARS_PER_PAGE = 400

# --- Gemini через сторонний OpenAI-совместимый прокси (ai.starimg.ru) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://ai.starimg.ru/v1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# --- Модель (обучается с нуля, веса случайные при первом запуске) ---
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "8000"))
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "256"))       # длина контекста
N_LAYER = int(os.getenv("N_LAYER", "4"))
N_HEAD = int(os.getenv("N_HEAD", "4"))
N_EMBD = int(os.getenv("N_EMBD", "256"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))

# --- Обучение ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-4"))
TRAIN_STEPS_PER_CYCLE = int(os.getenv("TRAIN_STEPS_PER_CYCLE", "100"))  # шагов за один цикл дообучения
MIN_NEW_DOCS_TO_RETRAIN = int(os.getenv("MIN_NEW_DOCS_TO_RETRAIN", "10"))

DEVICE = os.getenv("DEVICE", "cpu")  # на Railway обычно нет GPU -> cpu

# --- База данных ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() in ("1", "true", "yes")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-service:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))          # сколько документов подмешивать в контекст
RAG_SNIPPET_CHARS = int(os.getenv("RAG_SNIPPET_CHARS", "800"))
