import psycopg2
from psycopg2.extras import execute_values
import logging
from . import config

logger = logging.getLogger("db_utils")

def get_connection():
    return psycopg2.connect(config.DATABASE_URL)

def init_db():
    """Создает таблицу corpus, если она еще не существует."""
    query = """
    CREATE TABLE IF NOT EXISTS corpus (
        id SERIAL PRIMARY KEY,
        topic TEXT,
        url TEXT,
        text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_topic ON corpus(topic);
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()
        conn.close()
        logger.info("База данных инициализирована успешно")
    except Exception as e:
        logger.error(f"Ошибка при инициализации БД: {e}")
        raise

def migrate_jsonl_to_db():
    """Переносит данные из raw_corpus.jsonl в PostgreSQL."""
    if not config.RAW_CORPUS_PATH.exists():
        logger.info("Файл raw_corpus.jsonl не найден, миграция не требуется")
        return

    try:
        with open(config.RAW_CORPUS_PATH, encoding="utf-8") as f:
            lines = f.readlines()
        
        docs = []
        for line in lines:
            try:
                item = json.loads(line)
                docs.append((item.get("topic"), item.get("url"), item.get("text")))
            except json.JSONDecodeError:
                continue

        if not docs:
            logger.info("Нет данных для миграции")
            return

        conn = get_connection()
        with conn.cursor() as cur:
            execute_values(cur, 
                "INSERT INTO corpus (topic, url, text) VALUES %s", 
                docs)
        conn.commit()
        conn.close()
        logger.info(f"Успешно перенесено {len(docs)} документов в БД")
    except Exception as e:
        logger.error(f"Ошибка миграции: {e}")
