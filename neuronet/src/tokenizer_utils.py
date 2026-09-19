"""
Токенизатор обучается с нуля на твоём собранном корпусе (BPE), а не берётся
готовым от чужой модели — так словарь реально отражает твои данные.
"""
import json
import logging

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from . import config

logger = logging.getLogger("tokenizer")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def _iter_corpus_texts():
    if not config.RAW_CORPUS_PATH.exists():
        return
    with open(config.RAW_CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)["text"]
            except Exception:
                continue


def train_tokenizer() -> Tokenizer:
    """Обучает BPE-токенизатор на всём накопленном корпусе и сохраняет на диск."""
    texts = list(_iter_corpus_texts())
    if not texts:
        raise RuntimeError("Корпус пуст — сначала нужно собрать данные (scraper + gemini_filter)")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=config.VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(config.TOKENIZER_PATH))
    logger.info(f"токенизатор обучен, словарь: {tokenizer.get_vocab_size()} токенов")
    return tokenizer


def load_or_train_tokenizer() -> Tokenizer:
    if config.TOKENIZER_PATH.exists():
        return Tokenizer.from_file(str(config.TOKENIZER_PATH))
    return train_tokenizer()
