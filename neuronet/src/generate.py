"""
Загружает текущий чекпоинт и генерирует текст. Если модель ещё не обучена
(нет чекпоинта) — говорит об этом честно, а не выдаёт случайный шум как ответ.
"""
import torch

from . import config
from .model import TinyGPT
from .tokenizer_utils import load_or_train_tokenizer


def is_model_ready() -> bool:
    return config.MODEL_CKPT_PATH.exists() and config.TOKENIZER_PATH.exists()


def generate_text(prompt: str, max_new_tokens: int = 150, temperature: float = 0.9) -> str:
    if not is_model_ready():
        raise RuntimeError("Модель ещё не обучена ни на одном цикле — нет данных или обучения")

    tokenizer = load_or_train_tokenizer()
    model = TinyGPT(vocab_size=tokenizer.get_vocab_size())
    state = torch.load(config.MODEL_CKPT_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.to(config.DEVICE)
    model.eval()

    bos_id = tokenizer.token_to_id("<bos>")
    enc = tokenizer.encode(prompt)
    ids = [bos_id] + enc.ids
    idx = torch.tensor([ids], dtype=torch.long, device=config.DEVICE)

    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    return text
