"""
Дообучает TinyGPT на текущем корпусе. Каждый вызов делает config.TRAIN_STEPS_PER_CYCLE
шагов, сохраняет чекпоинт — так модель постепенно улучшается по мере роста корпуса,
без необходимости обучать с нуля каждый раз.
"""
import json
import logging

import torch

from . import config
from .model import TinyGPT
from .tokenizer_utils import load_or_train_tokenizer, train_tokenizer

logger = logging.getLogger("train")


def _load_corpus_ids(tokenizer) -> torch.Tensor:
    ids = []
    with open(config.RAW_CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            enc = tokenizer.encode(text)
            ids.extend(enc.ids + [tokenizer.token_to_id("<eos>")])
    return torch.tensor(ids, dtype=torch.long)


def _get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise RuntimeError("Корпус слишком маленький для текущего block_size — собери больше данных")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def load_model(vocab_size: int) -> TinyGPT:
    model = TinyGPT(vocab_size=vocab_size)
    if config.MODEL_CKPT_PATH.exists():
        state = torch.load(config.MODEL_CKPT_PATH, map_location=config.DEVICE)
        model.load_state_dict(state)
        logger.info("загружен существующий чекпоинт модели")
    else:
        logger.info("чекпоинт не найден — стартуем со случайных весов (модель с нуля)")
    return model.to(config.DEVICE)


def run_training_cycle(retrain_tokenizer: bool = False) -> dict:
    """Один цикл: (опционально) переобучает токенизатор, дообучает модель, сохраняет чекпоинт."""
    tokenizer = train_tokenizer() if retrain_tokenizer else load_or_train_tokenizer()
    data = _load_corpus_ids(tokenizer)
    if len(data) < config.BLOCK_SIZE * 2:
        raise RuntimeError("Недостаточно данных для обучения — нужно больше собранных документов")

    model = load_model(tokenizer.get_vocab_size())
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    losses = []
    for step in range(config.TRAIN_STEPS_PER_CYCLE):
        x, y = _get_batch(data, config.BLOCK_SIZE, config.BATCH_SIZE, config.DEVICE)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % 20 == 0:
            logger.info(f"step {step}: loss {loss.item():.4f}")

    torch.save(model.state_dict(), config.MODEL_CKPT_PATH)
    avg_loss = sum(losses) / len(losses)
    logger.info(f"цикл обучения завершён, средний loss: {avg_loss:.4f}")
    return {"steps": config.TRAIN_STEPS_PER_CYCLE, "avg_loss": avg_loss, "corpus_tokens": len(data)}
