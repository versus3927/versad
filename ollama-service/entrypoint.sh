#!/bin/sh
set -e

# Поднимаем сервер Ollama в фоне
ollama serve &
SERVER_PID=$!

# Ждём, пока сервер реально начнёт отвечать
echo "ждём запуска ollama serve..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "стягиваем модель: ${OLLAMA_MODEL}"
ollama pull "${OLLAMA_MODEL}" || echo "не удалось стянуть модель, продолжаем без неё"

# Держим процесс живым
wait $SERVER_PID
