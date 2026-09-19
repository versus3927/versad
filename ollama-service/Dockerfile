# Отдельный сервис в том же Railway-проекте. Разворачивает Ollama,
# при старте сам стягивает модель (см. OLLAMA_MODEL ниже) и держит её
# готовой. Главное приложение (папка neuronet) стучится сюда по
# приватной сети Railway, а не через интернет.
FROM ollama/ollama:latest

ENV OLLAMA_MODEL=llama3.2
ENV OLLAMA_HOST=0.0.0.0:11434

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 11434
ENTRYPOINT ["/entrypoint.sh"]
