FROM python:3.11-slim

WORKDIR /app

# Метаданные и пакет (зависимости из pyproject.toml)
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY config.yaml ./config.yaml

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

RUN mkdir -p /app/data

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["deepseek-cursor-proxy", "--config", "/app/config.yaml"]
