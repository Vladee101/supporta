# Образ приложения: API, воркеры и одноразовые задачи стенда - один образ, разные команды.
# Демо-стенд целиком: docker compose --profile demo up -d  (см. README).

# --- консоль оператора -------------------------------------------------------
FROM node:22-slim AS console
WORKDIR /console
COPY console/package.json console/package-lock.json ./
RUN npm ci
COPY console/ ./
RUN npm run build

# --- приложение --------------------------------------------------------------
FROM python:3.14-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Зависимости - отдельным слоем: правка кода не переустанавливает их заново.
# bge-m3 (extra embeddings, ~3 ГБ с torch) в образ не входит: стенд работает на
# хеширующих векторах, порог RAG для них задан в docker-compose.yml.
COPY pyproject.toml ./
COPY app/__init__.py ./app/__init__.py
RUN pip install -e .

COPY app ./app
COPY alembic.ini ./
COPY migrations ./migrations
COPY scripts ./scripts
COPY data ./data
COPY --from=console /console/dist ./console/dist

RUN useradd --create-home --uid 10001 support
USER support

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
