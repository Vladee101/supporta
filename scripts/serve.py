"""Запуск API с отдельным env-файлом (например, для нагрузочного стенда).

    python -m scripts.serve --env-file loadtest.env --port 8100

Значения из файла кладутся в окружение процесса до импорта приложения и
перекрывают `.env`: переменные окружения у pydantic-settings приоритетнее
env-файла.
"""

from __future__ import annotations

import argparse

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="API с отдельной конфигурацией")
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if not load_dotenv(args.env_file, override=True):
        raise SystemExit(f"env-файл не найден или пуст: {args.env_file}")
    uvicorn.run("app.main:app", port=args.port, workers=args.workers, log_level="warning")


if __name__ == "__main__":
    main()
