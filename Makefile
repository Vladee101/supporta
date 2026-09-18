.PHONY: help install up down migrate seed index golden eval test lint fmt run poller consumer scheduler console demo loadtest nfr-report

PY ?= .venv/Scripts/python.exe   # Linux/macOS: make PY=.venv/bin/python

help:
	@echo "install  - зависимости (dev)"
	@echo "up       - postgres + rabbitmq в docker"
	@echo "down     - остановить инфраструктуру"
	@echo "migrate  - alembic upgrade head"
	@echo "seed     - загрузить базу знаний из data/kb_seed.json"
	@echo "index    - посчитать embedding'и (требует extra: embeddings)"
	@echo "golden   - собрать golden set"
	@echo "eval     - метрики качества (NFR2, NFR10)"
	@echo "test     - pytest"
	@echo "lint     - ruff check"
	@echo "run      - uvicorn на :8000"
	@echo "poller   - outbox poller"
	@echo "consumer - escalation consumer"
	@echo "scheduler- таймауты эскалаций"

install:
	$(PY) -m pip install -e ".[dev]"

up:
	docker compose up -d

down:
	docker compose down

migrate:
	$(PY) -m alembic upgrade head

seed:
	$(PY) -m scripts.seed_kb

index:
	$(PY) -m scripts.index_kb

golden:
	$(PY) -m scripts.gen_golden_set

eval:
	$(PY) -m scripts.eval --rag-threshold 0.15

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff format .

run:
	$(PY) -m uvicorn app.main:app --reload

poller:
	$(PY) -m app.workers.outbox_poller

consumer:
	$(PY) -m app.workers.escalation_consumer

scheduler:
	$(PY) -m app.workers.scheduler

console:
	cd console && npm install && npm run build

demo:
	$(PY) -m scripts.demo_tickets

loadtest:
	$(PY) -m scripts.load_test --env-file loadtest.env --label fixed

nfr-report:
	$(PY) -m scripts.nfr_report
