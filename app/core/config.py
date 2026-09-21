"""Конфигурация приложения.

Пороги маршрутизации живут здесь, а не в коде правил: они калибруются на
golden set и меняются при смене модели (см. «Confidence и пороги»).
Домен ничего не знает про pydantic - `thresholds()` отдаёт простой dataclass.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.singleton import once
from app.domain.decision import Thresholds


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://support:support@localhost:5432/support"
    rabbitmq_url: str = "amqp://support:support@localhost:5672/"

    # Ёмкость процесса API (NFR8, раздел «Бюджет задержки и пропускная способность»).
    # Потоки: каждый тикет занимает поток на всё время ожидания LLM, поэтому
    # потоков нужно не меньше, чем одновременных тикетов, с запасом.
    # Соединения: освобождаются на время LLM, поэтому их нужно заметно меньше.
    api_worker_threads: int = Field(default=100, ge=1)
    db_pool_size: int = Field(default=10, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)
    db_pool_timeout_seconds: float = Field(default=10.0, gt=0)
    # Без таймаута подключение к Postgres может висеть вечно: порт-прокси
    # (Docker Desktop, балансировщик) принимает TCP, а сервера за ним нет.
    # Целые секунды, минимум 2 - ограничения libpq.
    db_connect_timeout_seconds: int = Field(default=5, ge=2)

    # Пороги маршрутизации (калибруются, см. «Confidence и пороги»).
    class_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    rag_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_clarifications: int = Field(default=1, ge=0)

    # Таймауты.
    clarification_timeout_minutes: int = Field(default=30, ge=1)  # NFR9
    escalation_claim_ttl_minutes: int = Field(default=15, ge=1)  # FR10
    client_escalation_window_hours: int = Field(default=24, ge=1)  # UC9
    raw_ticket_retention_days: int = Field(default=90, ge=1)  # NFR4

    # RAG.
    embedding_model: str = "BAAI/bge-m3"  # ADR-006
    embedding_dim: int = 1024
    rag_top_k: int = Field(default=5, ge=1)
    #: auto - bge-m3, если установлен, иначе хеширующий; bge / hashing - явно.
    #: Порог RAG откалиброван под конкретный провайдер: при hashing нужен свой.
    embedding_provider: Literal["auto", "bge", "hashing"] = "auto"

    # Каналы приёма (UC1). Ключ - имя канала, значение - секрет подписи вебхука.
    # Канал, которого здесь нет, считается неподдерживаемым (422).
    channel_secrets: dict[str, str] = {"web": "dev-only-web-secret"}
    #: Срок жизни токена клиента: должен покрывать окно эскалации после автоответа.
    ticket_token_ttl_hours: int = Field(default=168, ge=1)

    # LLM-провайдер (ADR-009). baseline - словарный классификатор и шаблонные
    # ответы без сети; остальные - адаптеры app.services.llm_adapters.
    llm_provider: Literal["baseline", "openai_compatible", "gigachat"] = "baseline"
    llm_classify_model: str = ""
    llm_generate_model: str = ""
    #: auto - logprobs, если провайдер их отдаёт, иначе k-sampling (ADR-009).
    llm_confidence_mode: Literal["auto", "logprobs", "k_sampling"] = "auto"
    llm_k_samples: int = Field(default=5, ge=1, le=15)
    #: None - не передавать температуру (для провайдеров, которые её не принимают).
    llm_sampling_temperature: float | None = 0.7
    # NFR6: повторы и общий дедлайн вызова.
    llm_max_retries: int = Field(default=3, ge=0)
    llm_retry_deadline_seconds: float = Field(default=10.0, gt=0)
    llm_request_timeout_seconds: float = Field(default=8.0, gt=0)
    #: Путь к бандлу корневых сертификатов (GigaChat - Russian Trusted Root CA).
    llm_ca_bundle: str | None = None
    # OpenAI-совместимый API: YandexGPT, vLLM и т.п.
    llm_base_url: str = ""
    llm_api_key: str | None = None
    llm_auth_scheme: str = "Bearer"
    llm_choice_constraint: Literal["none", "guided_choice", "structured_outputs"] = "none"
    # GigaChat.
    gigachat_credentials: str | None = None
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    gigachat_chat_url: str = "https://api.giga.chat/v2/chat/completions"

    # Имитация задержки LLM для нагрузочного теста (app.services.latency).
    # 0 - выключено; в рабочей конфигурации должно быть 0.
    simulated_llm_classify_ms: int = Field(default=0, ge=0)
    simulated_llm_generate_ms: int = Field(default=0, ge=0)

    # Аутентификация операторов. Значение по умолчанию годится только для
    # локальной разработки - в любом общем окружении AUTH_SECRET обязателен.
    auth_secret: str = "dev-only-insecure-secret-change-me"
    operator_token_ttl_hours: int = Field(default=12, ge=1)

    # Мост RabbitMQ → WebSocket в процессе API. Выключается в тестах и там,
    # где консоль работает только через REST.
    ws_bridge_enabled: bool = False


@once
def get_settings() -> Settings:
    return Settings()


def thresholds() -> Thresholds:
    s = get_settings()
    return Thresholds(
        class_confidence=s.class_confidence_threshold,
        rag_confidence=s.rag_confidence_threshold,
        max_clarifications=s.max_clarifications,
    )
