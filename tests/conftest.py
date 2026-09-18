"""Фикстуры интеграционных тестов.

Тесты идут в отдельную базу `support_test`, а не в рабочую: они пишут тикеты,
эскалации и события outbox, и подмешивать это к данным разработчика нельзя.
Если Postgres недоступен, интеграционные тесты пропускаются - юнит-уровень
(Decision Engine, PII, классификатор, граф) от базы не зависит и идёт всегда.

Схема в тестовой базе создаётся из metadata, а не миграциями: соответствие
миграций моделям проверяется отдельно командой `alembic check` на рабочей базе.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Base, KbDocument, KbDocumentVersion
from app.services.embeddings import HashingEmbeddingProvider

TEST_DB_NAME = "support_test"

KB_FIXTURE = [
    (
        "delivery-terms",
        "Сроки доставки",
        "Доставка по Москве и Санкт-Петербургу занимает 1-2 рабочих дня, "
        "по регионам от 3 до 7 рабочих дней. Срок отсчитывается с момента "
        "подтверждения заказа.",
    ),
    (
        "payment-methods",
        "Способы оплаты",
        "Доступна оплата картой на сайте, через СБП по QR-коду, а также "
        "наличными или картой при получении в пунктах выдачи.",
    ),
    (
        "return-policy",
        "Возврат товара надлежащего качества",
        "Товар надлежащего качества можно вернуть в течение 14 дней с момента "
        "получения, если сохранены товарный вид и упаковка.",
    ),
    (
        "app-crash-on-start",
        "Приложение закрывается при запуске",
        "Сбой при запуске устраняется обновлением приложения и перезапуском "
        "устройства. Если не помогло, очистите кэш приложения.",
    ),
]


def pytest_configure(config) -> None:
    """Тесты не должны зависеть от локального .env разработчика.

    Мост RabbitMQ → WebSocket стартует в lifespan приложения; включённый в .env,
    он поднимался бы в каждом TestClient. Тест моста управляет им явно.
    """
    os.environ["WS_BRIDGE_ENABLED"] = "false"
    get_settings.cache_clear()


def _test_database_url() -> URL:
    """Возвращает объект URL, а не строку: `str(URL)` маскирует пароль как `***`."""
    explicit = os.getenv("TEST_DATABASE_URL")
    if explicit:
        return make_url(explicit)
    return make_url(get_settings().database_url).set(database=TEST_DB_NAME)


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    url = _test_database_url()
    admin_url = url.set(database="postgres")

    try:
        admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with admin.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": url.database},
            ).scalar()
            if not exists:
                connection.execute(text(f'CREATE DATABASE "{url.database}"'))
    except OperationalError as exc:
        pytest.skip(f"Postgres недоступен, интеграционные тесты пропущены: {exc}")

    engine = create_engine(url, future=True)
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    """Сессия в транзакции, которая откатывается после теста."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def indexed_kb(db_session: Session) -> list[KbDocumentVersion]:
    """Небольшая база знаний, проиндексированная хеширующим провайдером."""
    provider = HashingEmbeddingProvider()
    contents = [content for _, _, content in KB_FIXTURE]
    vectors = provider.encode(contents)

    versions: list[KbDocumentVersion] = []
    for (slug, title, content), vector in zip(KB_FIXTURE, vectors, strict=True):
        document = KbDocument(slug=slug, title=title)
        db_session.add(document)
        db_session.flush()

        version = KbDocumentVersion(
            document_id=document.id,
            version=1,
            content=content,
            embedding=vector,
            embedding_model=provider.model_id,
        )
        db_session.add(version)
        db_session.flush()
        document.current_version_id = version.id
        versions.append(version)

    db_session.flush()
    return versions
