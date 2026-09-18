"""Отправить набор демонстрационных обращений через публичный API.

    python -m scripts.demo_tickets --base-url http://127.0.0.1:8000

Идёт тем же путём, что и настоящий канал: подписанный вебхук → агент →
эскалации → outbox. Чтобы эскалации дошли до консоли, должны работать
outbox poller и escalation consumer.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

import httpx

from app.core.auth import webhook_signature
from app.core.config import get_settings

DEMO = [
    "Подскажите, какие способы оплаты доступны?",
    "Требую вернуть деньги за бракованный чайник, он не включается",
    "Пишу претензию: курьер нахамил, коробка пришла помятой",
    "Приложение вылетает при открытии корзины",
    "Где мой заказ 1042315? Оформлял пять дней назад",
    "Здравствуйте",
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Демо-обращения")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    secret = get_settings().channel_secrets["web"]
    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        for text in DEMO:
            body = json.dumps(
                {"channel": "web", "external_id": f"demo-{uuid.uuid4().hex[:12]}", "content": text},
                ensure_ascii=False,
            ).encode("utf-8")
            response = client.post(
                "/api/v1/tickets",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Signature": webhook_signature(body, secret),
                },
            )
            response.raise_for_status()
            result = response.json()
            reply = f" → {result['reply'][:60]}…" if result["reply"] else ""
            print(f"{result['status']:24} {text[:50]}{reply}")


if __name__ == "__main__":
    main()
