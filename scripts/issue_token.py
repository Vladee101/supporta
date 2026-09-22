"""Выдать токен оператору (создаёт оператора, если его нет).

    python -m scripts.issue_token --email anna@example.com --name "Анна" --role operator

Для локальной разработки и демо. В общем окружении токены выдаёт IdP,
а AUTH_SECRET обязан отличаться от значения по умолчанию.
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import select

from app.core.auth import issue_token
from app.core.config import get_settings
from app.db.base import get_session_factory
from app.db.models import Operator
from app.domain.enums import OperatorRole

DEFAULT_SECRET = "dev-only-insecure-secret-change-me"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Выдать токен оператору")
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--role", choices=[role.value for role in OperatorRole], default="operator")
    parser.add_argument(
        "--console-url",
        default=None,
        help="напечатать ссылку для входа <url>#token=...: фрагмент не уходит на сервер",
    )
    args = parser.parse_args()

    if get_settings().auth_secret == DEFAULT_SECRET:
        print("внимание: используется AUTH_SECRET по умолчанию - только для локальной разработки")

    with get_session_factory()() as session:
        operator = session.scalar(select(Operator).where(Operator.email == args.email))
        if operator is None:
            operator = Operator(email=args.email, name=args.name or args.email, role=args.role)
            session.add(operator)
            session.commit()
            print(f"создан оператор {operator.email} ({operator.role})")

        token = issue_token(operator.id, OperatorRole(operator.role))

    if args.console_url:
        print()
        print(f"консоль оператора: {args.console_url.rstrip('/')}/#token={token}")
        print()
    else:
        print(token)


if __name__ == "__main__":
    main()
