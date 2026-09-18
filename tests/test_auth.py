"""Тесты токенов операторов."""

from __future__ import annotations

import uuid

import pytest

from app.core.auth import InvalidTokenError, issue_token, verify_token
from app.domain.enums import OperatorRole

SECRET = "test-secret"


def test_round_trip_preserves_identity_and_role():
    operator_id = uuid.uuid4()
    token = issue_token(operator_id, OperatorRole.ADMIN, secret=SECRET, ttl_seconds=60)
    principal = verify_token(token, secret=SECRET)
    assert principal.operator_id == operator_id
    assert principal.role is OperatorRole.ADMIN


def test_token_signed_with_other_secret_is_rejected():
    token = issue_token(uuid.uuid4(), OperatorRole.OPERATOR, secret="other", ttl_seconds=60)
    with pytest.raises(InvalidTokenError, match="подпись"):
        verify_token(token, secret=SECRET)


def test_tampered_payload_is_rejected():
    """Подменить роль в payload без секрета нельзя."""
    token = issue_token(uuid.uuid4(), OperatorRole.OPERATOR, secret=SECRET, ttl_seconds=60)
    forged = issue_token(uuid.uuid4(), OperatorRole.ADMIN, secret="attacker", ttl_seconds=60)
    tampered = f"{forged.split('.')[0]}.{token.split('.')[1]}"
    with pytest.raises(InvalidTokenError):
        verify_token(tampered, secret=SECRET)


def test_expired_token_is_rejected():
    token = issue_token(
        uuid.uuid4(), OperatorRole.OPERATOR, secret=SECRET, ttl_seconds=10, now=1_000_000
    )
    assert verify_token(token, secret=SECRET, now=1_000_009)
    with pytest.raises(InvalidTokenError, match="истёк"):
        verify_token(token, secret=SECRET, now=1_000_010)


@pytest.mark.parametrize("token", ["", "garbage", "a.b.c", "abc.def"])
def test_malformed_tokens_are_rejected(token):
    with pytest.raises(InvalidTokenError):
        verify_token(token, secret=SECRET)
