import asyncio

import pytest
from pydantic import ValidationError
from fastapi import HTTPException

from api.main import Settings
from api.auth import _get_bearer_user, create_access_token


def valid_production_settings(**overrides):
    values = {
        "database_url": "postgresql+asyncpg://luminark:9aV7sM2xP8rQ4kN6wT3yZ1bC5dF0hJ8L@db.example.net/luminark",
        "redis_url": "redis://redis.example.net:6379/0",
        "jwt_secret_key": "a" * 48,
        "api_keys": "b" * 48,
        "allowed_origins": ["https://app.example.net"],
        "environment": "production",
    }
    values.update(overrides)
    return Settings(**values)


def test_production_accepts_strong_secrets_and_https_origin():
    settings = valid_production_settings()
    assert settings.environment == "production"
    assert len(settings.api_key_set) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"jwt_secret_key": "CHANGE_ME_USE_OPENSSL_RAND_HEX_32"},
        {"jwt_secret_key": "short"},
        {"database_url": "postgresql+asyncpg://user:secret@db.example.net/db"},
        {"allowed_origins": ["http://app.example.net"]},
        {"allowed_origins": ["*"]},
        {"api_keys": "short-key"},
    ],
)
def test_production_rejects_weak_or_insecure_configuration(overrides):
    with pytest.raises(ValidationError):
        valid_production_settings(**overrides)


def test_bearer_auth_fails_closed_when_redis_revocation_store_is_missing():
    settings = type(
        "SettingsStub",
        (),
        {"jwt_secret_key": "z" * 48, "jwt_algorithm": "HS256", "jwt_expire_minutes": 15},
    )()
    token, _ = create_access_token(settings, "00000000-0000-0000-0000-000000000001", ["analyze"])

    async def run():
        with pytest.raises(HTTPException) as exc:
            await _get_bearer_user(token, settings, db=None, redis=None)
        assert exc.value.status_code == 503

    asyncio.run(run())
