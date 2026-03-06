"""
LUMINARK API — Auth Router
Complete OAuth2-compatible authentication with:
  • JWT access tokens (short-lived, JTI blacklisting on logout)
  • Opaque refresh tokens with rotation + reuse detection
  • Per-user API key management (DB-stored, sha256-hashed)
  • Audit logging on every auth event
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from .repositories import (
    AuditRepository, APIKeyRepository, RefreshTokenRepository, UserRepository, sha256_hex
)
from .schemas import (
    APIKeyCreatedOut, APIKeyOut, CreateAPIKeyRequest, CurrentUser,
    LogoutRequest, RefreshRequest, TokenResponse, UserOut
)

logger = structlog.get_logger("luminark.auth")

router = APIRouter(prefix="/auth", tags=["auth"])

# ─── Security schemes ──────────────────────────────────────────────────────
oauth2_scheme  = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

# ─── Crypto ────────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ─── Token helpers ─────────────────────────────────────────────────────────

def create_access_token(settings, user_id: str, scopes: List[str]) -> tuple[str, int]:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub":    user_id,
        "scopes": scopes,
        "exp":    exp,
        "iat":    datetime.now(timezone.utc),
        "jti":    str(uuid.uuid4()),
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, settings.jwt_expire_minutes * 60


def decode_access_token(settings, token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("Invalid token") from exc


def generate_refresh_token() -> tuple[str, str]:
    """Returns (raw_token, sha256_hash)."""
    raw  = secrets.token_urlsafe(64)
    hsh  = sha256_hex(raw)
    return raw, hsh


# ─── Dependency helpers (injected via app.state) ───────────────────────────

def _get_settings(request: Request):
    return request.app.state.settings

def _get_engine_ref(request: Request):
    return request.app.state.engine


async def _get_bearer_user(
    token: Optional[str],
    settings,
    db: AsyncSession,
    redis: Redis,
) -> Optional[CurrentUser]:
    if not token:
        return None
    try:
        payload = decode_access_token(settings, token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    jti = payload.get("jti")
    if jti and await redis.exists(f"blacklist:{jti}"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token_revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_token")

    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(uuid.UUID(user_id))
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user_inactive")

    return CurrentUser(
        id=str(user.id),
        username=user.username,
        scopes=payload.get("scopes", []),
        via="jwt",
        jti=jti,
    )


async def _get_apikey_user(
    raw_key: Optional[str],
    settings,
    db: AsyncSession,
) -> Optional[CurrentUser]:
    if not raw_key:
        return None

    if raw_key in settings.api_key_set:
        return CurrentUser(
            id="static",
            username="service-account",
            scopes=["analyze", "batch"],
            via="apikey",
        )

    key_hash = sha256_hex(raw_key)
    key_repo = APIKeyRepository(db)
    db_key   = await key_repo.get_by_hash(key_hash)
    if not db_key:
        return None

    user_repo = UserRepository(db)
    user      = await user_repo.get_by_id(db_key.user_id)
    if not user or not user.is_active:
        return None

    return CurrentUser(
        id=str(user.id),
        username=user.username,
        scopes=db_key.scopes,
        via="apikey",
    )


# Public dependency — call from any route
async def get_current_user(
    request: Request,
    bearer_token: Optional[str] = Depends(oauth2_scheme),
    raw_api_key:  Optional[str] = Depends(api_key_scheme),
) -> CurrentUser:
    settings = request.app.state.settings
    db       = request.app.state.db_session_factory()
    redis    = request.app.state.redis

    async with db as session:
        user = await _get_apikey_user(raw_api_key, settings, session)
        if not user:
            user = await _get_bearer_user(bearer_token, settings, session, redis)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="provide_bearer_or_api_key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_scope(*scopes: str):
    """Dependency factory: raises 403 if user lacks all listed scopes."""
    async def _guard(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not any(s in user.scopes for s in scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"insufficient_scope: need one of {scopes}",
            )
        return user
    return _guard


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/token", response_model=TokenResponse, summary="Login — get access + refresh tokens")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    settings = request.app.state.settings

    async with request.app.state.db_session_factory() as db:
        user_repo    = UserRepository(db)
        rt_repo      = RefreshTokenRepository(db)
        audit_repo   = AuditRepository(db)

        user = await user_repo.get_by_username(form_data.username)
        # Constant-time check — identical error regardless of whether user exists
        if not user or not user.is_active or not verify_password(form_data.password, user.hashed_password):
            await audit_repo.log("login_failed", "auth", request, detail={"username": form_data.username})
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid_credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token, expires_in = create_access_token(settings, str(user.id), user.scopes)
        raw_refresh, refresh_hash = generate_refresh_token()
        family_id  = uuid.uuid4()
        expires_at = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)

        await rt_repo.create(user.id, refresh_hash, family_id, expires_at)
        await user_repo.update_last_login(user.id)
        await audit_repo.log("login", "auth", request, user_id=user.id)
        await db.commit()

    logger.info("user_login", user_id=str(user.id))
    return TokenResponse(
        access_token=access_token,
        expires_in=expires_in,
        refresh_token=raw_refresh,
    )


@router.post("/refresh", response_model=TokenResponse, summary="Rotate refresh token")
async def refresh_token_endpoint(
    request: Request,
    body: RefreshRequest,
):
    settings = request.app.state.settings

    async with request.app.state.db_session_factory() as db:
        rt_repo    = RefreshTokenRepository(db)
        user_repo  = UserRepository(db)
        audit_repo = AuditRepository(db)

        token_hash = sha256_hex(body.refresh_token)
        row        = await rt_repo.get_by_hash(token_hash)

        if not row:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_refresh_token")

        if row.is_revoked:
            # Replay attack — revoke entire family
            await rt_repo.revoke_family(row.family_id)
            await audit_repo.log("refresh_reuse_attack", "auth", request, user_id=row.user_id)
            await db.commit()
            logger.warning("refresh_reuse_attack", family_id=str(row.family_id))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="token_family_revoked",
            )

        if row.expires_at < datetime.now(timezone.utc):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="refresh_token_expired")

        user = await user_repo.get_by_id(row.user_id)
        if not user or not user.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user_inactive")

        new_access, expires_in    = create_access_token(settings, str(user.id), user.scopes)
        new_raw_refresh, new_hash = generate_refresh_token()
        new_expires               = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)

        await rt_repo.rotate(row, new_hash, new_expires)
        await audit_repo.log("token_refresh", "auth", request, user_id=user.id)
        await db.commit()

    return TokenResponse(
        access_token=new_access,
        expires_in=expires_in,
        refresh_token=new_raw_refresh,
    )


@router.post("/logout", summary="Revoke current session")
async def logout(
    request: Request,
    body: LogoutRequest,
    bearer_token: Optional[str] = Depends(oauth2_scheme),
):
    settings = request.app.state.settings
    redis    = request.app.state.redis

    async with request.app.state.db_session_factory() as db:
        rt_repo    = RefreshTokenRepository(db)
        audit_repo = AuditRepository(db)

        token_hash = sha256_hex(body.refresh_token)
        await rt_repo.revoke_one(token_hash)

        if bearer_token:
            try:
                payload = decode_access_token(settings, bearer_token)
                jti = payload.get("jti")
                if jti:
                    exp = payload.get("exp", 0)
                    ttl = max(int(exp - datetime.now(timezone.utc).timestamp()), 1)
                    await redis.setex(f"blacklist:{jti}", ttl, "1")
            except ValueError:
                pass  # Token already invalid — safe to ignore

        user_id_str = None
        row = await rt_repo.get_by_hash(token_hash)
        if row:
            user_id_str = row.user_id

        await audit_repo.log("logout", "auth", request, user_id=user_id_str)
        await db.commit()

    return {"status": "logged_out"}


@router.post("/logout-all", summary="Revoke all sessions for current user")
async def logout_all(
    request: Request,
    bearer_token: Optional[str] = Depends(oauth2_scheme),
    current_user: CurrentUser = Depends(get_current_user),
):
    settings = request.app.state.settings
    redis    = request.app.state.redis

    async with request.app.state.db_session_factory() as db:
        rt_repo    = RefreshTokenRepository(db)
        audit_repo = AuditRepository(db)

        revoked = await rt_repo.revoke_all_for_user(uuid.UUID(current_user.id))

        if bearer_token and current_user.jti:
            try:
                payload = decode_access_token(settings, bearer_token)
                exp = payload.get("exp", 0)
                ttl = max(int(exp - datetime.now(timezone.utc).timestamp()), 1)
                await redis.setex(f"blacklist:{current_user.jti}", ttl, "1")
            except ValueError:
                pass

        await audit_repo.log(
            "logout_all", "auth", request,
            user_id=uuid.UUID(current_user.id),
            detail={"sessions_revoked": revoked},
        )
        await db.commit()

    return {"status": "logged_out_all", "sessions_revoked": revoked}


@router.get("/me", response_model=UserOut, summary="Current user profile")
async def me(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    async with request.app.state.db_session_factory() as db:
        user_repo = UserRepository(db)
        user = await user_repo.get_by_id(uuid.UUID(current_user.id))

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user_not_found")

    return UserOut(
        id=str(user.id),
        username=user.username,
        email=user.email,
        scopes=user.scopes,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=user.created_at,
    )


@router.post("/keys", response_model=APIKeyCreatedOut, summary="Create API key")
async def create_api_key(
    request: Request,
    body: CreateAPIKeyRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    raw_key    = secrets.token_urlsafe(32)
    key_prefix = raw_key[:8]
    key_hash   = sha256_hex(raw_key)

    async with request.app.state.db_session_factory() as db:
        key_repo   = APIKeyRepository(db)
        audit_repo = AuditRepository(db)

        db_key = await key_repo.create(
            user_id=uuid.UUID(current_user.id),
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=body.name,
            scopes=body.scopes,
        )

        await audit_repo.log(
            "api_key_created", "auth", request,
            user_id=uuid.UUID(current_user.id),
            detail={"key_name": body.name, "scopes": body.scopes},
        )
        await db.commit()

    logger.info("api_key_created", user_id=current_user.id, key_prefix=key_prefix)
    return APIKeyCreatedOut(
        id=str(db_key.id),
        name=db_key.name,
        key_prefix=key_prefix,
        scopes=db_key.scopes,
        is_active=db_key.is_active,
        created_at=db_key.created_at,
        key_value=raw_key,  # Shown ONCE
    )


@router.get("/keys", response_model=List[APIKeyOut], summary="List API keys")
async def list_api_keys(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    async with request.app.state.db_session_factory() as db:
        key_repo = APIKeyRepository(db)
        keys     = await key_repo.list_for_user(uuid.UUID(current_user.id))

    return [
        APIKeyOut(
            id=str(k.id),
            name=k.name,
            key_prefix=k.key_prefix,
            scopes=k.scopes,
            is_active=k.is_active,
            created_at=k.created_at,
        )
        for k in keys
    ]


@router.delete("/keys/{key_id}", summary="Revoke an API key")
async def revoke_api_key(
    key_id: uuid.UUID,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    async with request.app.state.db_session_factory() as db:
        key_repo   = APIKeyRepository(db)
        audit_repo = AuditRepository(db)

        revoked = await key_repo.revoke(key_id, uuid.UUID(current_user.id))
        if not revoked:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="key_not_found")

        await audit_repo.log(
            "api_key_revoked", "auth", request,
            user_id=uuid.UUID(current_user.id),
            detail={"key_id": str(key_id)},
        )
        await db.commit()

    return {"status": "revoked", "key_id": str(key_id)}
