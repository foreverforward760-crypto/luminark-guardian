"""
LUMINARK API — Async SQLAlchemy Repositories
Data-access layer for Users, RefreshTokens, APIKeys, and AuditLogs.
All methods use SQLAlchemy 2.0 async ORM style.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import structlog
from sqlalchemy import ARRAY, Boolean, DateTime, ForeignKey, String, Text, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

logger = structlog.get_logger("luminark.repositories")


# ─────────────────────────────────────────────────────────────────────────────
#  ORM Models
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id:              Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    username:        Mapped[str]       = mapped_column(String(64),  unique=True, nullable=False, index=True)
    email:           Mapped[str]       = mapped_column(String(256), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str]       = mapped_column(String(256), nullable=False)
    scopes:          Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    is_active:       Mapped[bool]      = mapped_column(Boolean, default=True,  nullable=False)
    is_superuser:    Mapped[bool]      = mapped_column(Boolean, default=False, nullable=False)
    created_at:      Mapped[datetime]  = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    api_keys       = relationship("APIKey",        back_populates="user", cascade="all, delete-orphan")
    audit_logs     = relationship("AuditLog",      back_populates="user")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id:             Mapped[uuid.UUID]           = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id:        Mapped[uuid.UUID]           = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash:     Mapped[str]                 = mapped_column(String(256), unique=True, nullable=False, index=True)
    family_id:      Mapped[uuid.UUID]           = mapped_column(nullable=False, index=True)
    is_revoked:     Mapped[bool]                = mapped_column(Boolean, default=False, nullable=False)
    replaced_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("refresh_tokens.id"), nullable=True)
    created_at:     Mapped[datetime]            = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user = relationship("User", back_populates="refresh_tokens")


class APIKey(Base):
    __tablename__ = "api_keys"

    id:           Mapped[uuid.UUID]           = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id:      Mapped[uuid.UUID]           = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    key_hash:     Mapped[str]                 = mapped_column(String(256), unique=True, nullable=False, index=True)
    key_prefix:   Mapped[str]                 = mapped_column(String(16),  nullable=False)
    name:         Mapped[str]                 = mapped_column(String(128), nullable=False)
    scopes:       Mapped[List[str]]           = mapped_column(ARRAY(String), nullable=False, default=list)
    is_active:    Mapped[bool]                = mapped_column(Boolean, default=True, nullable=False)
    created_at:   Mapped[datetime]            = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    last_used_at: Mapped[Optional[datetime]]  = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at:   Mapped[Optional[datetime]]  = mapped_column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="api_keys")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id:         Mapped[uuid.UUID]           = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id:    Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action:     Mapped[str]                 = mapped_column(String(64),  nullable=False)
    resource:   Mapped[Optional[str]]       = mapped_column(String(128), nullable=True)
    ip_address: Mapped[Optional[str]]       = mapped_column(String(64),  nullable=True)
    user_agent: Mapped[Optional[str]]       = mapped_column(Text, nullable=True)
    request_id: Mapped[Optional[str]]       = mapped_column(String(64),  nullable=True)
    detail:     Mapped[Optional[str]]       = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime]            = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    user = relationship("User", back_populates="audit_logs")


# ─────────────────────────────────────────────────────────────────────────────
#  UserRepository
# ─────────────────────────────────────────────────────────────────────────────

class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_username(self, username: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        username: str,
        email: str,
        hashed_password: str,
        scopes: List[str],
        is_superuser: bool = False,
    ) -> User:
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            scopes=scopes,
            is_superuser=is_superuser,
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def update_last_login(self, user_id: uuid.UUID) -> None:
        await self.session.execute(
            update(User)
            .where(User.id == user_id)
            .values(updated_at=datetime.now(timezone.utc))
        )

    async def list_active(self, limit: int = 100) -> List[User]:
        result = await self.session.execute(
            select(User).where(User.is_active == True).order_by(User.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())


# ─────────────────────────────────────────────────────────────────────────────
#  RefreshTokenRepository — with reuse detection
# ─────────────────────────────────────────────────────────────────────────────

class RefreshTokenRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        user_id: uuid.UUID,
        token_hash: str,
        family_id: uuid.UUID,
        expires_at: datetime,
    ) -> RefreshToken:
        token = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            family_id=family_id,
            expires_at=expires_at,
        )
        self.session.add(token)
        await self.session.flush()
        return token

    async def get_by_hash(self, token_hash: str) -> Optional[RefreshToken]:
        result = await self.session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def rotate(
        self,
        old_token: RefreshToken,
        new_token_hash: str,
        expires_at: datetime,
    ) -> RefreshToken:
        """
        Atomic rotation: marks old token revoked and creates new one in the
        same database flush — either both succeed or neither does.
        """
        new_token = RefreshToken(
            user_id=old_token.user_id,
            token_hash=new_token_hash,
            family_id=old_token.family_id,
            expires_at=expires_at,
        )
        self.session.add(new_token)
        await self.session.flush()  # assign new_token.id

        old_token.is_revoked     = True
        old_token.replaced_by_id = new_token.id
        await self.session.flush()

        return new_token

    async def detect_reuse(self, token_hash: str) -> bool:
        """Returns True if the token exists AND is already revoked — replay attack."""
        result = await self.session.execute(
            select(RefreshToken.is_revoked).where(RefreshToken.token_hash == token_hash)
        )
        row = result.scalar_one_or_none()
        return row is True   # None → token not found → not a reuse (just invalid)

    async def revoke_family(self, family_id: uuid.UUID) -> int:
        """Revoke all tokens in a family. Called on reuse attack or logout-all."""
        result = await self.session.execute(
            update(RefreshToken)
            .where(RefreshToken.family_id == family_id, RefreshToken.is_revoked == False)
            .values(is_revoked=True)
        )
        await self.session.flush()
        return result.rowcount

    async def revoke_one(self, token_hash: str) -> None:
        await self.session.execute(
            update(RefreshToken)
            .where(RefreshToken.token_hash == token_hash)
            .values(is_revoked=True)
        )
        await self.session.flush()

    async def revoke_all_for_user(self, user_id: uuid.UUID) -> int:
        result = await self.session.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.is_revoked == False)
            .values(is_revoked=True)
        )
        await self.session.flush()
        return result.rowcount

    async def purge_expired(self) -> int:
        result = await self.session.execute(
            delete(RefreshToken).where(RefreshToken.expires_at < datetime.now(timezone.utc))
        )
        await self.session.flush()
        return result.rowcount


# ─────────────────────────────────────────────────────────────────────────────
#  APIKeyRepository
# ─────────────────────────────────────────────────────────────────────────────

class APIKeyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        user_id: uuid.UUID,
        key_hash: str,
        key_prefix: str,
        name: str,
        scopes: List[str],
    ) -> APIKey:
        key = APIKey(
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            scopes=scopes,
        )
        self.session.add(key)
        await self.session.flush()
        return key

    async def get_by_hash(self, key_hash: str) -> Optional[APIKey]:
        result = await self.session.execute(
            select(APIKey).where(APIKey.key_hash == key_hash, APIKey.is_active == True)
        )
        return result.scalar_one_or_none()

    async def list_for_user(self, user_id: uuid.UUID) -> List[APIKey]:
        result = await self.session.execute(
            select(APIKey).where(APIKey.user_id == user_id).order_by(APIKey.created_at.desc())
        )
        return list(result.scalars().all())

    async def revoke(self, key_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Revoke a key. Returns False if not found or not owned by user_id."""
        result = await self.session.execute(
            update(APIKey)
            .where(APIKey.id == key_id, APIKey.user_id == user_id)
            .values(is_active=False)
        )
        await self.session.flush()
        return result.rowcount > 0


# ─────────────────────────────────────────────────────────────────────────────
#  AuditRepository
# ─────────────────────────────────────────────────────────────────────────────

class AuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def log(
        self,
        action: str,
        resource: str,
        request,  # starlette Request
        user_id: Optional[uuid.UUID] = None,
        detail: Optional[dict] = None,
    ) -> None:
        import json
        record = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            request_id=getattr(request.state, "request_id", None),
            detail=json.dumps(detail) if detail else None,
        )
        self.session.add(record)
        await self.session.flush()

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
    ) -> List[AuditLog]:
        result = await self.session.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# ─────────────────────────────────────────────────────────────────────────────
#  Hash helpers
# ─────────────────────────────────────────────────────────────────────────────

def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
