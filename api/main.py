"""
LUMINARK OVERWATCH PRIME v2.0 — Production FastAPI Backend

Features:
  • JWT auth with JTI blacklisting via Redis
  • Opaque refresh tokens with rotation + reuse detection
  • Per-user API key management
  • Rate limiting (SlowAPI + Redis)
  • Prometheus metrics (/metrics)
  • Structured logging (structlog)
  • Async PostgreSQL (SQLAlchemy 2.0 + asyncpg)
  • Extended engine: NSDT, CITI, Polyvagal, Cultural, Fractal analysis
  • Concurrent batch processing via asyncio.gather()

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic_settings import BaseSettings, SettingsConfigDict
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from luminark.extended_engine import ExtendedEngine, ExtendedResult
from luminark.guardian import GuardianResult, SAPStage
from luminark.principles import PRINCIPLE_PROFILES, CATEGORIES
from luminark.report import generate_text_report, generate_markdown_report, batch_to_csv

from .auth import router as auth_router
from .middleware import RequestTracingMiddleware
from .repositories import Base
from .schemas import (
    AnalyzeRequest, AnalyzeResponse,
    BatchRequest, BatchResponse, BatchResultItem,
    CurrentUser, ExtendedOut, ViolationOut,
)

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
#  Settings
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url:              str       = "postgresql+asyncpg://luminark:secret@localhost:5432/luminark_db"
    redis_url:                 str       = "redis://localhost:6379/0"
    jwt_secret_key:            str       = "CHANGE_ME_USE_OPENSSL_RAND_HEX_32"
    jwt_algorithm:             str       = "HS256"
    jwt_expire_minutes:        int       = 15
    refresh_token_expire_days: int       = 30
    refresh_reuse_detection:   bool      = True
    api_keys:                  str       = ""
    allowed_origins:           List[str] = ["http://localhost:3000", "http://localhost:8501"]
    rate_limit_default:        str       = "200/minute"
    rate_limit_analyze:        str       = "60/minute"
    rate_limit_token:          str       = "10/minute"
    environment:               str       = "development"
    log_level:                 str       = "INFO"

    @property
    def api_key_set(self) -> set:
        """Comma-separated static API keys as a set for O(1) lookup."""
        if not self.api_keys:
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

def configure_logging(log_level: str, production: bool) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer() if production
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger("luminark.api")


# ─────────────────────────────────────────────────────────────────────────────
#  Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    configure_logging(settings.log_level, settings.environment == "production")
    logger.info("startup", env=settings.environment)

    # Database
    engine = create_async_engine(
        settings.database_url,
        echo=(settings.environment != "production"),
        pool_pre_ping=True,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app.state.db_engine          = engine
    app.state.db_session_factory = session_factory

    # Redis
    if aioredis:
        redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
        app.state.redis = redis_client
    else:
        app.state.redis = None
        logger.warning("redis_unavailable", msg="Rate limiting and token blacklisting disabled")

    # Extended engine (CPU-bound; initialised once, shared across requests)
    app.state.engine = ExtendedEngine()
    logger.info("engine_ready")

    yield  # Application is running

    # Shutdown
    logger.info("shutdown")
    if app.state.redis:
        await app.state.redis.aclose()
    await engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────
#  App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = Settings()

    app = FastAPI(
        title="LUMINARK OVERWATCH PRIME v2.0",
        description=(
            "Production-grade AI ethics auditor. Bio-inspired • Ma'at-audited • "
            "SAP-staged • Extended engine: NSDT, CITI, Polyvagal, Cultural, Fractal."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.state.settings = settings

    # Middleware (outermost first)
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiter
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=settings.redis_url,
        default_limits=[settings.rate_limit_default],
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Prometheus metrics
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    # Routers
    app.include_router(auth_router)

    return app


app = create_app()


# ─────────────────────────────────────────────────────────────────────────────
#  Engine dispatch
# ─────────────────────────────────────────────────────────────────────────────

async def run_engine_async(
    extended_engine: ExtendedEngine,
    req: AnalyzeRequest,
) -> tuple[GuardianResult, Optional[ExtendedResult]]:
    """
    CPU-bound engine runs in a thread pool to avoid blocking the event loop.
    Returns (GuardianResult, ExtendedResult | None).
    """
    if req.has_extended:
        base, ext = await asyncio.to_thread(
            extended_engine.analyze_extended,
            req.text,
            req.nsdt,
            req.trapscore,
            req.cultural,
            req.hrv,
            req.stageHistory,
            req.domain,
            req.pvManual,
            req.citiStages,
            req.fractalDepth,
        )
        return base, ext
    else:
        base = await asyncio.to_thread(extended_engine.analyze, req.text)
        return base, None


def _build_response(
    base: GuardianResult,
    ext: Optional[ExtendedResult],
    request_id: str,
) -> AnalyzeResponse:
    d = base.to_dict()
    return AnalyzeResponse(
        input_id=d["input_id"],
        timestamp=d["timestamp"],
        request_id=request_id,
        badge=d["badge"],
        alignment_score=d["alignment_score"],
        threat_score=d["threat_score"],
        confidence=d["confidence"],
        stage=d["stage"],
        stage_label=d["stage_label"],
        stage_description=d["stage_description"],
        defense_mode=d["defense_mode"],
        containment=d["containment"],
        camouflage=d["camouflage"],
        violation_count=d["violation_count"],
        violations=[ViolationOut(**v) for v in d["violations"]],
        category_breakdown=d["category_breakdown"],
        rewrite=d["rewrite"],
        rewrite_applied=d["rewrite_applied"],
        changes_made=d["changes_made"],
        recommendations=d["recommendations"],
        extended=ExtendedOut(**ext.to_dict()) if ext else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Core routes
# ─────────────────────────────────────────────────────────────────────────────

from .auth import require_scope  # noqa: E402 (post-factory import to avoid circularity)


@app.post("/analyze", tags=["core"], summary="Analyze a single text")
async def analyze(
    req: AnalyzeRequest,
    request: Request,
    current_user: CurrentUser = Depends(require_scope("analyze")),
):
    """
    Full ethical audit. Requires Bearer token or X-API-Key with `analyze` scope.

    Supply any combination of extended fields (nsdt, hrv, citiStages, domain,
    cultural, fractalDepth) for augmented multi-layer analysis.
    """
    base, ext   = await run_engine_async(request.app.state.engine, req)
    request_id  = getattr(request.state, "request_id", str(uuid.uuid4()))

    fmt = req.format.lower()
    if fmt == "text":
        return PlainTextResponse(generate_text_report(base))
    if fmt == "markdown":
        return PlainTextResponse(generate_markdown_report(base), media_type="text/markdown")

    return _build_response(base, ext, request_id)


@app.post("/batch", response_model=BatchResponse, tags=["core"],
          summary="Concurrent batch analysis")
async def batch_analyze(
    req: BatchRequest,
    request: Request,
    current_user: CurrentUser = Depends(require_scope("analyze")),
):
    """Up to 100 texts analyzed concurrently. Returns badge distribution + per-item summaries."""
    engine  = request.app.state.engine
    tasks   = [run_engine_async(engine, AnalyzeRequest(text=t)) for t in req.texts]
    results = await asyncio.gather(*tasks)

    badge_counts: dict = {"PASS": 0, "CAUTION": 0, "FAIL": 0, "CRITICAL": 0}
    items: List[BatchResultItem] = []

    for i, (base, _) in enumerate(results):
        badge_counts[base.badge] = badge_counts.get(base.badge, 0) + 1
        items.append(BatchResultItem(
            index=i,
            input_preview=base.input_text[:120],
            badge=base.badge,
            alignment_score=round(base.alignment_score, 1),
            threat_score=round(base.threat_score, 3),
            violation_count=base.violation_count,
            stage=base.stage.name,
            rewrite_applied=base.rewrite_applied,
        ))

    return BatchResponse(
        total=len(results),
        pass_count=badge_counts.get("PASS", 0),
        caution_count=badge_counts.get("CAUTION", 0),
        fail_count=badge_counts.get("FAIL", 0),
        critical_count=badge_counts.get("CRITICAL", 0),
        results=items,
    )


@app.post("/batch/csv", tags=["core"], response_class=Response,
          summary="Batch analysis — CSV download")
async def batch_csv(
    req: BatchRequest,
    request: Request,
    current_user: CurrentUser = Depends(require_scope("analyze")),
):
    engine  = request.app.state.engine
    tasks   = [run_engine_async(engine, AnalyzeRequest(text=t)) for t in req.texts]
    pairs   = await asyncio.gather(*tasks)
    bases   = [base for base, _ in pairs]
    return Response(
        content=batch_to_csv(bases),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=luminark_audit.csv"},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Reference routes (public)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "LUMINARK OVERWATCH PRIME",
        "version": "2.0.0",
        "status":  "online",
        "docs":    "/docs",
        "metrics": "/metrics",
    }


@app.get("/health", tags=["meta"])
async def health(request: Request):
    checks: dict = {"api": "ok"}
    if request.app.state.redis:
        try:
            await request.app.state.redis.ping()
            checks["redis"] = "ok"
        except Exception:
            checks["redis"] = "unavailable"
    return {"status": "healthy", "checks": checks}


@app.get("/principles", tags=["reference"])
def list_principles():
    return {
        "count":      len(PRINCIPLE_PROFILES),
        "categories": CATEGORIES,
        "principles": [
            {
                "id":         p.principle.value,
                "name":       p.principle.name,
                "label":      p.label,
                "ai_meaning": p.ai_meaning,
                "severity":   p.severity,
                "category":   p.category,
            }
            for p in PRINCIPLE_PROFILES
        ],
    }


@app.get("/stages", tags=["reference"])
def list_stages():
    return {
        "stages": [
            {
                "value":       s.value,
                "name":        s.name,
                "label":       s.label,
                "description": s.description,
                "is_danger":   s.is_danger_zone,
                "multiplier":  s.risk_multiplier,
            }
            for s in SAPStage
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Exception handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error":      exc.detail,
            "request_id": getattr(request.state, "request_id", None),
        },
        headers=getattr(exc, "headers", None) or {},
    )


@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error":      "validation_error",
            "detail":     str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error":      "internal_server_error",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point  (pyproject.toml: luminark-api = "api.main:run")
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None,  # structlog handles all logging
    )


if __name__ == "__main__":
    run()
