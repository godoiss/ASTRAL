"""FastAPI application factory for local scientific serving."""

from __future__ import annotations

import logging
import time
import uuid

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.api.config import ApiSettings
from src.api.errors import ApiError
from src.api.routers import catalog, discovery, experiments, health, science
from src.api.schemas import ErrorResponse


LOGGER = logging.getLogger("astral.api")


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Create a deterministic local-only API without scientific side effects."""
    settings = settings or ApiSettings.from_environment()
    app = FastAPI(
        title=settings.title,
        version=settings.version,
        summary="Scientific services for the ASTRAL Atlas product interface",
        description=(
            "ASTRAL scientific API. It exposes persisted Kepler/TESS artifacts, "
            "trained ANN-01/ANN-02 inference, and cache-only Discovery Engine campaigns. "
            "It never trains models or downloads observations through HTTP."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        responses={
            400: {"model": ErrorResponse, "description": "Malformed or disallowed request"},
            403: {"model": ErrorResponse, "description": "Operation disabled for this deployment"},
            404: {"model": ErrorResponse, "description": "Resource is unavailable or inaccessible"},
            409: {"model": ErrorResponse, "description": "Ambiguous scientific identifier"},
            422: {"model": ErrorResponse, "description": "Request validation failed"},
            500: {"model": ErrorResponse, "description": "Unexpected local API error"},
            503: {"model": ErrorResponse, "description": "Persisted scientific artifact is unavailable"},
        },
    )
    app.state.settings = settings
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{1000 * (time.perf_counter() - started):.3f}"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request parameters or body are invalid.",
                    "details": details,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "route_not_found" if exc.status_code == 404 else "http_error"
        message = "The requested API route does not exist." if exc.status_code == 404 else str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "details": None,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled API error", extra={"request_id": getattr(request.state, "request_id", None)})
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected local API error occurred.",
                    "details": None,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    app.include_router(health.router)
    app.include_router(catalog.router)
    app.include_router(science.router)
    app.include_router(experiments.router)
    app.include_router(discovery.router)

    interface_directory = Path(__file__).resolve().parents[2] / "design" / "stage09" / "preview"

    @app.get("/", include_in_schema=False)
    def product_interface() -> RedirectResponse:
        return RedirectResponse(url="/app/")

    app.mount("/app", StaticFiles(directory=interface_directory, html=True), name="atlas-interface")
    return app


app = create_app()
