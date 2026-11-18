"""ASGI entry point for ``uvicorn api:app``."""

from src.api.app import app

__all__ = ["app"]
