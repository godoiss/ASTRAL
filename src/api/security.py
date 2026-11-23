"""Shared authorization helpers for bounded local scientific mutations."""

from __future__ import annotations

import secrets

from fastapi import Request

from src.api.config import ApiSettings
from src.api.errors import ApiError


LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def authorize_mutation(
    request: Request,
    settings: ApiSettings,
    *,
    feature_enabled: bool,
    feature_name: str,
) -> None:
    """Allow an explicitly enabled mutation from a local client or valid operation key."""
    if not feature_enabled or settings.public_read_only:
        raise ApiError(
            403,
            "operation_disabled",
            f"{feature_name} is disabled for this deployment.",
        )
    if settings.operation_key:
        supplied = request.headers.get("X-ASTRAL-Operation-Key", "")
        if not secrets.compare_digest(supplied, settings.operation_key):
            raise ApiError(403, "operation_not_authorized", "A valid operation key is required.")
        return
    host = request.client.host if request.client else ""
    if host not in LOCAL_CLIENTS:
        raise ApiError(
            403,
            "local_operation_only",
            "This mutation is restricted to an authenticated deployment or local research session.",
        )
