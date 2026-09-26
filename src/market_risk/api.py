"""FastAPI application exposing the engine over HTTP."""

from __future__ import annotations

from fastapi import FastAPI


def create_app(engine=None) -> FastAPI:  # pragma: no cover - stub
    raise NotImplementedError


app: FastAPI | None = None
