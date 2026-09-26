"""Durable journal: crash recovery distinguishes committed results from unfinished jobs."""

from __future__ import annotations


class Journal:
    def __init__(self, path: str) -> None:  # pragma: no cover - stub
        self.path = path

    def append_event(self, event: dict) -> None:
        raise NotImplementedError

    def recover(self) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
