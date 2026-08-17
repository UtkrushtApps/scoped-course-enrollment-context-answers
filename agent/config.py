from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path("/root/task")
ENV_PATH = BASE_DIR / ".env"


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    base_url: str | None
    context_token_budget: int
    pending_ttl_seconds: int


def _positive_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def load_settings(require_key: bool = False) -> Settings:
    load_dotenv(ENV_PATH)
    key = os.getenv("OPENAI_API_KEY") or None
    if require_key and not key:
        raise RuntimeError("OPENAI_API_KEY is required for an end-to-end run")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    if not model:
        raise ValueError("OPENAI_MODEL must not be empty")

    return Settings(
        api_key=key,
        model=model,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        context_token_budget=_positive_int("CONTEXT_TOKEN_BUDGET", "420"),
        pending_ttl_seconds=_positive_int("PENDING_TTL_SECONDS", "300"),
    )
