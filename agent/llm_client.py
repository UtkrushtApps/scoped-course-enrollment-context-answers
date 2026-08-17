from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence

from openai import OpenAI

from agent.config import Settings


@dataclass(frozen=True)
class ModelDecision:
    answer: str
    proposed_course_id: str | None


class ModelClient(Protocol):
    def complete(self, messages: Sequence[dict[str, str]]) -> ModelDecision:
        """Return advice and any proposed course action."""
        ...


class OpenAIModelClient:
    """Real OpenAI-compatible model client used by the agent CLI."""

    def __init__(self, settings: Settings) -> None:
        if not settings.api_key:
            raise RuntimeError("A provider key is required for model access")
        kwargs: dict[str, str] = {"api_key": settings.api_key}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._client = OpenAI(**kwargs)
        self._model = settings.model

    def complete(self, messages: Sequence[dict[str, str]]) -> ModelDecision:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=list(messages),
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("model returned an empty response")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("model returned invalid JSON") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("model response must be a JSON object")

        answer = payload.get("answer")
        proposed = payload.get("proposed_course_id")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("model response has no usable answer")
        if proposed is not None and not isinstance(proposed, str):
            raise RuntimeError("model proposed course must be text or null")

        normalized_proposal = proposed.strip() if isinstance(proposed, str) else None
        if normalized_proposal == "":
            normalized_proposal = None

        return ModelDecision(
            answer=answer.strip(),
            proposed_course_id=normalized_proposal,
        )
