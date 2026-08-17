from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tiktoken


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CatalogDocument:
    source_id: str
    tenant_id: str | None
    permissions: frozenset[str]
    course_id: str
    title: str
    updated_at: str
    text: str

    def render(self) -> str:
        return (
            f"[{self.source_id}] course={self.course_id}; title={self.title}; "
            f"updated={self.updated_at}\n{self.text}"
        )


@dataclass(frozen=True)
class SearchCandidate:
    document: CatalogDocument
    score: float


RetrievedDocument = CatalogDocument


class CatalogIndex:
    def __init__(self, documents: Iterable[CatalogDocument]) -> None:
        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("catalog index requires at least one document")

        source_ids = [item.source_id for item in self.documents]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("catalog source identifiers must be unique")

        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._document_terms = {
            item.source_id: Counter(self._terms(item.render())) for item in self.documents
        }
        self._document_frequency = self._build_document_frequency()

    @classmethod
    def from_jsonl(cls, path: Path) -> "CatalogIndex":
        documents: list[CatalogDocument] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON catalog fixture at line {line_number}"
                    ) from exc
                if payload.get("record_type") != "catalog_document":
                    continue
                try:
                    documents.append(
                        CatalogDocument(
                            source_id=payload["source_id"],
                            tenant_id=payload.get("tenant_id"),
                            permissions=frozenset(payload["permissions"]),
                            course_id=payload["course_id"],
                            title=payload["title"],
                            updated_at=payload["updated_at"],
                            text=payload["text"],
                        )
                    )
                except (KeyError, TypeError) as exc:
                    raise ValueError(
                        f"invalid catalog fixture at line {line_number}"
                    ) from exc
        return cls(documents)

    @staticmethod
    def _terms(value: str) -> list[str]:
        return _TOKEN_PATTERN.findall(value.lower())

    def _build_document_frequency(self) -> Counter[str]:
        frequency: Counter[str] = Counter()
        for terms in self._document_terms.values():
            frequency.update(terms.keys())
        return frequency

    def count_tokens(self, value: str) -> int:
        return len(self._encoding.encode(value))

    def search(self, query: str, limit: int = 12) -> list[SearchCandidate]:
        if limit <= 0:
            return []

        query_terms = Counter(self._terms(query))
        candidates: list[SearchCandidate] = []
        document_count = len(self.documents)

        for document in self.documents:
            terms = self._document_terms[document.source_id]
            score = 0.0
            for term, query_frequency in query_terms.items():
                if term not in terms:
                    continue
                inverse_frequency = math.log(
                    1 + document_count / (1 + self._document_frequency[term])
                )
                score += min(terms[term], query_frequency) * inverse_frequency
            candidates.append(SearchCandidate(document=document, score=score))

        candidates.sort(key=lambda item: (-item.score, item.document.source_id))
        return candidates[:limit]

    def select(
        self,
        query: str,
        tenant_id: str,
        permissions: frozenset[str],
        token_budget: int,
    ) -> list[RetrievedDocument]:
        """Select relevant, authorized documents without exceeding the budget.

        Tenant filtering and permission filtering happen before a document is
        admitted to prompt context. Global documents have a null tenant and are
        visible to every tenant, subject to their permission ACL.
        """
        if token_budget <= 0 or not permissions:
            return []

        selected: list[RetrievedDocument] = []
        used_tokens = 0

        for candidate in self.search(query):
            document = candidate.document

            # A zero-score record is not evidence for this request, even if it
            # would otherwise pass the access checks.
            if candidate.score <= 0:
                continue
            if document.tenant_id not in {None, tenant_id}:
                continue
            if document.permissions.isdisjoint(permissions):
                continue

            document_tokens = self.count_tokens(document.render())
            if used_tokens + document_tokens > token_budget:
                # Skip oversized candidates rather than stopping, because a
                # later relevant document may still fit in the remaining room.
                continue

            selected.append(document)
            used_tokens += document_tokens

        return selected
