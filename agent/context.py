from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock

from agent.llm_client import ModelClient
from agent.memory import PendingActionStore, SessionScope
from agent.prompts import build_messages
from agent.retrieval import CatalogIndex, RetrievedDocument


@dataclass(frozen=True)
class AdviceRequest:
    tenant_id: str
    student_id: str
    session_id: str
    permissions: frozenset[str]
    message: str

    @property
    def scope(self) -> SessionScope:
        return SessionScope(
            tenant_id=self.tenant_id,
            student_id=self.student_id,
            session_id=self.session_id,
        )


@dataclass(frozen=True)
class AdviceResponse:
    answer: str
    citations: tuple[str, ...]
    needs_confirmation: bool = False
    enrolled_course_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "citations": list(self.citations),
            "needs_confirmation": self.needs_confirmation,
            "enrolled_course_id": self.enrolled_course_id,
        }


class EnrollmentLedger:
    def __init__(self) -> None:
        self._rows: set[tuple[str, str, str]] = set()
        self._lock = RLock()

    def enroll(self, tenant_id: str, student_id: str, course_id: str) -> None:
        with self._lock:
            self._rows.add((tenant_id, student_id, course_id))

    def has(self, tenant_id: str, student_id: str, course_id: str) -> bool:
        with self._lock:
            return (tenant_id, student_id, course_id) in self._rows

    def count(self) -> int:
        with self._lock:
            return len(self._rows)


class CourseAdvisor:
    def __init__(
        self,
        index: CatalogIndex,
        memory: PendingActionStore,
        ledger: EnrollmentLedger,
        model: ModelClient,
        context_token_budget: int,
    ) -> None:
        if context_token_budget <= 0:
            raise ValueError("context token budget must be positive")
        self._index = index
        self._memory = memory
        self._ledger = ledger
        self._model = model
        self._context_token_budget = context_token_budget

    def gather_evidence(self, request: AdviceRequest) -> list[RetrievedDocument]:
        return self._index.select(
            query=request.message,
            tenant_id=request.tenant_id,
            permissions=request.permissions,
            token_budget=self._context_token_budget,
        )

    @staticmethod
    def _citation_ids(evidence: list[RetrievedDocument]) -> tuple[str, ...]:
        return tuple(item.source_id for item in evidence)

    @staticmethod
    def _canonical_course_id(value: str) -> str:
        return "-".join(value.strip().upper().replace("-", " ").split())

    @classmethod
    def _evidenced_course_id(
        cls,
        proposed_course_id: str,
        evidence: list[RetrievedDocument],
    ) -> str | None:
        proposed = cls._canonical_course_id(proposed_course_id)
        for document in evidence:
            if cls._canonical_course_id(document.course_id) == proposed:
                return document.course_id
        return None

    @staticmethod
    def _is_direct_enrollment_request(message: str) -> bool:
        normalized = " ".join(message.casefold().split())
        if not any(term in normalized for term in ("enroll", "register", "add")):
            return False
        if not any(
            marker in normalized
            for marker in (" i ", " me", "my ", "please", "can i", "may i")
        ):
            return False
        hypothetical_markers = (
            "what if",
            "hypothetically",
            "suppose",
            "if i were",
            "would i",
            "could i someday",
            "for example",
        )
        return not any(marker in normalized for marker in hypothetical_markers)

    def handle(
        self,
        request: AdviceRequest,
        now: datetime | None = None,
    ) -> AdviceResponse:
        instant = now or datetime.now(timezone.utc)

        # Confirmation resolution is deterministic and occurs before any model
        # call. Thus a model response in this turn can never authorize a write.
        authorized = self._memory.resolve_response(
            request.scope,
            request.message,
            instant,
        )
        if authorized:
            self._ledger.enroll(
                request.tenant_id,
                request.student_id,
                authorized.course_id,
            )
            return AdviceResponse(
                answer=f"You are enrolled in {authorized.course_id}.",
                citations=(),
                enrolled_course_id=authorized.course_id,
            )

        pending = self._memory.get(request.scope, instant)
        evidence = self.gather_evidence(request)
        messages = build_messages(
            user_message=request.message,
            evidence=evidence,
            pending_course=pending.course_id if pending else None,
        )
        decision = self._model.complete(messages)

        if decision.proposed_course_id and self._is_direct_enrollment_request(
            request.message
        ):
            # Never persist a hallucinated or unauthorized course. A proposal is
            # valid only when that course appeared in the filtered, budgeted
            # evidence supplied to the model for this request.
            evidenced_course = self._evidenced_course_id(
                decision.proposed_course_id,
                evidence,
            )
            if evidenced_course is not None:
                proposal = self._memory.remember_proposal(
                    request.scope,
                    evidenced_course,
                    instant,
                )
                return AdviceResponse(
                    answer=(
                        f"Please confirm whether you want to enroll in "
                        f"{proposal.course_id}."
                    ),
                    citations=self._citation_ids(evidence),
                    needs_confirmation=True,
                )

        return AdviceResponse(
            answer=decision.answer,
            citations=self._citation_ids(evidence),
        )
