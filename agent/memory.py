from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock


_COURSE_ID_PATTERN = re.compile(r"\b[a-z]{2,}(?:\s*[- ]\s*)\d+[a-z]?\b", re.IGNORECASE)
_NON_WORD_PATTERN = re.compile(r"[^a-z0-9-]+")


@dataclass(frozen=True)
class SessionScope:
    tenant_id: str
    student_id: str
    session_id: str


@dataclass(frozen=True)
class PendingAction:
    course_id: str
    created_at: datetime
    expires_at: datetime


class PendingActionStore:
    def __init__(self, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("pending action lifetime must be positive")
        self._ttl = timedelta(seconds=ttl_seconds)
        self._items: dict[SessionScope, PendingAction] = {}
        self._lock = RLock()

    @staticmethod
    def _instant(now: datetime | None) -> datetime:
        instant = now or datetime.now(timezone.utc)
        if instant.tzinfo is None:
            return instant.replace(tzinfo=timezone.utc)
        return instant

    @staticmethod
    def _normalize_message(value: str) -> str:
        return " ".join(_NON_WORD_PATTERN.sub(" ", value.casefold()).split())

    @staticmethod
    def _normalize_course_id(value: str) -> str:
        return re.sub(r"[\s-]+", "-", value.strip()).upper()

    @classmethod
    def _mentioned_courses(cls, value: str) -> set[str]:
        return {
            cls._normalize_course_id(match.group(0))
            for match in _COURSE_ID_PATTERN.finditer(value)
        }

    @classmethod
    def _is_clear_confirmation(cls, user_message: str, course_id: str) -> bool:
        # Questions and conditional/hypothetical language are not confirmations.
        if "?" in user_message:
            return False

        normalized = cls._normalize_message(user_message)
        hypothetical_markers = (
            "what if",
            "hypothetically",
            "suppose",
            "if i",
            "would i",
            "could i",
            "maybe",
            "not sure",
        )
        if any(marker in normalized for marker in hypothetical_markers):
            return False

        # If the response names a course, it must name only the pending course.
        mentioned = cls._mentioned_courses(user_message)
        expected = cls._normalize_course_id(course_id)
        if mentioned and mentioned != {expected}:
            return False

        course_suffix = r"(?: in [a-z]{2,}(?: |-)?\d+[a-z]?)?"
        affirmative_patterns = (
            r"yes",
            r"yes please",
            r"yes enroll me" + course_suffix,
            r"yes please enroll me" + course_suffix,
            r"please enroll me" + course_suffix,
            r"enroll me" + course_suffix,
            r"i confirm",
            r"i confirm enrollment",
            r"i confirm my enrollment",
            r"confirm",
            r"confirm enrollment",
            r"go ahead",
            r"go ahead and enroll me" + course_suffix,
            r"do it",
        )
        return any(re.fullmatch(pattern, normalized) for pattern in affirmative_patterns)

    @classmethod
    def _is_clear_rejection(cls, user_message: str) -> bool:
        normalized = cls._normalize_message(user_message)
        return normalized in {
            "no",
            "no thanks",
            "do not enroll me",
            "don t enroll me",
            "cancel",
            "cancel it",
            "never mind",
            "nevermind",
        }

    def get(
        self,
        scope: SessionScope,
        now: datetime | None = None,
    ) -> PendingAction | None:
        instant = self._instant(now)
        with self._lock:
            item = self._items.get(scope)
            if item and item.expires_at <= instant:
                self._items.pop(scope, None)
                return None
            return item

    def remember_proposal(
        self,
        scope: SessionScope,
        course_id: str,
        now: datetime | None = None,
    ) -> PendingAction:
        """Retain a proposed enrollment only inside its exact scope."""
        normalized_course = course_id.strip()
        if not normalized_course:
            raise ValueError("proposed course identifier must not be empty")

        instant = self._instant(now)
        proposal = PendingAction(
            course_id=normalized_course,
            created_at=instant,
            expires_at=instant + self._ttl,
        )
        with self._lock:
            self._items[scope] = proposal
        return proposal

    def resolve_response(
        self,
        scope: SessionScope,
        user_message: str,
        now: datetime | None = None,
    ) -> PendingAction | None:
        """Atomically consume a pending action only for a clear confirmation.

        The full SessionScope dictionary key prevents a response from another
        tenant, student, or conversation from observing or resolving the item.
        Expired actions are removed before the response is classified.
        """
        instant = self._instant(now)
        with self._lock:
            item = self._items.get(scope)
            if item is None:
                return None
            if item.expires_at <= instant:
                self._items.pop(scope, None)
                return None
            if self._is_clear_rejection(user_message):
                self._items.pop(scope, None)
                return None
            if not self._is_clear_confirmation(user_message, item.course_id):
                return None

            # Consume before returning so retries cannot authorize another write.
            self._items.pop(scope, None)
            return item

    def clear(self, scope: SessionScope) -> None:
        with self._lock:
            self._items.pop(scope, None)
