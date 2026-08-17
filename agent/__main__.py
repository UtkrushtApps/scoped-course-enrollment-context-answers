from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.config import load_settings
from agent.context import AdviceRequest, CourseAdvisor, EnrollmentLedger
from agent.llm_client import OpenAIModelClient
from agent.memory import PendingActionStore
from agent.prompts import build_messages, validate_templates
from agent.retrieval import CatalogIndex


BASE_DIR = Path("/root/task")
FIXTURE_PATH = BASE_DIR / "fixtures" / "example.jsonl"


def selfcheck() -> int:
    settings = load_settings()
    validate_templates()
    index = CatalogIndex.from_jsonl(FIXTURE_PATH)
    raw = index.search("CS-210 enrollment prerequisites")
    if not raw:
        raise RuntimeError("fixture index returned no records")

    selected = index.select(
        query="CS-210 enrollment prerequisites",
        tenant_id="north-campus",
        permissions=frozenset({"student"}),
        token_budget=settings.context_token_budget,
    )
    used = sum(index.count_tokens(item.render()) for item in selected)
    if used > settings.context_token_budget:
        raise RuntimeError("selected catalog context exceeds its token budget")
    if any(
        item.tenant_id not in {None, "north-campus"}
        or item.permissions.isdisjoint({"student"})
        for item in selected
    ):
        raise RuntimeError("selected catalog context contains forbidden records")

    preview = build_messages(
        user_message="Can you explain the prerequisites?",
        evidence=[],
        pending_course=None,
    )
    if len(preview) != 2:
        raise RuntimeError("prompt messages are structurally invalid")

    print(f"loaded {len(index.documents)} catalog records")
    print(f"validated model setting: {settings.model}")
    if settings.api_key:
        print("provider key found; model calls remain disabled during selfcheck")
    else:
        print("provider key not found; skipping model access")
    return 0


def inspect_query(query: str, tenant_id: str, permissions: set[str]) -> int:
    settings = load_settings()
    index = CatalogIndex.from_jsonl(FIXTURE_PATH)
    print("Raw retrieval candidates:")
    for candidate in index.search(query):
        document = candidate.document
        print(
            json.dumps(
                {
                    "source_id": document.source_id,
                    "tenant_id": document.tenant_id,
                    "permissions": sorted(document.permissions),
                    "score": round(candidate.score, 3),
                    "tokens": index.count_tokens(document.render()),
                },
                sort_keys=True,
            )
        )

    selected = index.select(
        query=query,
        tenant_id=tenant_id,
        permissions=frozenset(permissions),
        token_budget=settings.context_token_budget,
    )
    print(
        f"request tenant={tenant_id} permissions={sorted(permissions)} "
        f"budget={settings.context_token_budget}"
    )
    print("Selected evidence:")
    for document in selected:
        print(
            json.dumps(
                {
                    "source_id": document.source_id,
                    "tokens": index.count_tokens(document.render()),
                },
                sort_keys=True,
            )
        )
    return 0


def ask(message: str, tenant_id: str, student_id: str, session_id: str) -> int:
    settings = load_settings(require_key=True)
    index = CatalogIndex.from_jsonl(FIXTURE_PATH)
    advisor = CourseAdvisor(
        index=index,
        memory=PendingActionStore(settings.pending_ttl_seconds),
        ledger=EnrollmentLedger(),
        model=OpenAIModelClient(settings),
        context_token_budget=settings.context_token_budget,
    )
    request = AdviceRequest(
        tenant_id=tenant_id,
        student_id=student_id,
        session_id=session_id,
        permissions=frozenset({"student"}),
        message=message,
    )
    response = advisor.handle(request)
    print(json.dumps(response.to_dict(), indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Local course advising agent")
    command.add_argument("--selfcheck", action="store_true")
    command.add_argument("--inspect", metavar="QUERY")
    command.add_argument("--ask", metavar="MESSAGE")
    command.add_argument("--tenant", default="north-campus")
    command.add_argument("--student", default="student-42")
    command.add_argument("--session", default="demo-session")
    return command


def main() -> int:
    args = parser().parse_args()
    if args.selfcheck:
        return selfcheck()
    if args.inspect:
        return inspect_query(args.inspect, args.tenant, {"student"})
    if args.ask:
        return ask(args.ask, args.tenant, args.student, args.session)
    parser().print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
