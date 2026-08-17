# Solution Steps

1. Implement catalog retrieval as a two-stage pipeline: rank records lexically, then discard zero-score, cross-tenant, and permission-incompatible records before adding any text to model context.

2. Pack permitted records in relevance order while counting each rendered record with tiktoken. Skip records that do not fit and ensure the accumulated evidence never exceeds the configured budget.

3. Keep pending actions in a thread-safe dictionary keyed by the complete tenant, student, and session scope. Store creation and expiration timestamps for each proposal.

4. Resolve confirmations deterministically before calling the model. Remove expired actions, clear explicit rejections, accept only tightly defined affirmative messages, reject questions and hypothetical language, and atomically consume a confirmed action to prevent replay.

5. When an affirmative response names a course, compare it with the pending course so a request for a different course cannot accidentally confirm the old proposal.

6. Build prompts with visibly separated policy, workflow state, catalog evidence, and untrusted user-input layers. Instruct the model that proposals are not writes and that course facts must come from supplied evidence.

7. After a model call, persist a proposal only for a direct enrollment request and only when the proposed course identifier occurs in the already authorized, budgeted evidence. A proposal always requires a later message to complete enrollment.

8. Perform ledger writes only from the pending-action resolution path. Never write directly from a model decision, an expired action, an unrelated session, or the same message that generated a proposal.

9. Validate provider JSON responses, normalize empty proposed identifiers to null, and retain the real OpenAI-compatible path for end-to-end operation.

10. Run `./run.sh` for the readiness probe, then run `pytest -q` for the offline invariant suite. With `OPENAI_API_KEY` configured, use the CLI/provider path or instantiate one persistent `CourseAdvisor` for a two-message proposal-and-confirmation conversation.

