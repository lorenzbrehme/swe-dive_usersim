# EXPERT_NITPICKER — 10-task batch summary

**Job:** `2026-08-26__20-15-44_expert_nitpicker_10runs`
**Config:** OpenCode agent (`gemini-2.5-flash`), COLLABORATIVE mode, `max_turns=10`,
persona `EXPERT_NITPICKER` (refined, scope-locked prompt), sim model `gemini-3.6-flash`,
validation model `gemini-2.5-flash-lite`.
**Dataset:** 10 randomly sampled SWE-bench-verified tasks.

## Result

**3 / 10 resolved (30%).**

| Task | Resolved | Reward | FAIL_TO_PASS | PASS_TO_PASS regressions | Turns |
|---|---|---|---|---|---|
| matplotlib-22865 | ✅ | 1 | 3/3 | 0 | 3 |
| sympy-16886 | ✅ | 1 | 1/1 | 0 | 2 |
| requests-1142 | ✅* | 1 | 1/1 | 0 | n/a* |
| django-16938 | ❌ | 0 | 18/23 | 0 | 2 |
| django-10554 | ❌ | 0 | 0/2 | 1 | 6 |
| django-14311 | ❌ | 0 | 1/1 | 3 | 2 |
| django-16032 | ❌ | 0 | 0/2 | 1 | 3 |
| django-16950 | ❌ | 0 | 1/1 | 3 (all) | 3 |
| matplotlib-14623 | ❌ | 0 | 0/1 | 0 | 4 |
| pytest-7236 | ❌ | 0 | 0/1 | 1 | 5 |

\* `requests-1142` crashed with an `ApiRateLimitError` mid-session (Gemini rate limit hit at
turn 2) — the harness never got a clean `[DONE]`, but the verifier ran against the git state
at the moment of the crash and it happened to already be correct. Treat this as a pass with
an asterisk, not a clean simulator completion.

Turn counts above are follow-up turns (`sim_response` entries), not counting the final `[DONE]`
message.

## What this batch tells us

### 1. The "correct fix nitpicked into a regression" pattern is gone
This was the dominant failure mode in the earlier single-task (astropy-12907) experiments before
the persona prompt was scope-locked. It does **not** reappear here: in every failing trial, the
nitpicker's corrections are genuine, targeted fixes to real problems visible in the diff
(duplicate method definitions, wrong indentation, a condition that silently breaks siblings), not
speculative "make it more robust" rewrites of already-working code. The rule change (correct only
within the lines just touched, stop once nothing concrete is left to point at) appears to hold up
across a more diverse task set, not just the one task it was tuned against.

### 2. Two failures are agent capability limits, not persona misbehavior
- **django-16032** (0/2 F2P) and **matplotlib-14623** (0/1 F2P) and **pytest-7236** (0/1 F2P):
  the agent never landed the actual fix despite multiple in-scope, legitimate nitpicks. In
  pytest-7236 the nitpicker spent 4 of 5 turns just getting the agent to stop reintroducing
  naming/indentation errors (`_explicit_tearDown_func` → `explicit_teardown`, then fixing its
  indentation) — turns that in a well-functioning agent would not have been needed at all.
- **django-16938** (18/23 F2P): the fix correctly generalized from `python.py` to `base.py` (per
  the nitpicker's turn-1 correction) but missed a *third* serializer implementation (XML) that
  needed the identical treatment — an incomplete generalization neither the agent nor the persona
  noticed, since neither runs the XML-specific test subset.

### 3. django-10554: the agent, not the persona, is the bottleneck
This trial burned 6 of 10 turns on the nitpicker repeatedly catching the same class of defect —
duplicate `clear_ordering` method definitions / syntax errors reintroduced by the agent's own
edits (turns 4 and 5 both catch this). The corrections themselves are legitimate and scoped, but
by the time the code was even syntactically valid, the turn budget was mostly gone and the
underlying ordering-position bug was never actually fixed (0/2 F2P).

### 4. A genuine blind spot: fixes that overfit the named test and break siblings
**django-14311** and **django-16950** are the most interesting failures for the paper: in both
cases, `FAIL_TO_PASS` fully succeeded (the agent fixed exactly the case the issue described) and
the nitpicker made a legitimate, scoped correction and then correctly said `[DONE]` — yet
`PASS_TO_PASS` regressed on **sibling** tests in the *same* test class that were never part of
the conversation:
- django-14311: fixing `test_run_as_non_django_module_non_package` by switching a check from
  `__main__.__spec__.parent` to `__main__.__spec__.name` broke `test_run_as_module`,
  `test_run_as_non_django_module`, and `test_python_m_django` — closely related cases in the same
  `TestChildArguments` class.
- django-16950: fixing the target case broke all 3 other `PASS_TO_PASS` tests in the same
  inline-formset area — the largest proportional regression in the batch.

Neither the agent nor the EXPERT_NITPICKER persona has any way to see this: the persona only
reacts to the diff and the agent's own message, and (by design, per earlier discussion) nothing
in this setup executes the test suite. A human expert reviewer doing code review from a diff
alone could plausibly miss the same thing — but it means this failure mode is structural to the
current workflow, not something the persona prompt can fix on its own.

### 5. The three clean passes look like the intended behavior
matplotlib-22865 and sympy-16886 show the nitpicker working as designed: real, scoped,
correctness-preserving corrections (avoid a dead `else` branch, deduplicate a redundant
`np.dstack` call, double-check for a mirrored inverse-mapping dict) applied to a fix that was
already right, with a prompt `[DONE]` once nothing concrete was left to flag.

## Bottom line for the paper

Scope-locking the nitpicker's follow-up behavior eliminated the specific failure mode it was
designed to fix (over-refinement regressing an already-correct patch) — it does not reappear
once across 10 diverse tasks. But the overall resolve rate (30%) shows two other, larger
bottlenecks that are independent of persona design: (a) the coding agent's own reliability on
harder multi-file bugs (unforced edit errors, incomplete generalization), and (b) a structural
blind spot where a fix that passes the named test can silently break sibling tests the simulator
never inspects. Both would need agent-side or verification-side changes (not persona-prompt
changes) to address.
