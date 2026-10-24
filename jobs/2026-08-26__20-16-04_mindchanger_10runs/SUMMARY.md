# MIND_CHANGER — 10-task batch summary

**Job:** `2026-08-26__20-16-04_mindchanger_10runs`
**Config:** OpenCode agent (`gemini-2.5-flash`), COLLABORATIVE mode, `max_turns=10`,
persona `MIND_CHANGER` (refined, scope-locked prompt), sim model `gemini-3.6-flash`,
validation model `gemini-2.5-flash-lite`.
**Dataset:** the same 10 SWE-bench-verified tasks used in the EXPERT_NITPICKER batch
(`2026-08-26__20-15-44_expert_nitpicker_10runs`), enabling a direct persona comparison.

## Result

**3 / 10 resolved (30%)** — identical overall rate to EXPERT_NITPICKER, but not the same tasks.

| Task | Resolved | Reward | FAIL_TO_PASS | PASS_TO_PASS regressions | Turns |
|---|---|---|---|---|---|
| django-16938 | ✅ | 1 | 23/23 | 0 | 3 |
| sympy-16886 | ✅ | 1 | 1/1 | 0 | 1 |
| requests-1142 | ✅* | 1 | 1/1 | 0 | n/a* |
| django-16032 | ❌ | 0 | 2/2 | **17** | 5 |
| matplotlib-22865 | ❌ | 0 | 1/3 | 1 | 2 |
| django-14311 | ❌ | 0 | 1/1 | 3 | 3 |
| django-16950 | ❌ | 0 | 0/1 | 0 | 4 |
| django-10554 | ❌ | 0 | 0/2 | 1 | 3 |
| matplotlib-14623 | ❌ | 0 | 0/1 | 0 | 3 |
| pytest-7236 | ❌ | 0 | 0/1 | 0 | 3 |

\* `requests-1142` crashed with an `ApiRateLimitError` mid-session (same failure mode as in the
nitpicker batch) — no clean `[DONE]`, but the code at the moment of the crash was already
correct. Notably the mind-change itself happened *before* the crash: turn 1 proposed an opt-in
`omit_content_length` parameter; turn 2 reversed course — *"Actually, thinking about it again,
let's not add a new parameter... automatically omit... for GET and HEAD requests"* — a genuine
behavior-level reversal, not just an implementation tweak.

## What this batch tells us

### 1. Persona fidelity is substantially better than the original single-task run
In the earlier astropy-12907 experiment, MIND_CHANGER never actually changed the goal — it acted
as an increasingly precise bug reporter and handed the agent the exact fix. That does not
generalize here: **8 of the 10 non-trivial trials contain an explicit, unprompted reversal**,
consistently phrased as "Wait, looking at X...", "Actually, thinking about this again...", or
"Let's completely revert / change course / redirect our approach." Examples:
- django-16938: reverses direction twice — first from a `serializers.py`-local fix to a
  `ManyRelatedManager`-level fix, then reverses *that* too because it breaks explicit
  prefetching, landing on a third approach.
- django-16032: "you just re-introduced the exact same subquery hack under a different name...
  let's change course entirely."
- matplotlib-14623: pivots away from the exact file/function the issue names ("`scale.py` feels
  like the wrong layer") to a different layer of the axis-limit logic.

The one trial with no reversal is **sympy-16886** — a single-line dict-value fix — where the
persona correctly recognized there was nothing to redirect and just double-checked the edit was
applied, rather than manufacturing a pointless pivot. That's the right behavior for a trivial
task and suggests the persona isn't forcing reversals indiscriminately.

Most reversals are still *implementation-strategy* pivots (different file/layer/approach for the
same underlying requirement) rather than the stricter *requirement-level* reversal the persona
spec describes (e.g., "hide the command" → "remove it entirely"). `requests-1142`'s
opt-in-parameter → automatic-behavior-change reversal is the closest example in this batch to a
genuine change in *what* gets built rather than just *how*.

### 2. Same overall pass rate as EXPERT_NITPICKER, but a different profile — and a shared floor
Comparing directly against the nitpicker batch on the identical 10 tasks:

- **4 tasks fail under both personas** (django-10554, django-14311, matplotlib-14623,
  pytest-7236), and in near-identical ways — e.g. django-14311 breaks the exact same three
  sibling tests (`test_run_as_module`, `test_run_as_non_django_module`, `test_python_m_django`)
  regardless of which persona drove the session. This is strong evidence these four are a
  genuine capability ceiling for `gemini-2.5-flash` on this agent/task combination, not an
  artifact of persona behavior.
- **2 tasks flip outcome depending on persona:**
  - **django-16938**: nitpicker's single scoped correction generalized the fix from one
    serializer file to a second, but missed a third (XML) — 18/23 F2P. MIND_CHANGER's repeated
    reconsideration pushed the agent through three different approaches before landing on one
    that generalized fully — 23/23 F2P, 0 regressions. Here, the extra churn *helped*.
  - **matplotlib-22865**: nitpicker made small, scoped, correctness-preserving nitpicks to an
    already-correct fix and passed cleanly (3/3 F2P, 0 regressions). MIND_CHANGER instead
    reversed away from a working narrow fix — *"slicing `segments` based on `self.extend`
    directly... isn't the right approach... revert that and instead update..."* — toward a more
    general reimplementation that turned out incomplete (1/3 F2P) and introduced a regression.
    Here, the churn *hurt*: reversing a solution that was already on track cost the trial.

### 3. Thrashing has a real regression cost, even when it reaches the target fix
**django-16032** is the sharpest example: five rounds of substantive strategy reversal
eventually got both target `FAIL_TO_PASS` tests passing — but broke **17** previously-passing
tests along the way, by far the largest blast radius in either batch. Compare to the same task
under EXPERT_NITPICKER, which never got the bug fixed at all (0/2 F2P) but only broke 1 test.
Repeatedly ripping out and replacing an in-progress implementation (custom method → revert →
another custom method → revert again → direct mask mutation) left the codebase touched in more
places than a single coherent approach would have, and evidently not all of the intermediate
reversals were fully clean.

### 4. Net comparison for the paper
Both personas resolve 3/10 on this task set, but they are not simply "equally good" — they fail
differently. EXPERT_NITPICKER's failures are dominated by the agent simply not landing the fix
(4 of 7 failures have 0 F2P successes) with minimal collateral damage, consistent with a
scope-locked persona that mostly leaves working code alone. MIND_CHANGER's failures more often
involve real collateral damage from strategy churn (django-16032's 17 regressions;
matplotlib-22865's regression on top of an incomplete rewrite), consistent with a persona whose
job is to force the agent to reconsider and re-implement mid-task. The two personas appear to be
stressing genuinely different agent capabilities — tolerance for pointed correction vs. stability
under repeated redirection — which is a reasonable point to make in the paper about why multiple
personas are useful rather than redundant.
