# Vague_Requester — 10-task batch summary

**Job:** `2026-08-26__21-10-41_vaguerequester_10turns`
**Config:** OpenCode agent (`gemini-2.5-flash`), COLLABORATIVE mode, `max_turns=10`,
persona `Vague_Requester` (refined, scope-locked prompt), sim model `gemini-3.6-flash`,
validation model `gemini-2.5-flash-lite`.
**Dataset:** the same 10 SWE-bench-verified tasks used in the EXPERT_NITPICKER and
MIND_CHANGER batches (`..._expert_nitpicker_10runs`, `..._mindchanger_10runs`), completing a
3-way persona comparison on identical tasks.

## Result

**3 / 10 resolved (30%)** — same overall rate as the other two personas.

| Task | Resolved | Reward | FAIL_TO_PASS | PASS_TO_PASS regressions | Turns |
|---|---|---|---|---|---|
| django-16938 | ✅ | 1 | 23/23 | 0 | 1 |
| requests-1142 | ✅ | 1 | 1/1 | 0 | 3 |
| sympy-16886 | ✅ | 1 | 1/1 | 0 | 1 |
| django-10554 | ❌ | 0 | 0/2 | 0 | 1 |
| django-14311 | ❌ | 0 | 0/1 | 3 | 2 |
| django-16032 | ❌ | 0 | 0/2 | 0 | 1 |
| django-16950 | ❌ | 0 | 0/1 | 2 | 3 |
| matplotlib-14623 | ❌ | 0 | 0/1 | 0 | 10 (max) |
| matplotlib-22865 | ❌ | 0 | 1/3 | 1 | 3 |
| pytest-7236 | ❌ | 0 | 0/1 | 1 | 2 |

No `ApiRateLimitError` crashes in this batch (unlike the other two, where `requests-1142`
crashed mid-session both times) — all 10 trials completed cleanly through either `[DONE]` or the
turn limit.

## What this batch tells us

### 1. Persona fidelity held up: genuinely high-level throughout, with one clear slip
Almost every message stays at the outcome level — "that's not fixing it," "still not working,"
"looks like that fixed it, commit and push" (matplotlib-22865 turn 2, which almost verbatim
matches the persona's own canonical example) — with no code inspection. The one clear violation
is **requests-1142 turn 1**: *"Returning early from `prepare_body` skips setting `self.body`
entirely. Don't abort the whole function—just avoid setting the Content-Length header..."* — this
names an exact function and instance attribute, which the refined persona explicitly forbids.
It didn't cost the trial (this one still passed), but it shows the sim model (`gemini-3.6-flash`)
still occasionally reverts to "helpful technical reviewer" habits even under the tightened
prompt — just far less often than before the refinement.

### 2. The persona's core weakness: it cannot verify, so it cannot catch a wrong or partial fix
Because Vague_Requester is barred from inspecting code (by design), its only signal is whether
the agent's own narrative sounds finished. This produces two distinct failure patterns:
- **Premature acceptance on a fix that never worked at all**: django-10554, django-16032, and
  pytest-7236 were all accepted with `[DONE]` after just one or two turns, no pushback, despite
  the target bug never actually being fixed (0 `FAIL_TO_PASS` in all three). A nitpicker or
  mind-changer reading the diff would likely have noticed nothing meaningful changed; a vague
  requester, correctly in-character, cannot.
- **Accepting a fix that "looks" done but is incomplete**: matplotlib-22865 turn 2 — *"looks
  like that fixed it, the edges at the ends are showing up now. commit and push"* — after only a
  visual-sounding confirmation. The verifier shows only 1 of 3 `FAIL_TO_PASS` tests actually
  passed, plus a new regression. This mirrors the same failure mode seen in the original
  astropy-12907 experiment: an outcome-only judgment is exactly as unreliable as the agent's own
  self-report.

### 3. matplotlib-14623: the clearest illustration of the persona's ceiling
This is the only trial across all three persona batches that ran the full 10-turn budget without
reaching `[DONE]`. The transcript shows why a non-technical user is particularly badly suited to
this failure: the agent's first "fix" (turn 0) was not a library fix at all — it modified the
*reproduction script* to call `ax.invert_yaxis()` as a workaround, rather than changing
`matplotlib` itself. It took the vague user three full turns of "that doesn't fix the bug... we
are fixing matplotlib itself, not looking for user workarounds... you still haven't committed any
changes" before the agent switched to editing the actual library (`ticker.py`). From there the
agent's own attempts oscillated — one turn introduces an `OverflowError: cannot convert float
infinity to integer` crash in `LogLocator`, and the final turn's response degrades into a raw,
malformed tool-call dump rather than a coherent patch. The user simulator, correctly in-persona,
never had the technical vocabulary to name the crash or steer around it — all it could do was
repeat "still not working," which is realistic for this persona but structurally unable to break
the loop. This is the single worst outcome in any of the three batches (`0/1 FAIL_TO_PASS`,
10 turns burned, no regressions only because nothing coherent shipped).

## Three-way comparison across identical tasks (Nitpicker / Mind-Changer / Vague-Requester)

| Task | Nitpicker | Mind-Changer | Vague-Requester |
|---|---|---|---|
| sympy-16886 | ✅ | ✅ | ✅ |
| requests-1142 | ✅* | ✅* | ✅ |
| django-16938 | ❌ (18/23) | ✅ | ✅ |
| matplotlib-22865 | ✅ | ❌ | ❌ |
| django-10554 | ❌ | ❌ | ❌ |
| django-14311 | ❌ | ❌ | ❌ |
| django-16032 | ❌ | ❌ (17 regressions) | ❌ |
| django-16950 | ❌ | ❌ | ❌ |
| matplotlib-14623 | ❌ | ❌ | ❌ |
| pytest-7236 | ❌ | ❌ | ❌ |

\* crashed on an API rate limit mid-session in those two batches; code was already correct at
the point of the crash.

Across all three personas on the same 10 tasks:
- **2 tasks always pass** (sympy-16886, requests-1142) — simple, well-localized bugs where the
  agent gets it right on the first real attempt regardless of how the user talks to it.
- **5 tasks always fail** (django-10554, django-14311, django-16950, matplotlib-14623,
  pytest-7236) — persona-independent; these look like a genuine capability ceiling for
  `gemini-2.5-flash` on this agent setup, not an artifact of user behavior.
- **Only 2 tasks are persona-sensitive**: django-16938 (fails under the scope-locked nitpicker's
  single narrow correction, passes under both mind-changer's repeated reconsideration and
  vague-requester's one-shot acceptance of an already-comprehensive fix) and matplotlib-22865
  (passes under nitpicker's scoped, correctness-preserving corrections; fails under both
  mind-changer, which abandoned a working narrow fix for an incomplete rewrite, and
  vague-requester, which accepted an incomplete fix without being able to check it).

**For the paper:** the dominant driver of pass/fail on this task set is the underlying agent's
raw capability per bug (8 of 10 outcomes don't change across personas at all), not persona
choice. Where persona does matter, the pattern is consistent with each persona's design: the
scope-locked nitpicker is the only one of the three that reliably leaves an already-correct fix
alone, while mind-changer and vague-requester both lost the one case (matplotlib-22865) where the
first attempt was already on the right track — one by reversing away from it, the other by never
being able to tell it was only partially right.
