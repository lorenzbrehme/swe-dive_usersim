"""swe_dive.sandbox — code that runs **inside** the Harbor container.

Everything in this subpackage executes in-container during a trial:

  - ``orchestrator``   — entry point; drives the user-sim ↔ coding-agent loop
  - ``user_sim``       — user simulator, personas, prompt templates, validators
  - ``coding_agents``  — persistent session backends (OpenHands SDK, ACP)
  - ``models``         — bare LiteLLM chat wrapper (model from ``$LLM_MODEL``)

The harbor agent (``swe_dive.harbor_agents``) uploads just this subpackage
to the container and sets ``PYTHONPATH`` so it is importable — no pip install
of swe_dive itself required inside the sandbox.

Contrast with the **host-side** packages:
  - ``swe_dive.harbor_agents`` — Harbor entrypoints (need ``harbor`` installed)
  - ``swe_dive.adapters``      — dataset generation (host-side tooling)
"""
