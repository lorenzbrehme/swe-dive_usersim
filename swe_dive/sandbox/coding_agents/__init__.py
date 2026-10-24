"""Coding-agent backends driven by the user simulator.

These are **in-container** classes that hold a *persistent* session so the
simulator gets true multi-turn memory (no cold start). They are driven
turns-by-turn by ``swe_dive.orchestrator`` running inside the Harbor sandbox.

Contrast with ``swe_dive.harbor_agents``, which are the *host-side* Harbor
entrypoints that Harbor's trial runner calls to install deps and exec the
orchestrator.

Available backends:
- ``OpenHandsAgent`` — persistent OpenHands SDK ``Conversation``.
- ``ACPAgent``        — persistent ACP ``session/prompt`` loop; reuses Harbor's
  uploaded ``HarborAcpClient`` and launcher. Covers any ACP-registry agent
  (e.g. ``acp:github-copilot-cli``).
- ``MockCodingAgent`` — deterministic scripted backend for offline unit tests.
"""

from .acp_agent import ACPAgent
from .base import CodingAgentBackend
from .openhands_agent import OpenHandsAgent

__all__ = ["CodingAgentBackend", "OpenHandsAgent", "ACPAgent"]
