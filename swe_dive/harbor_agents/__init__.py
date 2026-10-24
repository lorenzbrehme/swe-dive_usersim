"""Harbor agent entrypoints for the swe_dive eval.

These are the **host-side** Harbor classes — subclasses of Harbor's
``BaseInstalledAgent`` (or ``AcpAgent``) that Harbor's trial runner calls on the
host to install the orchestrator + coding-agent deps into the sandbox and then
exec the user-simulator loop inside it.

Contrast with ``swe_dive.coding_agents``, which are the **in-container**
persistent session backends (OpenHands SDK Conversation, ACP session) that the
user simulator actually drives turn-by-turn.
"""

from .user_sim_agent import UserSimAcpAgent, UserSimOpenHandsAgent

__all__ = ["UserSimOpenHandsAgent", "UserSimAcpAgent"]
