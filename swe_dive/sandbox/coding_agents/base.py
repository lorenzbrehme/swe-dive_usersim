"""The ``CodingAgentBackend`` protocol.

The user simulator is fully agent-agnostic: it only needs to send a message,
read the agent's last reply, and inspect the current repository diff. Any coding
agent (OpenHands, Copilot, ...) is integrated by implementing this protocol.


"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class CodingAgentBackend(Protocol):
    """Minimal interface the user simulator drives."""

    async def setup(self) -> None:
        """Prepare the agent (connect to workspace, start session, ...).

        For in-container backends the repository is already checked out at the
        task's ``base_commit``; ``setup`` should NOT clone or checkout.
        """
        ...

    async def run_task(self, message: str) -> None:
        """Send one user message and let the agent run to completion."""
        ...

    def get_last_message(self) -> str | None:
        """Return the agent's most recent assistant text message."""
        ...

    def git_diff(self) -> str:
        """Return the staged working-tree diff (``git add -A && git diff --cached``)."""
        ...

    async def cleanup(self) -> None:
        """Release any resources held by the backend."""
        ...
