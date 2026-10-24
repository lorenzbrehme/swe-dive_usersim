"""A deterministic, dependency-free coding-agent backend for offline tests.

``MockCodingAgent`` lets us exercise the full user-simulator transformation
pipeline (decompose -> validate -> turn loop -> [DONE]) without OpenHands, Docker,
or any network access. It records every message it receives and returns scripted
replies / diffs.
"""

from collections.abc import Sequence


class MockCodingAgent:
    """Records received messages and replays scripted replies/diffs."""

    def __init__(
        self,
        replies: Sequence[str] | None = None,
        diffs: Sequence[str] | None = None,
        default_reply: str = "done, made the change.",
        default_diff: str = "",
    ) -> None:
        self._replies = list(replies or [])
        self._diffs = list(diffs or [])
        self._default_reply = default_reply
        self._default_diff = default_diff
        self.received_messages: list[str] = []
        self._turn = 0
        self._last_message: str | None = None
        self._current_diff = default_diff

    async def setup(self) -> None:  # noqa: D401 - protocol method
        return None

    async def run_task(self, message: str) -> None:
        self.received_messages.append(message)
        idx = self._turn
        self._last_message = (
            self._replies[idx] if idx < len(self._replies) else self._default_reply
        )
        self._current_diff = (
            self._diffs[idx] if idx < len(self._diffs) else self._default_diff
        )
        self._turn += 1

    def get_last_message(self) -> str | None:
        return self._last_message

    def git_diff(self) -> str:
        return self._current_diff

    @property
    def usage(self) -> dict | None:
        return None

    async def cleanup(self) -> None:  # noqa: D401 - protocol method
        return None
