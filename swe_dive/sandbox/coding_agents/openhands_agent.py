"""OpenHands persistent multi-turn backend (OpenHands Agent SDK).

Keeps a single OpenHands ``Conversation`` alive for the whole user-simulator
interaction. Each turn calls ``conversation.send_message(...)`` + ``run()`` on the
*same* conversation, so the agent retains full memory across turns (no cold
start). It runs against a local, in-container workspace (the Harbor sandbox the
orchestrator already executes in) — no nested ``DockerWorkspace`` — and operates
on the repo that is already checked out at ``base_commit`` (``/testbed``).

The model is a LiteLLM model string from the environment (``AGENT_LLM_MODEL`` or
``LLM_MODEL``); OpenHands resolves provider credentials from the env that Harbor
injected into the sandbox.

Heavy imports are lazy (inside :meth:`setup`) so the package stays importable —
and unit-testable — without OpenHands installed.
"""

import logging
import os
import re
import subprocess

logger = logging.getLogger("swe_dive.coding_agents.openhands")

# Action kinds that are orchestration signals, not tool invocations.
_SKIP_ACTION_KINDS = frozenset({
    "MessageEvent",
    "AgentFinishAction",
    "AgentThinkAction",
    "NullAction",
    "ChangeAgentStateAction",
    "AddTaskAction",
    "ModifyTaskAction",
})

# Event fields that are OpenHands SDK metadata, not tool arguments.
_OH_INTERNAL_FIELDS = frozenset({"kind", "id", "source", "timestamp", "thought"})

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _kind_to_function_name(kind: str) -> str:
    """Map an OpenHands action kind to a short ATIF function name.

    Examples::

        FileEditorAction → file_editor
        CmdRunAction     → cmd_run
        BashAction       → bash
    """
    base = kind.removesuffix("Action")
    return _CAMEL_RE.sub("_", base).lower()


class OpenHandsAgent:
    """Persistent OpenHands SDK conversation bound to the local workspace."""

    def __init__(self, workdir: str = "/testbed", llm_model: str | None = None) -> None:
        self.workdir = workdir
        self.llm_model = (
            (llm_model or os.getenv("AGENT_LLM_MODEL") or os.getenv("LLM_MODEL") or "")
            .strip()
        )
        if not self.llm_model:
            raise RuntimeError(
                "No model configured. Set LLM_MODEL (or AGENT_LLM_MODEL) "
                "(e.g. LLM_MODEL=openai/gpt-4o)."
            )
        self.agent = None
        self.conversation = None
        self.history: list[dict] = []
        # Index into self.history where the current turn's events begin.
        # Used by get_last_message() and get_turn_tool_calls() to scope their
        # search to the new events from the most recent run_task() call, rather
        # than scanning the full cumulative history and risking stale results.
        self._turn_start_idx: int = 0

    async def setup(self) -> None:
        from openhands.sdk import LLM, Agent, Conversation, Tool
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.task_tracker import TaskTrackerTool
        from openhands.tools.terminal import TerminalTool

        # Build LLM kwargs; pass provider-specific params explicitly so the
        # OpenHands SDK (which wraps LiteLLM) uses our configured values rather
        # than falling back to its own defaults (critical for Azure api_version).
        llm_kwargs: dict[str, object] = {"model": self.llm_model}
        api_key = os.getenv("LLM_API_KEY") or os.getenv("AZURE_API_KEY")
        if api_key:
            llm_kwargs["api_key"] = api_key
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("AZURE_API_BASE")
        if base_url:
            llm_kwargs["base_url"] = base_url
        api_version = os.getenv("LLM_API_VERSION") or os.getenv("AZURE_API_VERSION")
        if api_version:
            llm_kwargs["api_version"] = api_version
        llm = LLM(**llm_kwargs)

        self.agent = Agent(
            llm=llm,
            tools=[
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
                Tool(name=TaskTrackerTool.name),
            ],
        )
        # Local workspace = the directory the orchestrator already runs in.
        self.conversation = Conversation(agent=self.agent, workspace=self.workdir)

    async def run_task(self, message: str) -> None:
        # Record where this turn's new events start before we advance the
        # conversation. get_last_message() and get_turn_tool_calls() use this
        # boundary to avoid returning stale data from earlier turns.
        self._turn_start_idx = len(self.history)
        # Same conversation across turns → full multi-turn memory.
        self.conversation.send_message(message)
        self.conversation.run()
        self.history = [e.model_dump() for e in self.conversation.state.events]

    def get_last_message(self) -> str | None:
        # Prefer events from the current turn so we never return a stale
        # assistant message from a previous turn. Fall back to the full history
        # for the very first turn or when the SDK omits a new MessageEvent.
        new_events = self.history[self._turn_start_idx:]
        msg = self._extract_last_assistant_message(reversed(new_events))
        if msg:
            return msg
        # Models often end a turn by calling the ``finish`` tool directly
        # (e.g. Gemini) without emitting a standalone assistant MessageEvent.
        # The user-facing text then lives in the FinishAction's ``message``.
        # Fall back to it so the response is not treated as empty, which would
        # otherwise trip the orchestrator's AgentSilentFailureError guard even
        # though the agent did real work.
        finish = self._extract_finish_message(reversed(new_events))
        if finish:
            return finish
        msg = self._extract_last_assistant_message(reversed(self.history))
        if msg:
            return msg
        return self._extract_finish_message(reversed(self.history))

    def _extract_last_assistant_message(self, events) -> str | None:
        for ev in events:
            d = ev if isinstance(ev, dict) else ev
            if d.get("kind") != "MessageEvent":
                continue
            msg = d.get("llm_message") or {}
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                text = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                ).strip()
                if text:
                    return text
            elif isinstance(content, str) and content.strip():
                return content.strip()
        return None

    def _extract_finish_message(self, events) -> str | None:
        # Return the message from the most recent ``finish`` ActionEvent, if any.
        # OpenHands serialises the finish tool call as an ActionEvent whose
        # ``action`` payload is a FinishAction carrying the final ``message``.
        for ev in events:
            d = ev if isinstance(ev, dict) else ev
            if d.get("kind") != "ActionEvent":
                continue
            action = d.get("action") or {}
            if action.get("kind") != "FinishAction":
                continue
            message = action.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        return None

    def get_turn_tool_calls(self) -> list[dict]:
        """Return tool calls made during the most recent turn.

        Scans events added since the last ``run_task()`` call and pairs each
        action event with the observation that immediately follows it.  Returns
        a list of dicts with keys ``tool_call_id``, ``function_name``,
        ``arguments``, and ``output`` (the observation content, or ``None`` if
        no observation followed the action).
        """
        result: list[dict] = []
        pending: dict | None = None

        for ev in self.history[self._turn_start_idx:]:
            d = ev if isinstance(ev, dict) else ev
            kind = d.get("kind", "")

            if kind.endswith("Action") and kind not in _SKIP_ACTION_KINDS:
                # Flush an action that had no following observation.
                if pending is not None:
                    result.append(pending)
                args = {k: v for k, v in d.items() if k not in _OH_INTERNAL_FIELDS}
                pending = {
                    "tool_call_id": f"oh_{len(result)}",
                    "function_name": _kind_to_function_name(kind),
                    "arguments": args,
                    "output": None,
                }
            elif kind.endswith("Observation") and pending is not None:
                pending["output"] = d.get("content")
                result.append(pending)
                pending = None

        if pending is not None:
            result.append(pending)

        return result

    @property
    def usage(self) -> dict | None:
        if self.agent is None:
            return None
        metrics = getattr(self.agent.llm, "metrics", None)
        if metrics is None:
            return None
        cost = getattr(metrics, "accumulated_cost", None)
        token_usage = getattr(metrics, "accumulated_token_usage", None)
        input_tokens = getattr(token_usage, "prompt_tokens", None) if token_usage else None
        output_tokens = getattr(token_usage, "completion_tokens", None) if token_usage else None
        cache_read = getattr(token_usage, "cache_read_tokens", None) if token_usage else None
        reasoning = getattr(token_usage, "reasoning_tokens", None) if token_usage else None
        if input_tokens is None and output_tokens is None and cost is None:
            return None
        return {
            "input_tokens": input_tokens or None,
            "output_tokens": output_tokens or None,
            "cache_read_tokens": cache_read or None,
            "reasoning_tokens": reasoning or None,
            "cost_usd": cost or None,
        }

    def git_diff(self) -> str:
        subprocess.run(
            ["git", "add", "-A"], cwd=self.workdir, capture_output=True, text=True
        )
        result = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=self.workdir,
            capture_output=True,
            text=True,
        )
        return result.stdout

    async def cleanup(self) -> None:
        if self.conversation is not None:
            try:
                self.conversation.close()
            except Exception as e:  # noqa: BLE001
                logger.warning("Conversation cleanup failed: %s", e)
            finally:
                self.conversation = None
