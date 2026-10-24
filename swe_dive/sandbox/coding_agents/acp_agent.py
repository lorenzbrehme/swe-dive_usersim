"""Persistent, multi-turn ACP coding-agent backend (reuses Harbor's ACP stack).

Opens a single ACP session and drives it turn-by-turn so the agent retains full
conversational memory (no cold start per simulator turn).

Because this runs in-container, it imports Harbor's uploaded ``acp_runner`` module
and the ``acp`` package from the venv Harbor set up — see ``UserSimAcpAgent`` in
``swe_dive.agents.user_sim_agent`` for the host side that produces them. All heavy
imports are lazy (inside :meth:`setup`).
"""

import os
import re as _re
import subprocess
import sys
from contextlib import AsyncExitStack
from pathlib import Path
import logging

logger = logging.getLogger("swe_dive.coding_agents.acp")

# Defaults match Harbor's AcpAgent constants.
DEFAULT_LAUNCHER_PATH = "/installed-agent/acp-launch.sh"
DEFAULT_RUNNER_DIR = "/installed-agent"

# ── Banner-stripping helpers ─────────────────────────────────────────────────
# Some ACP agents (e.g. pi-acp) emit a version/upgrade banner before—or instead
# of—a real coding response. Strip those lines so they don't pollute the
# trajectory or trigger downstream content filters when included in follow-up
# prompts.

_BANNER_LINE_RE = _re.compile(
    r'^\s*(?:'
    r'[-]{3,}'                                    # --- or -----
    r'|[a-zA-Z0-9][\w.-]* v\d+\.\d+[\d.]*.*'    # "agentname v1.2.3..."
    r'|New version available:.*'                  # upgrade notice
    r'|Run:\s*`npm.*'                             # npm install suggestion
    r')\s*$',
    _re.IGNORECASE,
)


def _strip_agent_banner(text: str) -> str:
    """Remove leading/trailing version-banner lines from an ACP agent response.

    After stripping, if the result is empty the caller should treat the response
    as absent (``None``) so the empty-response guard in the orchestrator fires.
    """
    lines = text.splitlines()
    start = 0
    while start < len(lines) and _BANNER_LINE_RE.match(lines[start]):
        start += 1
    end = len(lines)
    while end > start and _BANNER_LINE_RE.match(lines[end - 1]):
        end -= 1
    return "\n".join(lines[start:end]).strip()


def _acp_kind_to_function_name(kind: str) -> str:
    """Map an ACP tool kind to a short ATIF function name."""
    return {
        "execute": "bash",
        "read": "file_read",
        "write": "file_write",
        "edit": "file_edit",
    }.get(kind, kind)


class ACPAgent:
    """A long-lived ACP session driven turn-by-turn by the user simulator."""

    def __init__(
        self,
        launcher_path: str = DEFAULT_LAUNCHER_PATH,
        runner_dir: str = DEFAULT_RUNNER_DIR,
        workdir: str = "/testbed",
        logs_dir: str = "/logs/agent",
        agent_model: str | None = None,
        permission_mode: str = "allow",
        auth_method_id: str | None = None,
    ) -> None:
        self.launcher_path = launcher_path
        self.runner_dir = runner_dir
        self.workdir = workdir
        self.logs_dir = logs_dir
        self.agent_model = (agent_model or os.getenv("AGENT_LLM_MODEL") or "").strip() or None
        self.permission_mode = permission_mode
        self.auth_method_id = auth_method_id

        self._stack: AsyncExitStack | None = None
        self._conn = None
        self._client = None
        self._session_id: str | None = None
        self._last_message: str | None = None
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cached_read_tokens: int = 0
        self._total_thought_tokens: int = 0
        self._total_cached_write_tokens: int = 0
        self._has_usage: bool = False

    async def setup(self) -> None:
        # Reuse Harbor's uploaded ACP client + the acp package from its venv.
        if self.runner_dir not in sys.path:
            sys.path.insert(0, self.runner_dir)

        from acp import PROTOCOL_VERSION, spawn_agent_process
        from acp.schema import (
            AgentMessageChunk,
            AuthCapabilities,
            ClientCapabilities,
            EnvVarAuthMethod,
            FileSystemCapabilities,
            UsageUpdate,
        )
        from acp_runner import HarborAcpClient  # uploaded by Harbor's AcpAgent

        class _TurnClient(HarborAcpClient):
            """Harbor's ACP client + per-turn agent-message and tool-call accumulation."""

            def __init__(self, logs_dir: Path, permission_mode: str) -> None:
                super().__init__(logs_dir=logs_dir, permission_mode=permission_mode)
                self._turn_chunks: list[str] = []
                # Completed tool calls for the current turn (in arrival order).
                self._turn_tool_calls: list[dict] = []
                # Partially built entries awaiting their "completed" update.
                self._pending_calls: dict[str, dict] = {}
                # Cumulative session cost from UsageUpdate events.
                self._last_cost_usd: float | None = None

            def reset_turn(self) -> None:
                self._turn_chunks = []
                self._turn_tool_calls = []
                self._pending_calls = {}

            def take_message(self) -> str | None:
                raw = "".join(self._turn_chunks).strip()
                if not raw:
                    return None
                cleaned = _strip_agent_banner(raw)
                return cleaned or None

            def take_tool_calls(self) -> list[dict]:
                """Return completed tool calls; flush any still-pending entries."""
                for entry in self._pending_calls.values():
                    self._turn_tool_calls.append(entry)
                self._pending_calls.clear()
                return list(self._turn_tool_calls)

            async def session_update(self, session_id, update, **kwargs):
                if isinstance(update, AgentMessageChunk):
                    text = getattr(update.content, "text", None)
                    if text:
                        self._turn_chunks.append(text)
                    # Don't call super() for message chunks: Harbor's acp_runner
                    # prints each chunk as "[thought] {token}" to stdout, which
                    # fragments orchestrator.txt word-by-word. We collect the
                    # complete message in take_message() after the turn finishes.
                    return

                if isinstance(update, UsageUpdate):
                    cost = getattr(update, "cost", None)
                    if cost is not None:
                        self._last_cost_usd = getattr(cost, "amount", None)
                    await super().session_update(session_id, update, **kwargs)
                    return

                # Collect ACP tool-call events for the ATIF trajectory.
                su_type = getattr(update, "sessionUpdate", None)
                if su_type == "tool_call":
                    call_id = getattr(update, "toolCallId", "") or ""
                    kind = getattr(update, "kind", "unknown") or "unknown"
                    raw_input = getattr(update, "rawInput", None) or {}
                    arguments: dict = (
                        raw_input
                        if isinstance(raw_input, dict)
                        else (raw_input.model_dump() if hasattr(raw_input, "model_dump") else {})
                    )
                    self._pending_calls[call_id] = {
                        "tool_call_id": call_id,
                        "function_name": _acp_kind_to_function_name(kind),
                        "arguments": arguments,
                        "output": None,
                    }
                elif su_type == "tool_call_update":
                    call_id = getattr(update, "toolCallId", "") or ""
                    status = getattr(update, "status", None)
                    if status == "completed" and call_id in self._pending_calls:
                        raw_output = getattr(update, "rawOutput", None) or {}
                        output: str | None
                        if isinstance(raw_output, dict):
                            output = raw_output.get("content")
                        elif hasattr(raw_output, "content"):
                            output = raw_output.content
                        else:
                            output = None
                        self._pending_calls[call_id]["output"] = output
                        self._turn_tool_calls.append(self._pending_calls.pop(call_id))

                await super().session_update(session_id, update, **kwargs)

        log_dir = Path(self.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._client = _TurnClient(log_dir, self.permission_mode)

        workspace = Path(self.workdir)
        workspace.mkdir(parents=True, exist_ok=True)
        self._stack = AsyncExitStack()

        # Ensure the ACP agent process (e.g. Copilot CLI) can find its auth
        # token under the standard GITHUB_TOKEN name regardless of which env
        # var the user set.
        child_env = dict(os.environ)
        for src in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN"):
            if src in child_env and "GITHUB_TOKEN" not in child_env:
                child_env["GITHUB_TOKEN"] = child_env[src]

        # Ensure pi's azure-foundry extension finds the Azure API key via both
        # names (AZURE_API_KEY is the primary; OPENAI_API_KEY is the fallback
        # used by pi's built-in openai provider and conn.authenticate()).
        if "AZURE_API_KEY" in child_env and "OPENAI_API_KEY" not in child_env:
            child_env["OPENAI_API_KEY"] = child_env["AZURE_API_KEY"]

        conn, _process = await self._stack.enter_async_context(
            spawn_agent_process(
                self._client,
                self.launcher_path,
                env=child_env,
                cwd=workspace,
                # Merge stderr into stdout so it appears in orchestrator.txt.
                transport_kwargs={"stderr": None},
            )
        )
        self._conn = conn

        init = await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                auth=AuthCapabilities(terminal=False),
                fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),
                terminal=True,
            ),
        )

        method_id = self.auth_method_id
        if method_id is None:
            for method in init.auth_methods or []:
                if isinstance(method, EnvVarAuthMethod):
                    required = [v.name for v in method.vars if not v.optional]
                    if all(os.environ.get(name) for name in required):
                        method_id = method.id
                        break
        if method_id:
            try:
                await conn.authenticate(method_id=method_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("ACP authenticate failed (continuing): %s", e)

        session = await conn.new_session(cwd=str(workspace), mcp_servers=[])
        self._session_id = session.session_id

        if self.agent_model:
            try:
                # The model is already in openai/X format (translated in
                # UserSimAcpAgent.run() before passing --agent-model).
                await conn.set_config_option(
                    config_id="model",
                    value=self.agent_model,
                    session_id=self._session_id,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("ACP set_config_option(model) failed (continuing): %s", e)

    async def run_task(self, message: str) -> None:
        from acp import text_block

        self._client.reset_turn()
        response = await self._conn.prompt(
            session_id=self._session_id, prompt=[text_block(message)]
        )
        self._last_message = self._client.take_message()

        usage = getattr(response, "usage", None)
        if usage is not None:
            self._has_usage = True
            self._total_input_tokens += getattr(usage, "input_tokens", 0) or 0
            self._total_output_tokens += getattr(usage, "output_tokens", 0) or 0
            self._total_cached_read_tokens += getattr(usage, "cached_read_tokens", 0) or 0
            self._total_thought_tokens += getattr(usage, "thought_tokens", 0) or 0
            self._total_cached_write_tokens += getattr(usage, "cached_write_tokens", 0) or 0
    def get_last_message(self) -> str | None:
        return self._last_message

    @property
    def usage(self) -> dict | None:
        if not self._has_usage:
            return None
        cost = self._client._last_cost_usd if self._client is not None else None
        if not cost and self.agent_model:
            try:
                import litellm
                input_cost, output_cost = litellm.cost_per_token(
                    model=self.agent_model,
                    prompt_tokens=self._total_input_tokens,
                    completion_tokens=self._total_output_tokens + self._total_thought_tokens,
                    cache_read_input_tokens=self._total_cached_read_tokens,
                    cache_creation_input_tokens=self._total_cached_write_tokens,
                )
                cost = input_cost + output_cost
            except Exception:
                pass
        return {
            "input_tokens": self._total_input_tokens or None,
            "output_tokens": self._total_output_tokens or None,
            "cached_read_tokens": self._total_cached_read_tokens or None,
            "thought_tokens": self._total_thought_tokens or None,
            "cached_write_tokens": self._total_cached_write_tokens or None,
            "cost_usd": cost or None,
        }

    def get_turn_tool_calls(self) -> list[dict]:
        """Return tool calls collected during the most recent turn."""
        if self._client is None:
            return []
        return self._client.take_tool_calls()

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
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception as e:  # noqa: BLE001
                logger.warning("ACP cleanup failed: %s", e)
            finally:
                self._stack = None
                self._conn = None
