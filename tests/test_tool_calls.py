"""Tests for tool-call capture, banner stripping, and related helpers.

All tests are offline — no Harbor, Docker, LLM, or network required.
"""

import asyncio
import json
from pathlib import Path

import pytest

from swe_dive.sandbox.coding_agents.acp_agent import (
    _acp_kind_to_function_name,
    _strip_agent_banner,
)
from swe_dive.sandbox.coding_agents.mock import MockCodingAgent
from swe_dive.sandbox.coding_agents.openhands_agent import (
    OpenHandsAgent,
    _kind_to_function_name,
)
from swe_dive.sandbox.user_sim.user_simulator import UserSimulator


# ---------------------------------------------------------------------------
# Helpers shared with test_user_simulator
# ---------------------------------------------------------------------------

class ScriptedModel:
    """Replays a fixed list of responses, cycling through them on demand."""

    MODEL_NAME = "scripted-fake"
    usage: dict = {}

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._cursor = 0
        self.history: list[dict] = []

    def _next(self) -> str:
        if self._cursor >= len(self._responses):
            return json.dumps({"follow-up": "[DONE]"})
        resp = self._responses[self._cursor]
        self._cursor += 1
        return resp

    def chat_with_model(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        reply = self._next()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def delete_last_message(self) -> None:
        if len(self.history) >= 2:
            self.history.pop()
            self.history.pop()

    def reset_chat(self) -> None:
        self.history = []
        self._cursor = 0

    def prompt(self, message: str) -> str:
        return self._next()


def _always_valid() -> ScriptedModel:
    return ScriptedModel([
        json.dumps({"is_valid": True, "reason": "ok"}),
    ] * 50)


def _valid_init(msg: str = "look at the code") -> str:
    return json.dumps({
        "initial_message": msg,
        "subtask_list": ["Step 1: find bug", "Step 2: fix it"],
    })


def _done() -> str:
    return json.dumps({"follow-up": "[DONE]"})


# ---------------------------------------------------------------------------
# _strip_agent_banner
# ---------------------------------------------------------------------------

class TestStripAgentBanner:
    def test_pi_full_banner_becomes_empty(self):
        banner = (
            "pi v0.74.2\n"
            "---\n"
            "\n"
            "---\n"
            "New version available: v0.80.2 (installed v0.74.2). "
            "Run: `npm i -g @earendil-works/pi-coding-agent`"
        )
        assert _strip_agent_banner(banner) == ""

    def test_banner_followed_by_real_content(self):
        text = "pi v0.74.2\n---\nI fixed the bug in separable.py."
        assert _strip_agent_banner(text) == "I fixed the bug in separable.py."

    def test_real_content_then_trailing_banner(self):
        text = "Here is the fix.\n---\nNew version available: v2.0.0"
        assert _strip_agent_banner(text) == "Here is the fix."

    def test_no_banner_passes_through(self):
        text = "The bug is on line 245. Change `= 1` to `= right`."
        assert _strip_agent_banner(text) == text

    def test_empty_string(self):
        assert _strip_agent_banner("") == ""

    def test_only_dashes(self):
        assert _strip_agent_banner("---\n---\n---") == ""

    def test_upgrade_notice_stripped(self):
        text = "New version available: v3.0.0"
        assert _strip_agent_banner(text) == ""

    def test_npm_run_line_stripped(self):
        text = "Run: `npm i -g some-package`"
        assert _strip_agent_banner(text) == ""


# ---------------------------------------------------------------------------
# _acp_kind_to_function_name
# ---------------------------------------------------------------------------

class TestAcpKindToFunctionName:
    def test_execute_maps_to_bash(self):
        assert _acp_kind_to_function_name("execute") == "bash"

    def test_read_maps_to_file_read(self):
        assert _acp_kind_to_function_name("read") == "file_read"

    def test_write_maps_to_file_write(self):
        assert _acp_kind_to_function_name("write") == "file_write"

    def test_edit_maps_to_file_edit(self):
        assert _acp_kind_to_function_name("edit") == "file_edit"

    def test_unknown_kind_passes_through(self):
        assert _acp_kind_to_function_name("custom_tool") == "custom_tool"


# ---------------------------------------------------------------------------
# OpenHands _kind_to_function_name
# ---------------------------------------------------------------------------

class TestOpenHandsKindToFunctionName:
    def test_file_editor_action(self):
        assert _kind_to_function_name("FileEditorAction") == "file_editor"

    def test_cmd_run_action(self):
        assert _kind_to_function_name("CmdRunAction") == "cmd_run"

    def test_bash_action(self):
        assert _kind_to_function_name("BashAction") == "bash"

    def test_task_tracker_action(self):
        assert _kind_to_function_name("TaskTrackerAction") == "task_tracker"


# ---------------------------------------------------------------------------
# OpenHandsAgent.get_last_message — turn-boundary scoping
# ---------------------------------------------------------------------------

class TestOpenHandsGetLastMessage:
    """Verifies that get_last_message() only looks at the current turn's events
    to avoid returning stale responses from previous turns."""

    def _msg_event(self, role: str, text: str) -> dict:
        return {"kind": "MessageEvent", "llm_message": {"role": role, "content": text}}

    def test_returns_latest_from_current_turn(self):
        agent = OpenHandsAgent(llm_model="fake")
        # Turn 1 events
        agent.history = [
            self._msg_event("user", "fix the bug"),
            self._msg_event("assistant", "turn-1 answer"),
        ]
        # Turn 2 events start at index 2
        agent.history += [
            self._msg_event("user", "check helpers too"),
            self._msg_event("assistant", "turn-2 answer"),
        ]
        agent._turn_start_idx = 2
        assert agent.get_last_message() == "turn-2 answer"

    def test_does_not_return_stale_message_from_prior_turn(self):
        """If the current turn has no new MessageEvent, get_last_message() falls
        back to the full history rather than silently returning None."""
        agent = OpenHandsAgent(llm_model="fake")
        agent.history = [
            self._msg_event("user", "fix the bug"),
            self._msg_event("assistant", "only answer"),
        ]
        # Simulate a turn boundary past the existing events (no new events yet).
        agent._turn_start_idx = 2
        # Falls back to full history.
        assert agent.get_last_message() == "only answer"

    def test_returns_none_when_no_assistant_event_at_all(self):
        agent = OpenHandsAgent(llm_model="fake")
        agent.history = [self._msg_event("user", "hello")]
        agent._turn_start_idx = 0
        assert agent.get_last_message() is None

    def test_multipart_content_joined(self):
        agent = OpenHandsAgent(llm_model="fake")
        event = {
            "kind": "MessageEvent",
            "llm_message": {
                "role": "assistant",
                "content": [{"text": "hello "}, {"text": "world"}],
            },
        }
        agent.history = [event]
        agent._turn_start_idx = 0
        assert agent.get_last_message() == "hello world"

    def _finish_event(self, message: str) -> dict:
        return {
            "kind": "ActionEvent",
            "tool_name": "finish",
            "action": {"kind": "FinishAction", "message": message},
        }

    def test_falls_back_to_finish_action_message(self):
        """When a turn ends via the finish tool with no standalone assistant
        MessageEvent (common with Gemini), the FinishAction message is used so
        the response is not treated as empty (which would trip the
        orchestrator's AgentSilentFailureError guard)."""
        agent = OpenHandsAgent(llm_model="fake")
        agent.history = [
            {"kind": "FileEditorAction", "command": "str_replace"},
            self._finish_event("Changed the Morse code mapping for '1'."),
        ]
        agent._turn_start_idx = 0
        assert agent.get_last_message() == "Changed the Morse code mapping for '1'."

    def test_assistant_message_preferred_over_finish(self):
        agent = OpenHandsAgent(llm_model="fake")
        agent.history = [
            self._msg_event("assistant", "here is my summary"),
            self._finish_event("done"),
        ]
        agent._turn_start_idx = 0
        assert agent.get_last_message() == "here is my summary"

    def test_finish_fallback_scoped_to_current_turn(self):
        agent = OpenHandsAgent(llm_model="fake")
        agent.history = [
            self._finish_event("turn-1 finish"),
            self._finish_event("turn-2 finish"),
        ]
        agent._turn_start_idx = 1
        assert agent.get_last_message() == "turn-2 finish"


# ---------------------------------------------------------------------------
# OpenHandsAgent.get_turn_tool_calls — action/observation pairing
# ---------------------------------------------------------------------------

class TestOpenHandsGetTurnToolCalls:
    def test_action_observation_pair(self):
        agent = OpenHandsAgent(llm_model="fake")
        agent._turn_start_idx = 0
        agent.history = [
            {"kind": "FileEditorAction", "command": "view", "path": "/testbed"},
            {"kind": "FileEditorObservation", "content": "file listing..."},
        ]
        calls = agent.get_turn_tool_calls()
        assert len(calls) == 1
        assert calls[0]["function_name"] == "file_editor"
        assert calls[0]["arguments"]["command"] == "view"
        assert calls[0]["output"] == "file listing..."

    def test_action_without_following_observation(self):
        agent = OpenHandsAgent(llm_model="fake")
        agent._turn_start_idx = 0
        agent.history = [
            {"kind": "CmdRunAction", "command": "pytest tests/"},
        ]
        calls = agent.get_turn_tool_calls()
        assert len(calls) == 1
        assert calls[0]["function_name"] == "cmd_run"
        assert calls[0]["output"] is None

    def test_skips_finish_and_message_events(self):
        agent = OpenHandsAgent(llm_model="fake")
        agent._turn_start_idx = 0
        agent.history = [
            {"kind": "MessageEvent", "llm_message": {"role": "assistant", "content": "done"}},
            {"kind": "AgentFinishAction", "outputs": {}},
            {"kind": "FileEditorAction", "command": "str_replace", "path": "/testbed/foo.py"},
            {"kind": "FileEditorObservation", "content": "replaced"},
        ]
        calls = agent.get_turn_tool_calls()
        assert len(calls) == 1
        assert calls[0]["function_name"] == "file_editor"

    def test_multiple_tool_calls_in_one_turn(self):
        agent = OpenHandsAgent(llm_model="fake")
        agent._turn_start_idx = 0
        agent.history = [
            {"kind": "FileEditorAction", "command": "view", "path": "/testbed"},
            {"kind": "FileEditorObservation", "content": "listing"},
            {"kind": "CmdRunAction", "command": "grep -r foo ."},
            {"kind": "CmdOutputObservation", "content": "matches"},
        ]
        calls = agent.get_turn_tool_calls()
        assert len(calls) == 2
        assert calls[0]["function_name"] == "file_editor"
        assert calls[1]["function_name"] == "cmd_run"
        assert calls[1]["output"] == "matches"

    def test_respects_turn_start_idx(self):
        """Events before _turn_start_idx must not appear in tool calls."""
        agent = OpenHandsAgent(llm_model="fake")
        agent.history = [
            # Turn 1 (should be ignored)
            {"kind": "CmdRunAction", "command": "old command"},
            {"kind": "CmdOutputObservation", "content": "old result"},
            # Turn 2 (should be included)
            {"kind": "CmdRunAction", "command": "new command"},
            {"kind": "CmdOutputObservation", "content": "new result"},
        ]
        agent._turn_start_idx = 2
        calls = agent.get_turn_tool_calls()
        assert len(calls) == 1
        assert calls[0]["arguments"]["command"] == "new command"
        assert calls[0]["output"] == "new result"

    def test_kind_not_in_arguments(self):
        """The 'kind' field must be excluded from tool-call arguments."""
        agent = OpenHandsAgent(llm_model="fake")
        agent._turn_start_idx = 0
        agent.history = [
            {"kind": "CmdRunAction", "command": "ls /testbed"},
        ]
        calls = agent.get_turn_tool_calls()
        assert "kind" not in calls[0]["arguments"]
        assert calls[0]["arguments"]["command"] == "ls /testbed"


# ---------------------------------------------------------------------------
# UserSimulator includes tool_calls in history entries
# ---------------------------------------------------------------------------

class MockAgentWithToolCalls(MockCodingAgent):
    """MockCodingAgent extended with scripted per-turn tool calls."""

    def __init__(self, *args, scripted_tool_calls: list[list[dict]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._scripted_tool_calls = scripted_tool_calls or []

    def get_turn_tool_calls(self) -> list[dict]:
        idx = max(0, self._turn - 1)
        if idx < len(self._scripted_tool_calls):
            return self._scripted_tool_calls[idx]
        return []


class TestSimulatorToolCallsInHistory:
    """The simulator must store tool_calls in every history entry."""

    def _run(self, sim) -> list[dict]:
        return asyncio.run(sim.simulateInteraction())

    def _build_sim(self, agent, tmp_path: Path) -> UserSimulator:
        sim_model = ScriptedModel([_valid_init(), _done()])
        val_model = _always_valid()
        return UserSimulator(
            issue="fix the bug",
            repo_context="",
            model=sim_model,
            agent=agent,
            validation_model=val_model,
            log_dir=str(tmp_path),
            max_turns=5,
        )

    def test_tool_calls_included_when_backend_supports_it(self, tmp_path):
        tool_call = {
            "tool_call_id": "tc0",
            "function_name": "bash",
            "arguments": {"command": "pytest tests/"},
            "output": "1 passed",
        }
        agent = MockAgentWithToolCalls(
            replies=["fixed the separability bug"],
            scripted_tool_calls=[[tool_call]],
        )
        history = self._run(self._build_sim(agent, tmp_path))

        assert len(history) >= 1
        first = history[0]
        assert "tool_calls" in first
        assert first["tool_calls"] == [tool_call]

    def test_tool_calls_empty_for_backend_without_method(self, tmp_path):
        """Backends that don't implement get_turn_tool_calls get [] per turn."""
        agent = MockCodingAgent(replies=["found the bug"])
        assert not hasattr(agent, "get_turn_tool_calls")
        history = self._run(self._build_sim(agent, tmp_path))

        assert len(history) >= 1
        assert history[0]["tool_calls"] == []

    def test_tool_calls_preserved_across_multiple_turns(self, tmp_path):
        calls_turn1 = [{"tool_call_id": "t0", "function_name": "bash",
                        "arguments": {"command": "grep bug"}, "output": "found"}]
        calls_turn2 = [{"tool_call_id": "t1", "function_name": "file_editor",
                        "arguments": {"command": "str_replace"}, "output": "ok"}]

        sim_model = ScriptedModel([
            _valid_init(),
            json.dumps({"follow-up": "now add a test"}),
            _done(),
        ])
        val_model = _always_valid()
        agent = MockAgentWithToolCalls(
            replies=["fixed", "test added"],
            scripted_tool_calls=[calls_turn1, calls_turn2],
        )
        sim = UserSimulator(
            issue="multi-turn",
            repo_context="",
            model=sim_model,
            agent=agent,
            validation_model=val_model,
            log_dir=str(tmp_path),
            max_turns=5,
        )
        history = self._run(sim)
        agent_turns = [t for t in history if t.get("agent_response")]
        assert len(agent_turns) >= 2
        assert agent_turns[0]["tool_calls"] == calls_turn1
        assert agent_turns[1]["tool_calls"] == calls_turn2
