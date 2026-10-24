"""Unit tests for orchestrator-level error detection.

All tests run entirely offline — no Harbor, Docker, LLM, or network required.
"""

import argparse
import asyncio
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_dive.sandbox.orchestrator import _run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _args(tmp_path: Path) -> argparse.Namespace:
    instr_file = tmp_path / "instruction.md"
    instr_file.write_text("fix the separability bug")
    return argparse.Namespace(
        instruction=None,
        instruction_file=str(instr_file),
        workdir=str(tmp_path),
        repo_context="",
        persona="EXPERT_NITPICKER",
        coding_mode="COLLABORATIVE",
        max_turns=2,
        max_retries=0,
        backend="acp",
        acp_launcher="/fake/launch.sh",
        acp_runner_dir="/fake",
        log_dir=str(tmp_path),
        debug=False,
        sim_model="fake-model",
        validation_model=None,
        agent_model=None,
    )


def _patched_run(history: list[dict], tmp_path: Path) -> int:
    """Run _run() with LiteLLMChat, _build_backend, and UserSimulator mocked out."""
    mock_llm = MagicMock()
    mock_llm.MODEL_NAME = "fake"
    mock_llm.usage = {"input_tokens": None, "output_tokens": None, "cost_usd": None}

    mock_backend = MagicMock()
    mock_backend.setup = AsyncMock()
    mock_backend.cleanup = AsyncMock()
    mock_backend.git_diff.return_value = ""
    mock_backend.agent_model = None
    mock_backend.llm_model = None
    mock_backend.usage = {"input_tokens": None, "output_tokens": None, "cost_usd": None}

    mock_sim = MagicMock()
    mock_sim.simulateInteraction = AsyncMock(return_value=history)

    with ExitStack() as stack:
        stack.enter_context(
            patch("swe_dive.sandbox.orchestrator.LiteLLMChat", return_value=mock_llm)
        )
        stack.enter_context(
            patch("swe_dive.sandbox.orchestrator._build_backend", return_value=mock_backend)
        )
        stack.enter_context(
            patch("swe_dive.sandbox.orchestrator.UserSimulator", return_value=mock_sim)
        )
        return asyncio.run(_run(_args(tmp_path)))


def _patched_run_multi(
    side_effects: list,
    tmp_path: Path,
    max_retries: int = 1,
) -> tuple[int, MagicMock, MagicMock]:
    """Like ``_patched_run``, but ``simulateInteraction()`` returns (or
    raises) a different value per call — one per retry attempt — and both
    ``_build_backend`` and ``UserSimulator`` construction are tracked so
    tests can assert how many attempts were actually made.
    """
    mock_llm = MagicMock()
    mock_llm.MODEL_NAME = "fake"
    mock_llm.usage = {"input_tokens": None, "output_tokens": None, "cost_usd": None}

    def _make_backend(_args):
        backend = MagicMock()
        backend.setup = AsyncMock()
        backend.cleanup = AsyncMock()
        backend.git_diff.return_value = ""
        backend.agent_model = None
        backend.llm_model = None
        backend.usage = {"input_tokens": None, "output_tokens": None, "cost_usd": None}
        return backend

    mock_sim = MagicMock()
    mock_sim.simulateInteraction = AsyncMock(side_effect=side_effects)

    args = _args(tmp_path)
    args.max_retries = max_retries

    with ExitStack() as stack:
        stack.enter_context(
            patch("swe_dive.sandbox.orchestrator.LiteLLMChat", return_value=mock_llm)
        )
        build_backend_mock = MagicMock(side_effect=_make_backend)
        stack.enter_context(
            patch("swe_dive.sandbox.orchestrator._build_backend", build_backend_mock)
        )
        user_sim_mock = MagicMock(return_value=mock_sim)
        stack.enter_context(
            patch("swe_dive.sandbox.orchestrator.UserSimulator", user_sim_mock)
        )
        # Real workdir isn't a git repo in this test — no-op the reset.
        stack.enter_context(
            patch("swe_dive.sandbox.orchestrator._reset_workdir", MagicMock())
        )
        result = asyncio.run(_run(args))
        return result, build_backend_mock, user_sim_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrchestratorErrorDetection:
    def test_all_empty_agent_responses_raises(self, tmp_path):
        """Every turn has an empty agent_response → RuntimeError (broken backend)."""
        history = [
            {"user_message": "find the bug", "agent_response": None, "agent_changes": ""},
            {"user_message": "you did nothing", "agent_response": "", "agent_changes": ""},
        ]
        with pytest.raises(RuntimeError, match="empty response on all"):
            _patched_run(history, tmp_path)

    def test_whitespace_only_response_treated_as_empty(self, tmp_path):
        """Whitespace-only responses are treated the same as empty."""
        history = [
            {"user_message": "fix it", "agent_response": "   \n  ", "agent_changes": ""},
            {"user_message": "[DONE]", "agent_response": "", "agent_changes": ""},
        ]
        with pytest.raises(RuntimeError, match="empty response on all"):
            _patched_run(history, tmp_path)

    def test_at_least_one_real_response_returns_zero(self, tmp_path):
        """At least one non-empty agent_response → normal exit (reward=0 is fine)."""
        history = [
            {"user_message": "find the bug", "agent_response": "I looked at the code.", "agent_changes": ""},
            {"user_message": "[DONE]", "agent_response": "", "agent_changes": ""},
        ]
        assert _patched_run(history, tmp_path) == 0

    def test_empty_history_raises(self, tmp_path):
        """Empty history (e.g. initial decomposition failed 3x) → RuntimeError,
        not a silent reward=0."""
        with pytest.raises(RuntimeError, match="aborted early"):
            _patched_run([], tmp_path)

    def test_partial_history_without_done_raises(self, tmp_path):
        """Fewer turns than max_turns and no '[DONE]' → the simulator aborted
        mid-loop (e.g. 3 failed follow-up parses), not a legitimate reward=0."""
        history = [
            {"user_message": "find the bug", "agent_response": "Looking into it.", "agent_changes": ""},
        ]
        with pytest.raises(RuntimeError, match="aborted early"):
            _patched_run(history, tmp_path)

    def test_done_signal_returns_zero(self, tmp_path):
        """History ending with an explicit '[DONE]' is a clean finish."""
        history = [
            {"user_message": "find the bug", "agent_response": "Looking into it.", "agent_changes": ""},
            {"user_message": "[DONE]", "agent_response": "", "agent_changes": ""},
        ]
        assert _patched_run(history, tmp_path) == 0

    def test_max_turns_reached_without_done_returns_zero(self, tmp_path):
        """Running the full max_turns budget without a '[DONE]' is also a
        legitimate (if unresolved) finish, not an infrastructure error."""
        history = [
            {"user_message": "find the bug", "agent_response": "Looking into it.", "agent_changes": ""},
            {"user_message": "still not done", "agent_response": "Still working.", "agent_changes": ""},
        ]
        assert _patched_run(history, tmp_path) == 0


class TestOrchestratorRetries:
    """Retrying a full trial when the failure is the simulator's fault, not
    the coding agent's."""

    def test_retries_after_simulator_abort_then_succeeds(self, tmp_path):
        """First attempt aborts early (simulator's fault); the retry succeeds
        with a clean '[DONE]' — the trial should end up passing."""
        aborted_history = [
            {"user_message": "find the bug", "agent_response": "Looking into it.", "agent_changes": ""},
        ]
        clean_history = [
            {"user_message": "find the bug", "agent_response": "Looking into it.", "agent_changes": ""},
            {"user_message": "[DONE]", "agent_response": "", "agent_changes": ""},
        ]
        result, build_backend_mock, user_sim_mock = _patched_run_multi(
            [aborted_history, clean_history], tmp_path, max_retries=1
        )
        assert result == 0
        assert build_backend_mock.call_count == 2
        assert user_sim_mock.call_count == 2

    def test_retries_exhausted_raises(self, tmp_path):
        """Every attempt aborts early — once retries are exhausted the
        original 'aborted early' error still propagates."""
        aborted_history = [
            {"user_message": "find the bug", "agent_response": "Looking into it.", "agent_changes": ""},
        ]
        with pytest.raises(RuntimeError, match="aborted early"):
            _patched_run_multi(
                [aborted_history, aborted_history], tmp_path, max_retries=1
            )

    def test_agent_silent_failure_is_not_retried(self, tmp_path):
        """An empty-response backend failure is the agent's fault, not the
        simulator's — it should propagate immediately without retrying."""
        history = [
            {"user_message": "find the bug", "agent_response": None, "agent_changes": ""},
            {"user_message": "[DONE]", "agent_response": "", "agent_changes": ""},
        ]
        with pytest.raises(RuntimeError, match="empty response on all"):
            _patched_run_multi([history], tmp_path, max_retries=2)

    def test_zero_max_retries_disables_retrying(self, tmp_path):
        """max_retries=0 means exactly one attempt, matching the pre-retry
        behavior."""
        aborted_history = [
            {"user_message": "find the bug", "agent_response": "Looking into it.", "agent_changes": ""},
        ]
        with pytest.raises(RuntimeError, match="aborted early"):
            _patched_run_multi([aborted_history], tmp_path, max_retries=0)
