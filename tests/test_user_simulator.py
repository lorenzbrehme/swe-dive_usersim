"""Unit tests for the swe_dive user-simulator transformation pipeline.

All tests run entirely offline using ``MockCodingAgent`` and a scripted fake
LLM model — no Harbor, OpenHands, ACP, or network access required.
"""

import asyncio
import json
import textwrap
from pathlib import Path

import pytest

from swe_dive.sandbox.coding_agents.mock import MockCodingAgent
from swe_dive.sandbox.user_sim.user_simulator import UserSimulator
from swe_dive.sandbox.user_sim.utils import parse_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ScriptedModel:
    """A fake LLM that plays back a pre-written response list."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._cursor = 0
        self.history: list[dict] = []
        self.MODEL_NAME = "scripted-fake"

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

    def prompt(self, message: str) -> str:
        return self._next()

    def reset_chat(self) -> None:
        self.history = []
        self._cursor = 0

    def get_chat_history(self) -> list[dict]:
        return self.history

    def delete_last_message(self) -> None:
        if len(self.history) >= 2:
            self.history.pop()
            self.history.pop()


def _valid_initial_response(subtasks: list[str] | None = None) -> str:
    return json.dumps({
        "initial_message": "fix the _cstack function in separable.py",
        "subtask_list": subtasks or [
            "Step 1: locate _cstack in separable.py",
            "Step 2: fix the cright assignment",
        ],
    })


def _valid_followup_response(message: str) -> str:
    return json.dumps({"follow-up": message})


def _valid_done_response() -> str:
    return json.dumps({"follow-up": "[DONE]"})


def _always_valid_model() -> ScriptedModel:
    """Validation model that always says is_valid=True."""
    return ScriptedModel([
        json.dumps({"is_valid": True, "reason": "looks good"}),
    ] * 50)


# ---------------------------------------------------------------------------
# parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_extracts_json_from_plain_object(self):
        result = parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_extracts_json_embedded_in_text(self):
        result = parse_json('Some text {"a": 1} trailing')
        assert result == {"a": 1}

    def test_returns_none_for_invalid_json(self):
        assert parse_json("not json") is None

    def test_returns_none_for_none(self):
        assert parse_json(None) is None

    def test_nested_object(self):
        raw = json.dumps({"x": {"y": [1, 2, 3]}})
        assert parse_json(raw) == {"x": {"y": [1, 2, 3]}}


# ---------------------------------------------------------------------------
# MockCodingAgent
# ---------------------------------------------------------------------------

class TestMockCodingAgent:
    def test_records_received_messages(self):
        agent = MockCodingAgent(replies=["done."], diffs=["diff --git a/f b/f"])
        asyncio.run(agent.setup())
        asyncio.run(agent.run_task("fix the bug"))
        assert agent.received_messages == ["fix the bug"]
        assert agent.get_last_message() == "done."
        assert "diff" in agent.git_diff()

    def test_default_reply_when_script_exhausted(self):
        agent = MockCodingAgent(replies=["r1"], default_reply="fallback")
        asyncio.run(agent.setup())
        asyncio.run(agent.run_task("msg1"))
        asyncio.run(agent.run_task("msg2"))
        assert agent.get_last_message() == "fallback"

    def test_cleanup_is_idempotent(self):
        agent = MockCodingAgent()
        asyncio.run(agent.setup())
        asyncio.run(agent.cleanup())
        asyncio.run(agent.cleanup())  # second cleanup must not raise


# ---------------------------------------------------------------------------
# UserSimulator — full pipeline
# ---------------------------------------------------------------------------

class TestSimulatorInitialPrompt:
    def test_initial_prompt_contains_issue(self):
        sim = UserSimulator(
            issue="fix the separability matrix",
            repo_context="diff ...",
            model=None,
            agent=None,
            validation_model=None,
        )
        prompt = sim.initial_prompt()
        assert "fix the separability matrix" in prompt
        assert "diff ..." in prompt

    def test_followup_prompt_contains_agent_message(self):
        sim = UserSimulator(model=None, agent=None, validation_model=None)
        prompt = sim.followup_prompt(
            agent_last_message="I changed the cright line",
            agent_changes="diff ...",
        )
        assert "I changed the cright line" in prompt
        assert "diff ..." in prompt


class TestSimulatorInteraction:
    """Test the full simulateInteraction loop using scripted models."""

    def _run(self, sim):
        return asyncio.run(sim.simulateInteraction())

    def test_single_turn_ends_on_done(self, tmp_path):
        # Sim model: initial response → 1 follow-up → DONE
        sim_model = ScriptedModel([
            _valid_initial_response(),
            _valid_done_response(),
        ])
        # Validation model: always valid
        val_model = _always_valid_model()
        agent = MockCodingAgent(
            replies=["changed the file."],
            diffs=["diff --git a/separable.py b/separable.py\n+fix"],
        )
        sim = UserSimulator(
            issue="fix separability",
            repo_context="",
            model=sim_model,
            agent=agent,
            validation_model=val_model,
            log_dir=str(tmp_path),
            max_turns=5,
        )
        history = self._run(sim)
        # Must have at least one turn + the [DONE] message
        assert len(history) >= 2
        assert "[DONE]" in history[-1]["user_message"]
        assert agent.received_messages[0] == "fix the _cstack function in separable.py"

    def test_multi_turn_accumulates_history(self, tmp_path):
        sim_model = ScriptedModel([
            _valid_initial_response(subtasks=["Step 1", "Step 2", "Step 3"]),
            _valid_followup_response("now fix the second part"),
            _valid_followup_response("clean up the helper"),
            _valid_done_response(),
        ])
        val_model = _always_valid_model()
        agent = MockCodingAgent(
            replies=["done step 1", "done step 2", "done step 3"],
            default_reply="done",
        )
        sim = UserSimulator(
            issue="multi-step fix",
            repo_context="",
            model=sim_model,
            agent=agent,
            validation_model=val_model,
            log_dir=str(tmp_path),
            max_turns=10,
        )
        history = self._run(sim)
        # 3 agent turns + the final [DONE]
        assert len(history) == 4
        assert len(agent.received_messages) == 3

    def test_empty_response_retried_then_aborts(self, tmp_path):
        """If the sim model consistently fails to parse, raise after 3 retries."""
        sim_model = ScriptedModel(["not json"] * 20)
        val_model = _always_valid_model()
        agent = MockCodingAgent()
        sim = UserSimulator(
            model=sim_model,
            agent=agent,
            validation_model=val_model,
            log_dir=str(tmp_path),
        )
        with pytest.raises(RuntimeError, match="Failed to get a valid follow-up"):
            self._run(sim)

    def test_max_turns_respected(self, tmp_path):
        # Model never sends [DONE], so the loop should stop at max_turns.
        sim_model = ScriptedModel(
            [_valid_initial_response()]
            + [_valid_followup_response("keep going")] * 20
        )
        val_model = _always_valid_model()
        agent = MockCodingAgent(default_reply="still working")
        sim = UserSimulator(
            issue="endless task",
            repo_context="",
            model=sim_model,
            agent=agent,
            validation_model=val_model,
            log_dir=str(tmp_path),
            max_turns=3,
        )
        history = self._run(sim)
        assert len(agent.received_messages) <= 3


# ---------------------------------------------------------------------------
# SWE-Bench adapter
# ---------------------------------------------------------------------------

class TestSWEBenchAdapter:
    @pytest.fixture()
    def sample_row(self) -> dict:
        return {
            "instance_id": "astropy__astropy-12907",
            "repo": "astropy/astropy",
            "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
            "problem_statement": "separability_matrix is broken for nested models",
            "test_patch": textwrap.dedent("""\
                diff --git a/astropy/modeling/tests/test_separable.py b/astropy/modeling/tests/test_separable.py
                --- a/astropy/modeling/tests/test_separable.py
                +++ b/astropy/modeling/tests/test_separable.py
                @@ -1,1 +1,2 @@
                +import pytest
            """),
            "patch": "diff ...",
            "FAIL_TO_PASS": '["astropy/modeling/tests/test_separable.py::test_separable"]',
            "PASS_TO_PASS": '["astropy/modeling/tests/test_separable.py::test_coord_matrix"]',
            "version": "4.3",
            "difficulty": "15 min - 1 hour",
        }

    def test_emit_task_creates_expected_files(self, sample_row, tmp_path):
        from swe_dive.adapters.swebench import SWEBenchAdapter

        adapter = SWEBenchAdapter.__new__(SWEBenchAdapter)
        task_dir = adapter.emit_task(sample_row, tmp_path)

        assert (task_dir / "instruction.md").exists()
        assert (task_dir / "task.toml").exists()
        assert (task_dir / "environment").is_dir()
        assert (task_dir / "tests" / "test.sh").exists()
        assert (task_dir / "tests" / "test_patch.diff").exists()
        assert (task_dir / "tests" / "eval_spec.json").exists()
        assert (task_dir / "tests" / "parse_report.py").exists()

    def test_instruction_matches_problem_statement(self, sample_row, tmp_path):
        from swe_dive.adapters.swebench import SWEBenchAdapter

        adapter = SWEBenchAdapter.__new__(SWEBenchAdapter)
        task_dir = adapter.emit_task(sample_row, tmp_path)

        assert "separability_matrix" in (task_dir / "instruction.md").read_text()

    def test_task_toml_has_correct_image(self, sample_row, tmp_path):
        from swe_dive.adapters.swebench import SWEBenchAdapter

        adapter = SWEBenchAdapter.__new__(SWEBenchAdapter)
        task_dir = adapter.emit_task(sample_row, tmp_path)

        toml_text = (task_dir / "task.toml").read_text()
        assert "astropy_1776_astropy-12907" in toml_text

    def test_eval_spec_contains_required_keys(self, sample_row, tmp_path):
        from swe_dive.adapters.swebench import SWEBenchAdapter

        adapter = SWEBenchAdapter.__new__(SWEBenchAdapter)
        task_dir = adapter.emit_task(sample_row, tmp_path)

        spec = json.loads((task_dir / "tests" / "eval_spec.json").read_text())
        assert "fail_to_pass" in spec
        assert "pass_to_pass" in spec
        assert "base_commit" in spec
        assert "test_files" in spec

    def test_test_files_extracted_from_patch(self, sample_row, tmp_path):
        from swe_dive.adapters.swebench import SWEBenchAdapter, test_files_from_patch

        files = test_files_from_patch(sample_row["test_patch"])
        assert "astropy/modeling/tests/test_separable.py" in files

    def test_image_key_replaces_double_underscore(self):
        from swe_dive.adapters.swebench import instance_image_key

        assert instance_image_key("astropy__astropy-12907") == "astropy_1776_astropy-12907"
        assert instance_image_key("django__django-11099") == "django_1776_django-11099"

    @pytest.mark.slow
    def test_parse_report_scores_correctly(self, sample_row, tmp_path):
        """parse_report.py must output 1 when all tests pass."""
        from swe_dive.adapters.swebench import SWEBenchAdapter

        adapter = SWEBenchAdapter.__new__(SWEBenchAdapter)
        sample = {
            "instance_id": "test__test-1",
            "repo": "test/test", "base_commit": "abc",
            "problem_statement": "x",
            "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n+++ b/tests/test_x.py\n",
            "FAIL_TO_PASS": '["tests/test_x.py::test_foo"]',
            "PASS_TO_PASS": '["tests/test_x.py::test_bar"]',
            "version": "1.0", "difficulty": "easy",
        }
        task_dir = adapter.emit_task(sample, tmp_path)

        eval_spec = task_dir / "tests" / "eval_spec.json"
        pytest_output = tmp_path / "pytest_out.txt"
        pytest_output.write_text(
            "PASSED tests/test_x.py::test_foo\nPASSED tests/test_x.py::test_bar\n"
        )

        import subprocess, sys
        result = subprocess.run(
            [sys.executable, str(task_dir / "tests" / "parse_report.py"),
             "--eval-spec", str(eval_spec),
             "--pytest-output", str(pytest_output)],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == "1"

    @pytest.mark.slow
    def test_parse_report_scores_zero_on_failure(self, tmp_path):
        from swe_dive.adapters.swebench import SWEBenchAdapter

        adapter = SWEBenchAdapter.__new__(SWEBenchAdapter)
        sample = {
            "instance_id": "test__test-2",
            "repo": "test/test", "base_commit": "abc",
            "problem_statement": "x",
            "test_patch": "diff --git a/tests/t.py b/tests/t.py\n+++ b/tests/t.py\n",
            "FAIL_TO_PASS": '["tests/t.py::test_new"]',
            "PASS_TO_PASS": '["tests/t.py::test_old"]',
            "version": "1.0", "difficulty": "easy",
        }
        task_dir = adapter.emit_task(sample, tmp_path)

        eval_spec = task_dir / "tests" / "eval_spec.json"
        pytest_output = tmp_path / "pytest_out.txt"
        pytest_output.write_text(
            "FAILED tests/t.py::test_new\nPASSED tests/t.py::test_old\n"
        )

        import subprocess, sys
        result = subprocess.run(
            [sys.executable, str(task_dir / "tests" / "parse_report.py"),
             "--eval-spec", str(eval_spec),
             "--pytest-output", str(pytest_output)],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == "0"


# ---------------------------------------------------------------------------
# opencode provider config
# ---------------------------------------------------------------------------

class TestOpencodeProviderConfig:
    """_opencode_provider_config maps a LiteLLM model name to an opencode.json
    provider block. The model id it emits must match what set_config_option
    sends, or opencode rejects it with 'model not found' and silently falls
    back to its default model."""

    def _fn(self):
        pytest.importorskip("harbor")
        from swe_dive.harbor_agents.user_sim_agent import _opencode_provider_config
        return _opencode_provider_config

    def test_gemini_maps_to_google_provider(self):
        cfg = self._fn()("gemini/gemini-2.5-flash")
        assert cfg["model"] == "google/gemini-2.5-flash"
        assert cfg["small_model"] == "google/gemini-2.5-flash"
        assert cfg["provider"]["google"]["models"] == {"gemini-2.5-flash": {}}
        assert cfg["provider"]["google"]["npm"] == "@ai-sdk/google"

    def test_google_prefix_also_supported(self):
        cfg = self._fn()("google/gemini-2.5-pro")
        assert cfg["model"] == "google/gemini-2.5-pro"

    def test_unknown_provider_returns_none(self):
        assert self._fn()("cohere/command-r") is None


class TestAcpAgentModelForFlags:
    """UserSimAcpAgent._agent_model_for_flags translates the LiteLLM model name
    into the id the ACP backend expects. It must read the ``agent_model`` kwarg
    (self.model_name is typically None) and stay in sync with the opencode.json
    provider config, or opencode rejects the model as 'model not found'."""

    def _agent(self, *, agent_model, model_name, registry_spec):
        pytest.importorskip("harbor")
        from swe_dive.harbor_agents.user_sim_agent import UserSimAcpAgent
        a = UserSimAcpAgent.__new__(UserSimAcpAgent)
        a.agent_model = agent_model
        a.model_name = model_name
        a._registry_spec = registry_spec
        return a

    def test_opencode_gemini_becomes_google(self):
        a = self._agent(agent_model="gemini/gemini-2.5-flash",
                        model_name=None, registry_spec="opencode")
        assert a._agent_model_for_flags() == "google/gemini-2.5-flash"

    def test_reads_kwarg_when_model_name_none(self):
        a = self._agent(agent_model="anthropic/claude-sonnet-4",
                        model_name=None, registry_spec="opencode")
        assert a._agent_model_for_flags() == "anthropic/claude-sonnet-4"

    def test_pi_acp_azure_becomes_azure_foundry(self):
        a = self._agent(agent_model="azure/gpt-5",
                        model_name=None, registry_spec="pi-acp")
        assert a._agent_model_for_flags() == "azure-foundry/gpt-5"

    def test_gemini_not_translated_for_non_opencode(self):
        a = self._agent(agent_model="gemini/gemini-2.5-flash",
                        model_name=None, registry_spec="pi-acp")
        assert a._agent_model_for_flags() == "gemini/gemini-2.5-flash"

    def test_none_when_no_model(self):
        a = self._agent(agent_model=None, model_name=None, registry_spec="opencode")
        assert a._agent_model_for_flags() is None


# ---------------------------------------------------------------------------
# Auth loader
# ---------------------------------------------------------------------------

# (Auth tests removed — swe_dive.config was deleted in favour of Harbor-
# idiomatic shell env vars; credentials no longer need a custom loader.)
