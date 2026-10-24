"""Host-side Harbor agent entrypoints that install and run the orchestrator.

These classes live on the **host** and are invoked by Harbor's trial runner.
They are responsible for:
  1. Installing the orchestrator's dependencies into the Harbor sandbox
     (swe_dive package, LiteLLM, OpenHands SDK, Node/npm for ACP agents, …).
  2. Writing the task instruction into the container.
  3. Exec-ing ``swe_dive.sandbox.orchestrator`` inside the sandbox, which runs the
     user-simulator loop driving a ``swe_dive.coding_agents`` backend.

Contrast with ``swe_dive.coding_agents`` (the *in-container* session backends)
and ``swe_dive.sandbox.orchestrator`` (the *in-container* multi-turn loop).

**Auth** is entirely Harbor-idiomatic: credentials live in the shell environment
(``OPENAI_API_KEY``, ``LLM_MODEL``, etc.). Harbor's ``BaseAgent._get_env``
resolves from ``AgentConfig.env`` (extra_env) first, then ``os.environ``.
We forward only the LLM credential variables explicitly via
``exec_as_agent(env=…)`` so the in-container orchestrator finds them.

**Simulator config** (persona, coding_mode, max_turns) is supplied as
first-class constructor kwargs — ``AgentConfig.kwargs`` in job configs and
``main.py`` both map to these directly, so they appear verbatim in Harbor's
recorded config rather than being buried in opaque env-var strings.

**swe_dive package installation**: because swe_dive is not published on PyPI,
``install()`` builds a tarball of the local source tree at runtime and uploads
it to the container via ``environment.upload_file``, then installs from that
tarball.  Set ``SWE_DIVE_PIP_SPEC`` to a ``git+…`` URL to override this for
CI/production deployments.

**Trajectory parity**: ``populate_context_post_run()`` is implemented in
``_SimConfigMixin`` so both agents convert the orchestrator's
``conversation.json`` into an ATIF ``trajectory.json`` after each trial.
This gives ``harbor view`` full turn-by-turn visibility and puts our agents
on parity with Harbor's built-in agents.
"""

import json
import os
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import override

# Path to the TypeScript extension that gets uploaded into the container.
_PI_AZURE_FOUNDRY_EXTENSION_PATH = Path(__file__).parent / "azure_foundry_extension.ts"

from harbor.agents.installed.acp import AcpAgent
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.models.trial.result import AgentInfo, ModelInfo

_OPENHANDS_VENV       = "/opt/swe-dive-venv"   # venv created by UserSimOpenHandsAgent
_ORCH_LOG_DIR         = "/logs/agent"
_INSTRUCTION_PATH     = "/logs/agent/instruction.md"
_SWE_DIVE_CONTAINER_PATH = "/tmp/swe_dive_src.tar.gz"
_SWE_DIVE_PYTHONPATH     = "/tmp"

# ── LLM credential env vars forwarded into the container ────────────────────
_LLM_ENV_KEYS = [
    "LLM_MODEL", "AGENT_LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL",
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_VERSION",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION", "AZURE_MODEL",
    "LLM_API_VERSION",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME",
    "GITHUB_TOKEN", "GH_TOKEN", "COPILOT_GITHUB_TOKEN",
]

# Repo root — two levels up from this file:
#   swe-dive/swe_dive/harbor_agents/user_sim_agent.py
#               ↑ parents[1]         ↑ parents[2]
_REPO_ROOT = Path(__file__).parents[2]


# ── swe_dive source upload ───────────────────────────────────────────────────
# swe_dive is not on PyPI.  Rather than packaging it, we tar just the source
# directory, upload it to the container, extract it to /tmp/, and add /tmp to
# PYTHONPATH so `python -m swe_dive.sandbox.orchestrator` works without a pip install.
# This avoids package build tools and pip resolution for the swe_dive code
# itself; only its *dependencies* (litellm, openhands-sdk, …) go through pip.


def _tarball_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Exclude build artefacts and caches from the uploaded source tarball."""
    skip = {"__pycache__", ".venv", ".git"}
    for part in tarinfo.name.split("/"):
        if part in skip or part.endswith((".pyc", ".egg-info")):
            return None
    return tarinfo


async def _upload_swe_dive_source(
    agent: BaseInstalledAgent, environment: BaseEnvironment
) -> None:
    """Tar up ``swe_dive/sandbox/`` and upload it to the container.

    The tar contains two entries so that ``import swe_dive.sandbox.*`` works
    with ``PYTHONPATH=/tmp`` inside the container:

      /tmp/swe_dive/__init__.py    (minimal; makes swe_dive a proper package)
      /tmp/swe_dive/sandbox/…      (the full sandbox subpackage)
    """
    # swe_dive/__init__.py on the host (two levels up from harbor_agents/)
    swe_dive_init = _REPO_ROOT / "swe_dive" / "__init__.py"
    sandbox_dir   = _REPO_ROOT / "swe_dive" / "sandbox"

    fd, tmp = tempfile.mkstemp(suffix=".tar.gz", prefix="swe_dive_sandbox_")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(swe_dive_init, arcname="swe_dive/__init__.py")
            tar.add(sandbox_dir,   arcname="swe_dive/sandbox", filter=_tarball_filter)
        await environment.upload_file(
            source_path=tmp_path,
            target_path=_SWE_DIVE_CONTAINER_PATH,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _llm_env(agent: BaseInstalledAgent) -> dict[str, str]:
    """Return LLM credentials present on the host, ready to inject into exec."""
    env: dict[str, str] = {}
    for key in _LLM_ENV_KEYS:
        val = agent._get_env(key)
        if val:
            env[key] = val
    if agent.model_name:
        env.setdefault("LLM_MODEL", agent.model_name)
        # Also propagate as AGENT_LLM_MODEL so the coding agent backend knows
        # which model to use.  acp_agent.py translates LiteLLM-style provider
        # prefixes (e.g. "azure/gpt-5.4") to each agent's own format before
        # calling set_config_option, so this is safe for ACP agents as well.
        env.setdefault("AGENT_LLM_MODEL", agent.model_name)
    return env


async def _write_instruction(
    agent: BaseInstalledAgent, environment: BaseEnvironment, instruction: str
) -> None:
    await agent.exec_as_agent(
        environment,
        command=(
            f"mkdir -p {_ORCH_LOG_DIR} && "
            f"cat > {_INSTRUCTION_PATH} <<'SWE_DIVE_EOF'\n"
            f"{instruction}\n"
            "SWE_DIVE_EOF"
        ),
    )


def _opencode_provider_config(model_name: str) -> dict | None:
    """Build an opencode.json config for the given LiteLLM model name.

    opencode in headless ACP mode reads its standard config files, so we can
    pre-configure a custom provider that points at whatever LLM backend the
    job is using.  The returned dict uses opencode's ``{env:VAR}`` interpolation
    so credentials are resolved by opencode at runtime rather than being
    embedded in the config at install time.

    Returns None when the model name cannot be mapped to a supported provider.
    """
    if not model_name:
        return None
    provider_prefix, _, model_id = model_name.partition("/")
    if not model_id:
        return None

    if provider_prefix in ("azure", "azure-openai-responses"):
        # Use the bundled @ai-sdk/azure package, which opencode recognises
        # natively and handles correctly:
        #   • constructs the correct deployment URL:
        #     {baseURL}/openai/deployments/{modelId}/chat/completions
        #   • appends ?api-version={apiVersion} as a query parameter
        #   • uses max_completion_tokens (not max_tokens) for newer models
        # AZURE_API_BASE and AZURE_API_VERSION are not secrets so they are
        # resolved and normalised at install time.  Only AZURE_API_KEY stays
        # as {env:…} so the secret is never written to disk.
        azure_base = os.environ.get("AZURE_API_BASE", "").rstrip("/")
        azure_version = os.environ.get("AZURE_API_VERSION", "2025-04-01-preview")
        # @ai-sdk/azure constructs deployment URLs as:
        #   {baseURL}/deployments/{modelId}/chat/completions?api-version=...
        # When using resourceName it builds baseURL as
        #   https://{resourceName}.openai.azure.com/openai
        # so we must append /openai to the raw AZURE_API_BASE endpoint to
        # match that convention.
        base_url = f"{azure_base}/openai" if azure_base else None
        azure_version = os.environ.get("AZURE_API_VERSION", "2025-04-01-preview")
        # Normalise the opencode model id: opencode uses "azure/<deployment>"
        # regardless of whether the LiteLLM prefix was "azure" or
        # "azure-openai-responses" (the latter is LiteLLM-specific for the
        # Responses API; opencode talks to the same deployment via @ai-sdk/azure
        # with useCompletionUrls=true and doesn't need the distinction).
        opencode_model = f"azure/{model_id}"
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": opencode_model,
            # Point the small/title agent at the same deployment so opencode
            # does not fall back to a non-existent claude-haiku model.
            "small_model": opencode_model,
            "provider": {
                "azure": {
                    "npm": "@ai-sdk/azure",
                    "options": {
                        "baseURL": base_url,
                        "apiKey": "{env:AZURE_API_KEY}",
                        "apiVersion": azure_version,
                        # Force deployment-based URL:
                        #   {baseURL}/deployments/{model}/chat/completions
                        # Without this @ai-sdk/azure uses the OpenAI-compat
                        # path {baseURL}/v1/chat/completions which Azure rejects
                        # with 'API version not supported'.
                        "useDeploymentBasedUrls": True,
                        # Use chat/completions endpoint (not Responses API).
                        "useCompletionUrls": True,
                    },
                    "models": {model_id: {}},
                }
            },
        }

    if provider_prefix == "anthropic":
        # Anthropic's built-in provider reads ANTHROPIC_API_KEY from the env.
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": model_name,
        }

    if provider_prefix == "openai":
        # OpenAI's built-in provider reads OPENAI_API_KEY from the env.
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": model_name,
        }

    if provider_prefix in ("gemini", "google"):
        # opencode's Google provider uses the @ai-sdk/google package and refers
        # to models as "google/<model_id>".  The LiteLLM prefix is "gemini/" (or
        # "google/"); normalise both to opencode's "google/" so set_config_option
        # matches the provider model id below (otherwise opencode rejects the
        # model with "model not found" and silently falls back to its default).
        opencode_model = f"google/{model_id}"
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": opencode_model,
            # Keep the small/title agent on the same model so opencode does not
            # fall back to a non-existent default (e.g. a claude-haiku model).
            "small_model": opencode_model,
            "provider": {
                "google": {
                    "npm": "@ai-sdk/google",
                    "options": {
                        # @ai-sdk/google defaults to GOOGLE_GENERATIVE_AI_API_KEY;
                        # our container forwards the credential as GOOGLE_API_KEY.
                        "apiKey": "{env:GOOGLE_API_KEY}",
                    },
                    "models": {model_id: {}},
                }
            },
        }

    return None


# ── Simulator config mixin ────────────────────────────────────────────────────

class _SimConfigMixin:
    """First-class simulator config kwargs + ATIF trajectory population.

    Shared by ``UserSimOpenHandsAgent`` and ``UserSimAcpAgent``.

    **Constructor kwargs** (``AgentConfig.kwargs`` in job configs):
      - ``persona``      — user-simulator persona (EXPERT_NITPICKER, …)
      - ``coding_mode``  — COLLABORATIVE | VIBE_CODING
      - ``max_turns``    — max turns before the simulator gives up
      - ``max_retries``  — extra whole-trial attempts (fresh models, fresh backend session, clean workdir) when a trial fails for reasons unrelated to the coding agent (e.g. repeated user-simulator parse/validation failures)
      - ``debug``        — pass ``--debug`` to the in-container orchestrator

    **``populate_context_post_run``** converts the orchestrator's
    ``conversation.json`` (legacy-format conversation log) into an ATIF
    ``trajectory.json`` after the trial completes, giving ``harbor view``
    full turn-by-turn visibility — the same parity as Harbor's built-in agents.
    The conversation.json itself is kept unchanged so it remains human-readable
    and backward-compatible with the legacy ``user_sim/conversations/`` format.
    """

    def __init__(
        self,
        *args,
        persona: str = "EXPERT_NITPICKER",
        coding_mode: str = "COLLABORATIVE",
        max_turns: int = 10,
        max_retries: int = 2,
        debug: bool = False,
        sim_model: str | None = None,        # ← add
        validation_model: str | None = None, # ← add
        agent_model: str | None = None,      # ← add
        **kwargs,
    ) -> None:
        self.sim_model = sim_model
        self.validation_model = validation_model
        self.agent_model = agent_model
        self.persona = persona
        self.coding_mode = coding_mode
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.debug = debug
        super().__init__(*args, **kwargs)

    def _agent_model_for_flags(self) -> str | None:
        """Model id passed as ``--agent-model`` to the orchestrator.

        Defaults to the ``agent_model`` kwarg (which, unlike ``self.model_name``,
        is always populated for these agents).  Subclasses override this to
        translate the LiteLLM-style model name into the identifier the specific
        coding-agent backend expects (e.g. opencode's ``google/`` provider).
        """
        return self.agent_model or self.model_name

    def _orchestrator_sim_flags(self) -> str:
        flags = (
            f"--persona {shlex.quote(self.persona)} "
            f"--coding-mode {shlex.quote(self.coding_mode)} "
            f"--max-turns {self.max_turns} "
            f"--max-retries {self.max_retries}"
        )
        if self.sim_model:
            flags += f" --sim-model {shlex.quote(self.sim_model)}"
        if self.validation_model:
            flags += f" --validation-model {shlex.quote(self.validation_model)}"
        agent_model = self._agent_model_for_flags()
        if agent_model:
            flags += f" --agent-model {shlex.quote(agent_model)}"
        if self.debug:
            flags += " --debug"
        return flags

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Convert conversation.json → ATIF trajectory.json after the trial.

        Reads the orchestrator's ``conversation.json`` (written by
        ``swe_dive.sandbox.orchestrator``), builds an ATIF trajectory from each
        simulator turn, overwrites ``trajectory.json`` with the ATIF version
        so ``harbor view`` can render it, and populates context token counts,
        cost, and artifacts.
        """
        conv_path = self.logs_dir / "conversation.json"
        if not conv_path.exists():
            self.logger.debug("No conversation.json found at %s", conv_path)
            return

        try:
            conv = json.loads(conv_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Failed to read conversation.json: %s", exc)
            return

        sim_response: list[dict] = conv.get("sim_response", [])
        steps: list[Step] = []

        for turn in sim_response:
            user_msg: str = turn.get("user_message", "")
            agent_msg: str = turn.get("agent_response", "") or ""
            agent_diff: str = turn.get("agent_changes", "") or ""
            raw_tool_calls: list[dict] = turn.get("tool_calls") or []

            if user_msg:
                steps.append(Step(
                    step_id=len(steps) + 1,
                    source="user",
                    message=user_msg,
                ))

            if agent_msg or agent_diff or raw_tool_calls:
                # Build ATIF ToolCall objects; track IDs of those that succeed
                # so observation source_call_ids only reference real entries
                # (Trajectory.validate_tool_call_references enforces this).
                atif_tool_calls: list[ToolCall] | None = None
                built_ids: set[str] = set()
                if raw_tool_calls:
                    built: list[ToolCall] = []
                    for tc in raw_tool_calls:
                        try:
                            call_id = str(tc.get("tool_call_id") or "")
                            built.append(ToolCall(
                                tool_call_id=call_id,
                                function_name=str(tc.get("function_name") or "unknown"),
                                arguments=tc.get("arguments") or {},
                            ))
                            built_ids.add(call_id)
                        except Exception:  # noqa: BLE001
                            pass
                    atif_tool_calls = built or None

                # Observation: per-tool-call outputs (source_call_id only set
                # when the ToolCall was actually built) + cumulative git diff.
                obs_results: list[ObservationResult] = []
                for tc in raw_tool_calls:
                    output = tc.get("output")
                    if output is not None:
                        call_id = str(tc.get("tool_call_id") or "") or None
                        # Only link to the tool call if its ToolCall was built.
                        if call_id not in built_ids:
                            call_id = None
                        obs_results.append(ObservationResult(
                            source_call_id=call_id,
                            content=str(output),
                        ))
                if agent_diff:
                    obs_results.append(ObservationResult(content=agent_diff))
                obs = Observation(results=obs_results) if obs_results else None

                steps.append(Step(
                    step_id=len(steps) + 1,
                    source="agent",
                    message=agent_msg or "(no message)",
                    tool_calls=atif_tool_calls,
                    observation=obs,
                ))

        if not steps:
            self.logger.debug("No steps extracted from conversation.json")
            return

        # ── Usage / cost from sim_usage recorded by the orchestrator ────────────
        sim_usage: dict = conv.get("sim_usage") or {}
        n_input: int | None = sim_usage.get("input_tokens") or None
        n_output: int | None = sim_usage.get("output_tokens") or None
        cost: float | None = sim_usage.get("cost_usd") or None

        trajectory = Trajectory(
            schema_version="ATIF-v1.7",
            session_id=self.logs_dir.name,
            agent=Agent(
                name=self.to_agent_info().name,
                version=self.version() or "unknown",
                model_name=conv.get("agent_model") or self.model_name,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_steps=len(steps),
                total_prompt_tokens=n_input,
                total_completion_tokens=n_output,
                total_cost_usd=cost,
            ),
        )

        traj_path = self.logs_dir / "trajectory.json"
        try:
            traj_path.write_text(
                json.dumps(trajectory.to_json_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.logger.debug(
                "Wrote ATIF trajectory (%d steps) to %s", len(steps), traj_path
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Failed to write trajectory.json: %s", exc)

        # ── Populate AgentContext for Harbor result.json ────────────────────────
        context.n_input_tokens = n_input
        context.n_output_tokens = n_output
        context.cost_usd = cost

        # ── Write artifacts (git diff) so Harbor view shows them ───────────────
        # logs_dir is the trial's agent/ subdir; artifacts/ lives one level up.
        artifacts_dir = self.logs_dir.parent / "artifacts"
        try:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            manifest_entries: list[dict] = []

            git_diff: str = conv.get("git_diff") or ""
            if git_diff:
                diff_path = artifacts_dir / "git_diff.txt"
                diff_path.write_text(git_diff, encoding="utf-8")
                manifest_entries.append({
                    "source": "/testbed",
                    "destination": "artifacts/git_diff.txt",
                    "type": "file",
                    "status": "ok",
                    "service": None,
                })

            # conversation.json as a top-level artifact for easy access
            conv_src = self.logs_dir / "conversation.json"
            if conv_src.exists():
                shutil.copy2(conv_src, artifacts_dir / "conversation.json")
                manifest_entries.append({
                    "source": "/logs/agent/conversation.json",
                    "destination": "artifacts/conversation.json",
                    "type": "file",
                    "status": "ok",
                    "service": None,
                })

            if manifest_entries:
                (artifacts_dir / "manifest.json").write_text(
                    json.dumps(manifest_entries, indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Failed to write artifacts: %s", exc)

        # ── Expose orchestrator output in Harbor view's agent-logs panel ────────
        # Harbor view reads agent/command-N/stdout.txt for the "Commands" log.
        # command-0 = full orchestrator log (symlink → orchestrator.txt).
        # command-1 = user-sim-only log    (symlink → user_sim.log).
        orch_path = self.logs_dir / "orchestrator.txt"
        cmd0_dir  = self.logs_dir / "command-0"
        cmd0_path = cmd0_dir / "stdout.txt"
        if orch_path.exists() and not cmd0_path.exists():
            try:
                cmd0_dir.mkdir(parents=True, exist_ok=True)
                cmd0_path.symlink_to("../orchestrator.txt")
                self.logger.debug("Linked command-0/stdout.txt → orchestrator.txt")
            except Exception as exc:  # noqa: BLE001
                self.logger.debug("Failed to link command-0/stdout.txt: %s", exc)

        sim_log_path = self.logs_dir / "user_sim.log"
        cmd1_dir  = self.logs_dir / "command-1"
        cmd1_path = cmd1_dir / "stdout.txt"
        if sim_log_path.exists() and not cmd1_path.exists():
            try:
                cmd1_dir.mkdir(parents=True, exist_ok=True)
                cmd1_path.symlink_to("../user_sim.log")
                self.logger.debug("Linked command-1/stdout.txt → user_sim.log")
            except Exception as exc:  # noqa: BLE001
                self.logger.debug("Failed to link command-1/stdout.txt: %s", exc)


# ── Concrete Harbor agents ────────────────────────────────────────────────────

class UserSimOpenHandsAgent(_SimConfigMixin, BaseInstalledAgent):
    """Harbor agent: installs OpenHands SDK + swe_dive, runs orchestrator.

    Host-side entry point — Harbor calls install() then run() per trial.
    Inside the sandbox the orchestrator holds a persistent OpenHands SDK
    Conversation across all simulator turns.

    Conversation log: ``jobs/<job>/<trial>/agent/conversation.json`` (legacy
    format matching ``user_sim/conversations/*.json``).
    ATIF trajectory: ``jobs/<job>/<trial>/agent/trajectory.json`` (written by
    ``populate_context_post_run``; visible in ``harbor view``).
    """

    SUPPORTS_ATIF = True

    @staticmethod
    @override
    def name() -> str:
        return "swe-dive-user-sim-openhands"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y git curl build-essential",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        # Upload swe_dive source; it will be extracted in the install command.
        await _upload_swe_dive_source(self, environment)
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "curl -LsSf https://astral.sh/uv/install.sh | sh && "
                'if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi && '
                "uv python install 3.12 && "
                f"uv venv {_OPENHANDS_VENV} --seed --python 3.12 && "
                f"{_OPENHANDS_VENV}/bin/pip install --upgrade pip && "
                f"{_OPENHANDS_VENV}/bin/pip install "
                f"'openhands-sdk>=1.24' 'openhands-tools>=1.24' 'litellm' 'tenacity' && "
                f"tar xzf {_SWE_DIVE_CONTAINER_PATH} -C {_SWE_DIVE_PYTHONPATH}"
            ),
        )

    @with_prompt_template
    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        await _write_instruction(self, environment, instruction)
        await self.exec_as_agent(
            environment,
            command=(
                f"{_OPENHANDS_VENV}/bin/python -m swe_dive.sandbox.orchestrator "
                f"--backend openhands "
                f"--instruction-file {_INSTRUCTION_PATH} "
                f"--workdir /testbed --log-dir {_ORCH_LOG_DIR} "
                f"{self._orchestrator_sim_flags()} "
                f"2>&1 | tee {_ORCH_LOG_DIR}/orchestrator.txt"
            ),
            env={**_llm_env(self), "PYTHONPATH": _SWE_DIVE_PYTHONPATH},
        )


class UserSimAcpAgent(_SimConfigMixin, AcpAgent):
    """Harbor agent: reuses Harbor's AcpAgent install, runs orchestrator (ACP backend).

    Host-side entry point — Harbor calls install() then run() per trial.
    install() inherits Harbor's AcpAgent (registry resolution + launcher +
    acp venv + acp_runner.py), then adds swe_dive + LiteLLM.
    Inside the sandbox the orchestrator holds a persistent ACP session across
    all simulator turns.
    """

    SUPPORTS_ATIF = True

    @staticmethod
    @override
    def name() -> str:
        return "swe-dive-user-sim-acp"

    def _acp_agent_name(self) -> str:
        """Return a name that identifies both the sim framework and the specific ACP agent."""
        spec = getattr(self, "_registry_spec", None)
        return f"swe-dive-user-sim-acp/{spec}" if spec else "swe-dive-user-sim-acp"

    @override
    def to_agent_info(self):
        return AgentInfo(
            name=self._acp_agent_name(),
            version=self.version() or "unknown",
            model_info=(
                ModelInfo(
                    name=self._parsed_model_name,
                    provider=self._parsed_model_provider,
                )
                if self._parsed_model_name
                else None
            ),
        )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)

        # Install Node.js 22 from the official tarball — more reliable than
        # apt/NodeSource across SWE-bench images (avoids package-repo conflicts).
        # pi v0.80+ requires Node.js 22+.
        await self.exec_as_root(
            environment,
            command=(
                "curl -fsSL https://nodejs.org/dist/v22.13.0/node-v22.13.0-linux-x64.tar.xz "
                "-o /tmp/node.tar.xz && "
                "tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 && "
                "rm /tmp/node.tar.xz && "
                "node --version"
            ),
        )

        # pi-acp is an ACP adapter that internally spawns the `pi` binary.
        # Install/upgrade it globally so the container uses the latest pi
        # version (which has gpt-5.4 in its Azure model list). The registry spec
        # is "acp:pi-acp" so we check for the "pi-acp" suffix rather than
        # an exact match on the bare spec name.
        if "pi-acp" in (self._registry_spec or ""):
            await self.exec_as_root(
                environment,
                command="npm install -g @earendil-works/pi-coding-agent",
            )
            # Upload the pi extension that registers the "azure-foundry" provider.
            # Uses Azure Chat Completions with ?api-version as a query param —
            # the only endpoint+auth combo that works for this Azure resource.
            await environment.upload_file(
                source_path=str(_PI_AZURE_FOUNDRY_EXTENSION_PATH),
                target_path="/tmp/azure-foundry-ext.ts",
            )
            await self.exec_as_agent(
                environment,
                command=(
                    'mkdir -p "$HOME/.pi/agent/extensions" && '
                    'cp /tmp/azure-foundry-ext.ts "$HOME/.pi/agent/extensions/azure-foundry.ts"'
                ),
            )

        await _upload_swe_dive_source(self, environment)
        await self.exec_as_agent(
            environment,
            command=(
                f"{self._RUNNER_VENV_PATH}/bin/pip install 'litellm' 'tenacity' && "
                f"tar xzf {_SWE_DIVE_CONTAINER_PATH} -C {_SWE_DIVE_PYTHONPATH}"
            ),
        )

        # Write opencode.json so opencode can authenticate with the LLM provider
        # in headless ACP mode.  opencode reads its config even when started
        # via `opencode acp`, but has no credentials unless we configure them
        # here.  We use opencode's {env:VAR} interpolation so secrets are
        # resolved at runtime from the container env rather than being stored
        # in the config file.
        if (self._registry_spec or "").strip() == "opencode":
            # Use the effective agent model (the ``agent_model`` kwarg is the
            # source of truth here; ``self.model_name`` is often None).
            cfg = _opencode_provider_config((self.agent_model or self.model_name) or "")
            if cfg:
                cfg_json = json.dumps(cfg, indent=2)
                await self.exec_as_agent(
                    environment,
                    command=(
                        'mkdir -p "$HOME/.config/opencode" && '
                        'cat > "$HOME/.config/opencode/opencode.json" '
                        "<<'OPENCODE_CFG_EOF'\n"
                        f"{cfg_json}\n"
                        "OPENCODE_CFG_EOF"
                    ),
                )

    @with_prompt_template
    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        await _write_instruction(self, environment, instruction)
        # The --agent-model flag (and its per-backend translation) is emitted by
        # _orchestrator_sim_flags() → _agent_model_for_flags() below, so both
        # openhands and acp backends share a single source of truth.
        await self.exec_as_agent(
            environment,
            command=(
                f"{self._RUNNER_VENV_PATH}/bin/python -m swe_dive.sandbox.orchestrator "
                f"--backend acp "
                f"--instruction-file {_INSTRUCTION_PATH} "
                f"--workdir /testbed --log-dir {_ORCH_LOG_DIR} "
                f"--acp-launcher {self._LAUNCHER_REMOTE_PATH} "
                f"--acp-runner-dir /installed-agent "
                f"{self._orchestrator_sim_flags()} "
                f"2>&1 | tee {_ORCH_LOG_DIR}/orchestrator.txt"
            ),
            env={**_llm_env(self), "PYTHONPATH": _SWE_DIVE_PYTHONPATH},
        )

    def _agent_model_for_flags(self) -> str | None:
        # The ACP backend drives a specific coding agent (pi-acp, opencode, …)
        # that expects its own model identifier, which differs from the
        # LiteLLM-style name Harbor passes.  Translate here so that both
        # --agent-model and the opencode.json provider config agree; otherwise
        # opencode rejects the model with "model not found" and silently falls
        # back to its default.
        #   • pi-acp:   azure/X                 → azure-foundry/X
        #   • any acp:  azure-openai-responses/X → azure/X
        #   • opencode: gemini/X                → google/X
        model = self.agent_model or self.model_name
        if not model:
            return None
        spec = (self._registry_spec or "").strip()
        if model.startswith("azure/") and "pi-acp" in spec:
            return "azure-foundry/" + model[len("azure/"):]
        if model.startswith("azure-openai-responses/"):
            return "azure/" + model[len("azure-openai-responses/"):]
        if spec == "opencode" and model.startswith("gemini/"):
            return "google/" + model[len("gemini/"):]
        return model
