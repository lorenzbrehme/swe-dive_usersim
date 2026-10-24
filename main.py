"""
Run SWE-Dive eval using the Harbor Python SDK.
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from harbor.job import Job
    from harbor.models.job.config import JobConfig
except ImportError as exc:
    sys.exit(
        f"Harbor SDK not found ({exc}).\n"
        "Install it:  uv tool install harbor\n"
        "Then run:    uv run --with harbor main.py"
    )

# Load .env (the project's existing credentials file) before reading any env vars.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # shell env wins; .env fills gaps
except ImportError:
    pass  # dotenv not installed; rely on shell env only

# If LLM_MODEL is still unset, try to derive it from provider-specific vars.
# This covers the common case where .env has AZURE_MODEL / OPENAI_MODEL etc.
# but not the unified LLM_MODEL that LiteLLM / the orchestrator need.
def _maybe_set_llm_model() -> None:
    if os.getenv("LLM_MODEL"):
        return
    checks = [
        # Keep azure/ prefix for the LiteLLM simulator — it constructs the
        # deployment-specific chat/completions URL which works at this endpoint.
        # OPENAI_BASE_URL / OPENAI_API_KEY are set separately so pi can use the
        # endpoint-level Responses API without affecting the simulator.
        ("azure",     ["AZURE_API_KEY"],     "AZURE_MODEL",     "azure/{}"),
        ("openai",    ["OPENAI_API_KEY"],    "OPENAI_MODEL",    "openai/{}"),
        ("anthropic", ["ANTHROPIC_API_KEY"], "ANTHROPIC_MODEL", "anthropic/{}"),
        ("gemini",    ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "GEMINI_MODEL", "gemini/{}"),
        ("openrouter",["OPENROUTER_API_KEY"],"OPENROUTER_MODEL","{}"),
    ]
    for _name, cred_keys, model_key, fmt in checks:
        if any(os.getenv(k) for k in cred_keys):
            model = os.getenv(model_key)
            if model:
                os.environ["LLM_MODEL"] = fmt.format(model)
                # For Azure: also populate openai-SDK vars so pi and other ACP
                # agents can call the Azure Responses API endpoint directly.
                # The LiteLLM simulator continues to use the azure/ provider
                # (reads AZURE_API_BASE) so setting OPENAI_BASE_URL doesn't
                # affect it.
                if _name == "azure":
                    azure_base = os.getenv("AZURE_API_BASE", "").rstrip("/")
                    azure_key  = os.getenv("AZURE_API_KEY", "")
                    azure_ver  = os.getenv("AZURE_API_VERSION", "")
                    if azure_base and not os.getenv("OPENAI_BASE_URL"):
                        # Endpoint-level Responses API base path.
                        # pi's openai provider appends /responses → correct URL.
                        os.environ["OPENAI_BASE_URL"] = f"{azure_base}/openai"
                    if azure_key and not os.getenv("OPENAI_API_KEY"):
                        os.environ["OPENAI_API_KEY"] = azure_key
                    if azure_ver and not os.getenv("OPENAI_API_VERSION"):
                        os.environ["OPENAI_API_VERSION"] = azure_ver
                return

_maybe_set_llm_model()

DEFAULT_CONFIG = "jobs/swebench-copilot.yaml"

AGENT_IMPORT_PATHS = {
    "openhands": "swe_dive.harbor_agents.user_sim_agent:UserSimOpenHandsAgent",
    "acp":       "swe_dive.harbor_agents.user_sim_agent:UserSimAcpAgent",
}

AGENT_NAMES = {
    "openhands": "swe-dive-user-sim-openhands",
    "acp":       "swe-dive-user-sim-acp",  # base; overridden per-spec below
}


def _acp_agent_name(acp_spec: str) -> str:
    """Return a display name that embeds the specific ACP agent spec."""
    spec_id = acp_spec.removeprefix("acp:")
    return f"swe-dive-user-sim-acp/{spec_id}"


# ── YAML loading ─────────────────────────────────────────────────────────────

def load_config_from_yaml(path: Path) -> JobConfig:
    """Load a JobConfig from a YAML file using Harbor's own Pydantic model."""
    try:
        import yaml  # pyyaml; transitively installed by harbor
    except ImportError:
        sys.exit(
            "pyyaml is required to load YAML configs.\n"
            "Install it:  pip install pyyaml"
        )
    data = yaml.safe_load(path.read_text())
    return JobConfig.model_validate(data)


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="swe_dive user-simulator eval on Harbor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config file (mutually exclusive with per-flag job construction) ──────
    p.add_argument(
        "-c", "--config", metavar="YAML",
        default=DEFAULT_CONFIG,
        help=(
            "Path to a Harbor job YAML (e.g. jobs/swebench-openhands.yaml). "
            "This is the base config; any other flags provided on the "
            "command line will override the corresponding fields in the YAML. "
            f"(default: {DEFAULT_CONFIG})"
        ),
    )

    # ── Task selection ───────────────────────────────────────────────────────
    p.add_argument(
        "--tasks", nargs="*", metavar="PATTERN",
        default=None,
        help="SWE-bench instance IDs or fnmatch patterns (e.g. 'astropy__*').",
    )
    p.add_argument(
        "--n-tasks", type=int, default=None, metavar="N",
        help="Cap on number of tasks (applied after --tasks filtering).",
    )
    p.add_argument(
        "--dataset", default=None,
        help="Harbor package dataset name.",
    )

    # ── Agent ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--agent", choices=list(AGENT_IMPORT_PATHS), default=None,
    )
    p.add_argument(
        "--acp-spec", default=None, metavar="SPEC",
        help="ACP registry spec used when --agent=acp.",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--model", default=None, metavar="MODEL",
        help="LiteLLM model string (e.g. openai/gpt-4o). Overrides $LLM_MODEL.",
    )

    # ── Simulator config ─────────────────────────────────────────────────────
    p.add_argument(
        "--persona", default=None,
        choices=["EXPERT_NITPICKER", "Vague_Requester", "MIND_CHANGER"],
    )
    p.add_argument(
        "--coding-mode", default=None,
        choices=["COLLABORATIVE", "VIBE_CODING"],
    )
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=None)

    # ── Job-level overrides (apply whether or not -c is given) ───────────────
    p.add_argument("--concurrency", type=int, default=None, metavar="N",
                   help="Max concurrent trials. Overrides the value in -c if set.")
    p.add_argument("--jobs-dir", default=None, metavar="DIR",
                   help="Directory for job results (default: jobs/).")
    p.add_argument("--job-name", default=None, metavar="NAME")
    p.add_argument("--job-suffix", default=None, metavar="SUFFIX",
                   help="Append __SUFFIX to the auto-generated timestamp job name.")
    p.add_argument("--debug", action="store_true",
                   help="Enable Harbor debug logging (exec output, timing, etc.).")

    return p


def apply_overrides(config: JobConfig, args) -> JobConfig:
    """Apply all explicitly-provided CLI flags as overrides onto a YAML config."""
    overrides: dict = {}

    # ── Job-level ────────────────────────────────────────────────────────────
    if args.debug:
        overrides["debug"] = True
    if args.jobs_dir is not None:
        overrides["jobs_dir"] = Path(args.jobs_dir)

    override_job_name = None
    if args.job_name is not None:
        override_job_name = args.job_name
    elif args.job_suffix is not None:
        override_job_name = f"{datetime.now().strftime("%Y-%m-%d__%H-%M-%S")}__{args.job_suffix}"
    if override_job_name is not None:
        overrides["job_name"] = override_job_name
    if args.concurrency is not None:
        overrides["n_concurrent_trials"] = args.concurrency

    # ── Dataset / task overrides ─────────────────────────────────────────────
    ds_updates: dict = {}
    if args.dataset is not None:
        ds_updates["name"] = args.dataset
    if args.tasks is not None:
        ds_updates["task_names"] = args.tasks
    if args.n_tasks is not None:
        ds_updates["n_tasks"] = args.n_tasks
    if ds_updates and config.datasets:
        overrides["datasets"] = [
            ds.model_copy(update=ds_updates) for ds in config.datasets
        ]

    # ── Agent / model / simulator overrides ──────────────────────────────────
    agent_updates: dict = {}
    if args.model is not None:
        agent_updates["model_name"] = args.model
    elif os.getenv("LLM_MODEL"):
        # Fall back to env var when no explicit --model flag was given
        agent_updates["model_name"] = os.getenv("LLM_MODEL")
    if args.agent is not None:
        agent_updates["import_path"] = AGENT_IMPORT_PATHS[args.agent]
        agent_updates["name"] = AGENT_NAMES[args.agent]  # may be refined below

    kwargs_updates: dict = {}
    if args.persona is not None:
        kwargs_updates["persona"] = args.persona
    if args.coding_mode is not None:
        kwargs_updates["coding_mode"] = args.coding_mode
    if args.max_turns is not None:
        kwargs_updates["max_turns"] = args.max_turns
    if args.max_retries is not None:
        kwargs_updates["max_retries"] = args.max_retries
    if args.acp_spec is not None:
        kwargs_updates["registry_spec"] = args.acp_spec
    if args.debug:
        kwargs_updates["debug"] = True

    if (agent_updates or kwargs_updates) and config.agents:
        new_agents = []
        for agent in config.agents:
            updates = dict(agent_updates)
            if kwargs_updates:
                merged_kwargs = dict(agent.kwargs or {})
                merged_kwargs.update(kwargs_updates)
                updates["kwargs"] = merged_kwargs
            # If the effective agent is ACP, derive the name from the
            # resolved registry_spec (flag override wins over base config).
            effective_import = updates.get("import_path") or agent.import_path or ""
            if "UserSimAcpAgent" in effective_import:
                effective_spec = (
                    kwargs_updates.get("registry_spec")
                    or (agent.kwargs or {}).get("registry_spec")
                )
                updates["name"] = _acp_agent_name(effective_spec)
            new_agents.append(agent.model_copy(update=updates) if updates else agent)
        overrides["agents"] = new_agents

    return config.model_copy(update=overrides) if overrides else config


# ── Run ───────────────────────────────────────────────────────────────────────

async def run(config: JobConfig) -> int:
    try:
        job = await Job.create(config)
    except ValueError as exc:
        # Most common cause: dataset name wrong, task_names filter matches nothing,
        # or a network error fetching the package registry.
        print(f"ERROR: Could not resolve tasks — {exc}", file=sys.stderr)
        print(
            "Hint: run `harbor dataset list` to check available dataset names, "
            "and verify --tasks patterns match the task IDs in the dataset.",
            file=sys.stderr,
        )
        return 1
    agents_desc = ", ".join(
        a.import_path.rsplit(":", 1)[-1] if a.import_path else (a.name or "?")
        for a in config.agents
    )
    print(f"Running {len(job)} trial(s)  [{agents_desc}]")
    if config.debug:
        print("Debug logging enabled — exec output and timing will be shown.")

    result = await job.run()

    print(f"\nDone — {result.stats.n_completed_trials} completed, "
          f"{result.stats.n_errored_trials} errored, "
          f"{result.stats.n_cancelled_trials} cancelled")
    for evals_key, evals in result.stats.evals.items():
        metrics_str = ", ".join(
            f"{k}={v:.3f}"
            for m in evals.metrics
            for k, v in m.items()
            if isinstance(v, (int, float))
        )
        errors = f"  ({evals.n_errors} error(s))" if evals.n_errors else ""
        print(f"  {evals_key}: {metrics_str or '(no metrics)'}{errors}")

    return 0 if result.stats.n_errored_trials == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = load_config_from_yaml(Path(args.config))
    config = apply_overrides(config, args)

    return asyncio.run(run(config))


if __name__ == "__main__":
    raise SystemExit(main())
