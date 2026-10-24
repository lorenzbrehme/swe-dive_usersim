"""In-container orchestrator: run one task's user-sim <-> coding-agent loop.

This is the entrypoint the Harbor agent executes *inside* the sandbox. It:

1. reads the task instruction (the SWE-Bench problem statement);
2. builds the simulator + validation chat models (bare LiteLLM via ``LLM_MODEL``);
3. builds the selected coding-agent backend (OpenHands local workspace by default);
4. runs ``UserSimulator.simulateInteraction()`` — applying every transformation;
5. writes the trajectory to ``/logs/agent/trajectory.json`` for Harbor to collect;
6. leaves the agent's edits in the repo working tree for the verifier to grade.

Usage (inside the container):

    python -m swe_dive.orchestrator \
        --instruction-file /logs/agent/instruction.md \
        --workdir /testbed \
        --persona EXPERT_NITPICKER \
        --coding-mode COLLABORATIVE
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import litellm

from .coding_agents.acp_agent import ACPAgent
from .coding_agents.openhands_agent import OpenHandsAgent
from .models import LiteLLMChat
from .user_sim import UserSimulator

logger = logging.getLogger("swe_dive.orchestrator")


class SimulatorAbortedError(RuntimeError):
    """The user-simulator LLM repeatedly failed to produce a parseable/valid
    response (see ``UserSimulator.simulateInteraction``).

    This is a simulator-side failure, not the coding agent's fault, so
    ``_run`` retries the whole trial (fresh models + fresh backend session)
    up to ``--max-retries`` times before giving up.
    """


class AgentSilentFailureError(RuntimeError):
    """The coding-agent backend never produced a response on any turn.

    This points at a broken backend/config (bad model, auth, unsupported
    protocol call, ...) rather than the user simulator, so it is *not*
    retried — retrying would just reproduce the same failure.
    """


# Exceptions where retrying is certain to reproduce the same failure (bad
# credentials never fix themselves, and a reached LLM limit stays reached even
# after chat.py's in-call back-off) — fail fast instead of burning retries.
_NON_RETRYABLE_EXCEPTIONS = (
    litellm.AuthenticationError,
    litellm.RateLimitError,
)


def _read_instruction(args) -> str:
    if args.instruction:
        return args.instruction
    if args.instruction_file:
        return Path(args.instruction_file).read_text()
    env_val = os.getenv("SWE_DIVE_INSTRUCTION")
    if env_val:
        return env_val
    # Harbor renders the instruction onto stdin for some agent setups.
    data = sys.stdin.read()
    if data.strip():
        return data
    raise SystemExit(
        "No instruction provided (use --instruction / --instruction-file / "
        "SWE_DIVE_INSTRUCTION / stdin)."
    )


def _build_backend(args):
    if args.backend == "openhands":
        return OpenHandsAgent(workdir=args.workdir, llm_model=args.agent_model)
    if args.backend == "acp":
        return ACPAgent(
            launcher_path=args.acp_launcher,
            runner_dir=args.acp_runner_dir,
            workdir=args.workdir,
            logs_dir=args.log_dir,
            agent_model=args.agent_model,
        )
    raise SystemExit(f"Unknown backend: {args.backend!r}")


def _reset_workdir(workdir: str) -> None:
    """Discard whatever the coding agent did during a failed attempt.

    Runs before each retry so the next attempt starts from the same clean
    ``base_commit`` checkout Harbor originally prepared, rather than layering
    a second attempt's edits on top of the first's.
    """
    subprocess.run(
        ["git", "reset", "--hard"], cwd=workdir, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "clean", "-fd"], cwd=workdir, capture_output=True, text=True
    )


async def _run_attempt(args, instruction: str, log_dir: Path, attempt: int) -> int:
    """Run one full user-sim <-> coding-agent interaction.

    Builds a fresh pair of chat models and a fresh backend/simulator so that
    retries never inherit conversation history or a stale backend session
    from a previous failed attempt.
    """
    sim_model = LiteLLMChat(model=args.sim_model)
    validation_model = LiteLLMChat(model=args.validation_model or args.sim_model)

    backend = _build_backend(args)
    await backend.setup()

    sim = UserSimulator(
        persona=args.persona,
        coding_mode=args.coding_mode,
        issue=instruction,
        repo_context=args.repo_context,
        model=sim_model,
        agent=backend,
        validation_model=validation_model,
        log_dir=str(log_dir),
        max_turns=args.max_turns,
    )

    try:
        history = await sim.simulateInteraction()
    except Exception as e:
        # A fatal error propagated out of the simulator (e.g. ContentPolicyViolationError,
        # AuthenticationError).  Log it clearly so it's visible in orchestrator.txt,
        # then re-raise so the process exits non-zero and Harbor counts the trial
        # as errored rather than silently proceeding to the verifier.
        logger.error(
            "[FATAL] Simulator aborted with a permanent error (%s: %s). "
            "Proceeding to verifier is skipped.",
            type(e).__name__,
            e,
        )
        raise
    finally:
        await backend.cleanup()

    final_diff = backend.git_diff()
    agent_model = getattr(backend, "agent_model", None) or getattr(backend, "llm_model", None)

    sim_usage = sim_model.usage
    agent_usage = getattr(backend, "usage", None)
    validation_usage = validation_model.usage if validation_model else None
    

    # ── Conversation log (legacy format, matches user_sim/conversations/*.json)
    # Loads task metadata from /tests/eval_spec.json when available (written by
    # swe_dive.adapters.swebench).  Falls back gracefully for the registered
    # Harbor SWE-bench dataset which uses a different layout.
    eval_spec_path = Path("/tests/eval_spec.json")
    test_case: dict = {"problem_statement": instruction}
    if eval_spec_path.exists():
        try:
            test_case.update(json.loads(eval_spec_path.read_text()))
        except Exception:  # noqa: BLE001
            pass

    conversation_log = {
        "sim_response": history,
        "test_case": test_case,
        "git_diff": final_diff,
        "persona": args.persona,
        "coding_mode": args.coding_mode,
        "sim_model": sim_model.MODEL_NAME,
        "agent_model": agent_model, 
        "validation_model": validation_model.MODEL_NAME,
        "sim_usage": sim_usage,
        "attempt": attempt,
        'agent_usage': agent_usage,
        "validation_usage": validation_usage,
    }
    (log_dir / "conversation.json").write_text(json.dumps(conversation_log, indent=2))
    logging.getLogger("swe_dive").info(
        "Wrote conversation log (%d turn(s)) → %s",
        len(history),
        log_dir / "conversation.json",
    )
    logging.getLogger("swe_dive").info(
        "Final diff: %d char(s) left in %s for the verifier.",
        len(final_diff),
        args.workdir,
    )

    # Detect invalid runs.
    # Valid runs either run for max_turns or end with [DONE]. BASE_MODE is a
    # single-turn run by design (no persona-driven follow-up loop), so it
    # never emits [DONE] and never reaches max_turns — treat it as complete
    # on its own.
    last_user_message = (history[-1].get("user_message") or "") if history else ""
    ran_to_completion = (
        args.persona == "BASE_MODE"
        or "[DONE]" in last_user_message
        or len(history) >= args.max_turns
    )
    if not ran_to_completion:
        raise SimulatorAbortedError(
            f"Simulator aborted early after {len(history)} turn(s) without a "
            f"'[DONE]' signal (max_turns={args.max_turns}). This indicates a "
            "repeated parse/validation failure in the user-simulator LLM, not "
            "a legitimate reward=0. Check agent/user_sim.log for details."
        )

    # If every turn has an empty agent_response, the coding-agent backend failed silently (e.g. unsupported session/set_model, missing auth, or a
    if history and not any((t.get("agent_response") or "").strip() for t in history):
        raise AgentSilentFailureError(
            f"Agent returned an empty response on all {len(history)} turn(s). "
            "The coding-agent backend likely failed silently "
            "(e.g. model not configured, auth error, or unsupported session/set_model). "
            "Check agent/orchestrator.txt for details."
        )

    return 0


async def _run(args) -> int:
    # Configure logging before anything else so all downstream loggers inherit it.
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s" if args.debug
               else "%(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress litellm's own verbose output unless we're in debug mode.
    logging.getLogger("LiteLLM").setLevel(
        logging.DEBUG if args.debug else logging.WARNING
    )

    instruction = _read_instruction(args)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log turn-level activity to a dedicated file
    _sim_log_path = log_dir / "user_sim.log"
    _sim_file_handler = logging.FileHandler(_sim_log_path, encoding="utf-8")
    _sim_file_handler.setLevel(logging.INFO)
    _sim_file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("swe_dive.user_sim").addHandler(_sim_file_handler)

    # Retry the simulation N times except when the agent failed due to incorrect inits.
    # ``attempt`` is 0-based (matches the "attempt" field written to
    # trajectory.json / conversation.json): 0 is the first try, 1 the first
    # retry, and so on.
    max_attempts = max(args.max_retries, 0) + 1
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        if attempt > 0:
            logger.warning(
                "[RETRY] Attempt %d/%d after a simulator-side failure (%s: %s). "
                "Resetting %s to a clean checkout and starting over.",
                attempt, max_attempts - 1, type(last_exc).__name__, last_exc, args.workdir,
            )
            _reset_workdir(args.workdir)
        try:
            return await _run_attempt(args, instruction, log_dir, attempt)
        except _NON_RETRYABLE_EXCEPTIONS:
            raise
        except AgentSilentFailureError:
            raise
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt >= max_attempts - 1:
                logger.error(
                    "[FATAL] Giving up after %d attempt(s). Last error: %s: %s",
                    attempt + 1, type(e).__name__, e,
                )
                raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(  # pragma: no cover — unreachable
        "_run: retry loop exited without returning or raising"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="swe_dive in-container orchestrator")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--instruction", help="Task instruction text.")
    src.add_argument("--instruction-file", help="Path to a file with the instruction.")
    parser.add_argument("--workdir", default="/testbed", help="Repository working dir.")
    parser.add_argument("--repo-context", default="", help="Optional repo context for prompts.")
    parser.add_argument("--persona", default="EXPERT_NITPICKER")
    parser.add_argument("--coding-mode", default="COLLABORATIVE")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help=(
            "Extra attempts (fresh models + fresh backend session + clean "
            "workdir) when a trial fails for reasons unrelated to the coding "
            "agent (e.g. repeated user-simulator parse/validation failures). "
            "0 disables retries."
        ),
    )
    parser.add_argument("--backend", default="openhands", choices=["openhands", "acp"])
    parser.add_argument(
        "--acp-launcher",
        default="/installed-agent/acp-launch.sh",
        help="Path to the ACP launcher script (uploaded by Harbor's AcpAgent).",
    )
    parser.add_argument(
        "--acp-runner-dir",
        default="/installed-agent",
        help="Dir containing Harbor's uploaded acp_runner.py (for HarborAcpClient).",
    )
    parser.add_argument("--log-dir", default="/logs/agent")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging (full prompts, responses, diffs).",
    )
    parser.add_argument(
        "--sim-model",
        default=os.getenv("LLM_MODEL"),
        help="LiteLLM model for the user simulator (defaults to $LLM_MODEL).",
    )
    parser.add_argument(
        "--validation-model",
        default=os.getenv("VALIDATION_LLM_MODEL"),
        help="LiteLLM model for validation (defaults to --sim-model).",
    )
    parser.add_argument(
        "--agent-model",
        default=os.getenv("AGENT_LLM_MODEL"),
        help="LiteLLM model for the coding agent (defaults to $LLM_MODEL).",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
