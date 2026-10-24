"""The multi-turn user simulator.

Output goes through Python ``logging`` rather than bare ``print()``,
so verbosity is controlled by the log level set in the orchestrator:

  - ``INFO``  — turn numbers, brief message summaries, validation pass/fail,
                [DONE] / abort signals.  Always visible.
  - ``DEBUG`` — full prompts sent to the simulator LLM, full raw responses,
                complete git diffs, validation reasoning.
                Enabled by ``uv run main.py --debug``.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .prompt_templates import CODING_MODES, PERSONAS, TEMPLATES
from .utils import parse_json
from .validator.validator import validate_agent_input, validate_subtask

logger = logging.getLogger("swe_dive.user_sim")


class UserSimulator:
    """Callable user simulator that assembles prompts for the coding agent."""

    def __init__(
        self,
        persona: str = "EXPERT_NITPICKER",
        coding_mode: str = "COLLABORATIVE",
        issue: str = "",
        repo_context: str = "",
        model=None,
        agent=None,
        validation_model=None,
        log_dir: str | os.PathLike[str] = "/logs/agent",
        max_turns: int = 10,
    ):
        self.persona_name = persona
        self.persona = PERSONAS.get(persona)
        self.coding_mode = CODING_MODES.get(coding_mode)
        self.issue = issue
        self.repo_context = repo_context
        self.model = model
        self.agent = agent
        self.validation_model = validation_model
        self.max_turns = max_turns

        self.log_dir = Path(log_dir)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.log_dir = Path("logs")
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _write_validation_log(self, record: dict) -> None:
        with open(self.log_dir / "validation_log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def initial_prompt(self) -> str:
        template = TEMPLATES.get("inital_prompt")
        return template.format(
            PERSONA=self.persona,
            CODING_MODE=self.coding_mode,
            ISSUE=self.issue,
            REPO_CONTEXT=self.repo_context,
        ).strip()

    def followup_prompt(
        self, agent_last_message: str, agent_changes: Optional[str] = ""
    ) -> str:
        template = TEMPLATES.get("followup_prompt")
        return template.format(
            AGENT_LAST_MESSAGE=agent_last_message,
            AGENT_CHANGES=agent_changes or "",
        ).strip()

    async def simulateInteraction(self):  # noqa: N802

        if self.persona_name == "BASE_MODE":
            logger.debug("[PROMPT] BASE_MODE persona — running the standard benchmark task, no multi-turn simulation.")
            logger.info("── BASE_MODE ── running standard benchmark task (single turn).")

            await self.agent.run_task(self.issue)
            last_message = self.agent.get_last_message()
            git_diff = self.agent.git_diff()
            tool_calls: list[dict] = (
                self.agent.get_turn_tool_calls()
                if hasattr(self.agent, "get_turn_tool_calls")
                else []
            )

            logger.info("[AGENT_RESP] %s", last_message)
            logger.debug("[AGENT_RESP] Full response:\n%s", last_message)
            logger.debug(
                "[DIFF] %d chars:\n%s",
                len(git_diff or ""),
                (git_diff or "(empty)"),
            )
            if tool_calls:
                logger.debug("[TOOL_CALLS] %d call(s) this turn.", len(tool_calls))

            return [
                {
                    "user_message": self.issue,
                    "agent_response": last_message,
                    "agent_changes": git_diff,
                    "subtask_list": None,
                    "tool_calls": tool_calls,
                }
            ]

        complete_history = []
        init_prompt = self.initial_prompt()

        logger.debug("[PROMPT] Initial prompt:\n%s", init_prompt)

        validation_result = False
        response = self.model.chat_with_model(init_prompt)
        counter = 0
        agent_message = None
        subtask_list = None

        # ── Initial decomposition + validation loop ───────────────────────────
        while not validation_result:
            logger.debug("[LLM_RESPONSE] Raw response:\n%s", response)

            response_parse_to_json = parse_json(response)
            if (
                response_parse_to_json is not None
                and "initial_message" in response_parse_to_json
                and "subtask_list" in response_parse_to_json
            ):
                agent_message = response_parse_to_json["initial_message"]
                subtask_list = response_parse_to_json["subtask_list"]

                logger.debug(
                    "[PARSED] initial_message=%r  subtasks=%s",
                    agent_message,
                    subtask_list,
                )

                validate_subtask_result = validate_subtask(
                    problem_description=self.issue,
                    subtask=subtask_list,
                    model=self.validation_model,
                )
                if validate_subtask_result is not None and validate_subtask_result.get(
                    "is_valid"
                ):
                    logger.debug(
                        "[VALIDATION] Subtask list valid: %s",
                        validate_subtask_result.get("reason"),
                    )
                    validate_agent_input_result = validate_agent_input(
                        problem_description=self.issue,
                        repo_state=self.repo_context,
                        subtasks=subtask_list,
                        agent_input=agent_message,
                        model=self.validation_model,
                    )
                    if (
                        validate_agent_input_result is not None
                        and validate_agent_input_result.get("is_valid")
                    ):
                        logger.debug(
                            "[VALIDATION] Agent input valid: %s",
                            validate_agent_input_result.get("reason"),
                        )
                        validation_result = True
                    else:
                        reason = (validate_agent_input_result or {}).get(
                            "reason", "Unknown reason"
                        )
                        logger.info(
                            "[VALIDATION] Agent input invalid (attempt %d/3): %s",
                            counter + 1,
                            reason,
                        )
                        regeneration_prompt = TEMPLATES.get(
                            "regeneration_prompt"
                        ).format(REASON=reason)
                        response = self.model.chat_with_model(regeneration_prompt)
                        self._write_validation_log(
                            {
                                "type": "agent_input_validation_failure",
                                "issue": self.issue,
                                "repo_state": "",
                                "subtasks": subtask_list,
                                "agent_input": agent_message,
                                "validation_response": validate_agent_input_result,
                            }
                        )
                        counter += 1
                else:
                    reason = (validate_subtask_result or {}).get(
                        "reason", "Unknown reason"
                    )
                    logger.info(
                        "[VALIDATION] Subtask list invalid (attempt %d/3): %s",
                        counter + 1,
                        reason,
                    )
                    regeneration_prompt = TEMPLATES.get(
                        "regeneration_prompt"
                    ).format(REASON=reason)
                    response = self.model.chat_with_model(regeneration_prompt)
                    self._write_validation_log(
                        {
                            "type": "subtask_validation_failure",
                            "issue": self.issue,
                            "subtasklist": subtask_list,
                            "validation_response": validate_subtask_result,
                        }
                    )
                    counter += 1
            else:
                logger.info(
                    "[PARSE_FAIL] Could not parse initial response (attempt %d/3)",
                    counter + 1,
                )
                # Only delete if the model returned something — chat_with_model
                # already pops the user message when _complete returns None, so
                # calling delete_last_message in that case would remove the
                # previous successful turn and corrupt the context.
                if response is not None:
                    self.model.delete_last_message()
                response = self.model.chat_with_model(init_prompt)
                counter += 1

            if counter >= 3:
                logger.warning(
                    "[ABORT] Failed to get a valid initial response after 3 retries."
                )
                # return error
                raise RuntimeError(
                    "Failed to get a valid follow-up after 3 retries."
                )

        # ── Turn loop ─────────────────────────────────────────────────────────
        for turn in range(self.max_turns):
            logger.info(
                "── Turn %d/%d ──  message: %s",
                turn + 1,
                self.max_turns,
                agent_message,
            )
            logger.debug("[AGENT_MSG] Full message to agent:\n%s", agent_message)

            await self.agent.run_task(agent_message)
            last_message = self.agent.get_last_message()
            git_diff = self.agent.git_diff()
            # Collect per-turn tool calls for the ATIF trajectory when the
            # backend supports it (ACP and OpenHands backends do; mock does not).
            tool_calls: list[dict] = (
                self.agent.get_turn_tool_calls()
                if hasattr(self.agent, "get_turn_tool_calls")
                else []
            )

            logger.info("[AGENT_RESP] %s", last_message)
            logger.debug("[AGENT_RESP] Full response:\n%s", last_message)
            logger.debug(
                "[DIFF] %d chars:\n%s",
                len(git_diff or ""),
                (git_diff or "(empty)"),
            )
            if tool_calls:
                logger.debug("[TOOL_CALLS] %d call(s) this turn.", len(tool_calls))

            complete_history.append(
                {
                    "user_message": agent_message,
                    "agent_response": last_message,
                    "agent_changes": git_diff,
                    "subtask_list": subtask_list,
                    "tool_calls": tool_calls,
                }
            )
            if turn + 1 >= self.max_turns:
                logger.info(
                    "[ABORT] Reached maximum number of turns (%d). Ending simulation.",
                    self.max_turns,
                )
                return complete_history
            followup = self.followup_prompt(
                agent_last_message=last_message, agent_changes=git_diff
            )
            logger.debug("[FOLLOWUP_PROMPT]:\n%s", followup)
            response = self.model.chat_with_model(followup)
            logger.debug("[FOLLOWUP_RESPONSE]:\n%s", response)

            validation_result = False
            counter = 0

            while not validation_result:
                if response is not None and "[DONE]" in response:
                    logger.info("[DONE] Simulator satisfied — ending after turn %d.", turn + 1)
                    complete_history.append(
                        {
                            "user_message": response,
                            "agent_response": "",
                            "agent_changes": "",
                        }
                    )
                    return complete_history

                response_parse_to_json = parse_json(response)
                if (
                    response_parse_to_json is not None
                    and "follow-up" in response_parse_to_json
                ):
                    agent_message = response_parse_to_json["follow-up"]
                    validate_agent_input_result = validate_agent_input(
                        problem_description=self.issue,
                        repo_state=self.repo_context,
                        subtasks=subtask_list,
                        agent_input=agent_message,
                        model=self.validation_model,
                    )
                    if (
                        validate_agent_input_result is not None
                        and validate_agent_input_result.get("is_valid")
                    ):
                        logger.debug(
                            "[VALIDATION] Follow-up valid: %s",
                            validate_agent_input_result.get("reason"),
                        )
                        validation_result = True
                    else:
                        reason = (validate_agent_input_result or {}).get(
                            "reason", "Unknown reason"
                        )
                        logger.info(
                            "[VALIDATION] Follow-up invalid (attempt %d/3): %s",
                            counter + 1,
                            reason,
                        )
                        regeneration_prompt = TEMPLATES.get(
                            "regeneration_prompt"
                        ).format(REASON=reason)
                        response = self.model.chat_with_model(regeneration_prompt)
                        self._write_validation_log(
                            {
                                "type": "agent_input_validation_failure",
                                "issue": self.issue,
                                "repo_state": git_diff,
                                "subtasks": subtask_list,
                                "agent_input": agent_message,
                                "validation_response": validate_agent_input_result,
                            }
                        )
                        counter += 1
                else:
                    logger.info(
                        "[PARSE_FAIL] Could not parse follow-up (attempt %d/3)",
                        counter + 1,
                    )
                    # Only delete if the model returned something — chat_with_model
                    # already pops the user message when _complete returns None, so
                    # calling delete_last_message in that case would remove the
                    # previous successful turn and corrupt the context.
                    if response is not None:
                        self.model.delete_last_message()
                    response = self.model.chat_with_model(followup)
                    counter += 1

                if counter >= 3:
                    logger.warning(
                        "[ABORT] Failed to get a valid follow-up after 3 retries."
                    )
                    # return error
                    raise RuntimeError(
                        "Failed to get a valid follow-up after 3 retries."
                    )
                    
                    
            if response is not None and "[DONE]" in response:
                break

            agent_message = response_parse_to_json["follow-up"]
            if not isinstance(agent_message, str):
                logger.warning(
                    "[ABORT] Follow-up is not a string (%r). Ending simulation.",
                    type(agent_message).__name__,
                )
                break
        logger.info(
            "[ABORT] Reached maximum number of turns (%d). Ending simulation.",
            self.max_turns
        )
        return complete_history
