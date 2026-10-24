"""LLM-based validators."""

import logging

from ..utils import parse_json
from .templates_validation import VALIDATION_PROMPTS

logger = logging.getLogger("swe_dive.user_sim.validator")


def _coerce_result(validation_result):
    """Normalize ``is_valid`` to a bool, or return None if unusable."""
    if (
        validation_result is not None
        and "is_valid" in validation_result
        and "reason" in validation_result
    ):
        is_valid = validation_result["is_valid"]
        if isinstance(is_valid, bool):
            return validation_result
        if isinstance(is_valid, str):
            if is_valid.lower() == "true":
                validation_result["is_valid"] = True
                return validation_result
            if is_valid.lower() == "false":
                validation_result["is_valid"] = False
                return validation_result
    return None


def validate_subtask(problem_description, subtask, model):
    subtaskpromt = VALIDATION_PROMPTS.get("validate_subtask_list").format(
        problem_description=problem_description,
        subtask_list=subtask,
    )
    for tries in range(3):
        model.reset_chat()  # reset model state before each validation attempt
        validation_response = model.prompt(subtaskpromt)
        result = _coerce_result(parse_json(validation_response))
        if result is not None:
            return result
        logger.warning("Validation attempt %d failed to parse JSON. Retrying...", tries + 1)

    return {
        "is_valid": False,
        "reason": "Failed to parse validation response as JSON after 3 attempts. Please regenerate",
    }


def validate_agent_input(problem_description, repo_state, subtasks, agent_input, model):
    validation_prompt = VALIDATION_PROMPTS.get("validate_agent_input").format(
        problem_description=problem_description,
        repo_state=repo_state,
        subtasks=subtasks,
        agent_input=agent_input,
    )
    for tries in range(3):
        model.reset_chat()  # reset model state before each validation attempt
        validation_response = model.prompt(validation_prompt)
        result = _coerce_result(parse_json(validation_response))
        if result is not None:
            return result
        logger.warning("Validation attempt %d failed to parse JSON. Retrying...", tries + 1)

    return {
        "is_valid": False,
        "reason": "Failed to parse validation response as JSON after 3 attempts. Please regenerate",
    }
