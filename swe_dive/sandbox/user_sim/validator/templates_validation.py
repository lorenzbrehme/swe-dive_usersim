"""Validation prompt templates (verbatim from legacy)."""

VALIDATION_PROMPTS = {
    "validate_subtask_list": """
    You are an expert technical architect. Your task is to review a list of generation subtasks against a problem description to ensure the plan is logical, helpful, and sufficient to guide an agent.

    ### Input Data
    - Problem Description: {problem_description}
    - Subtask List: {subtask_list}

    ### Validation Criteria 
    The plan is considered VALID if it meets the following balanced criteria:
    1. Broad Alignment: The subtasks should reasonably target the problem description. Allow minor refactoring, exploratory steps, or setup tasks if they logically pave the way for the fix. Only reject if the subtasks are completely unrelated.
    2. Understandable Direction: Each subtask should provide enough clarity for an agent to understand its high-level goal, even if the exact files or precise line changes aren't specified yet.

    ### Output Format
    Return a strict JSON object:
    {{
        "is_valid": bool,  // true if the plan is a reasonable and workable approach to the problem; false only if it completely misses the mark or is severely flawed
        "reason": "Provide a helpful summary of the assessment. If is_valid is false, specify the major gaps or critical flaws in the plan."
    }}
    """,
    "validate_agent_input": """
            Validate the following agent input based on the provided Problem desrciption and the current state of the repository.
            Problem Description: {problem_description}
            Current Repository State (Changes made so far): {repo_state}
            Subtasks: {subtasks}
            Agent Input: {agent_input}
            Validation Criteria:
            1. Broad Alignment: The agent input should reasonably align with the overall goal of solving the problem or progressing through the subtasks. It does not need to be the perfect next step, but it shouldn't be completely unrelated.
            2. High-Level Intent: The agent's intent should be understandable, even if the exact implementation details or syntax are not fully fleshed out yet.
            
            Return a JSON object with the following format:
            {{
                "is_valid": bool,  // true if the agent input is valid based on the criteria, false otherwise
                "reason": str      // if is_valid is false, provide a brief explanation of why the agent input is not valid based on the criteria
            }}
            
        """,
}
