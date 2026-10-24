"""LLM access for swe_dive — bare LiteLLM, configured via environment variables.

``swe_dive`` uses LiteLLM with default options. The model is set with
``LLM_MODEL`` and provider credentials are read by LiteLLM straight from the
container environment (injected by Harbor). The only helper here is
:class:`LiteLLMChat`, a thin conversation-history adapter the user simulator
needs.
"""

from .chat import LiteLLMChat

__all__ = ["LiteLLMChat"]
