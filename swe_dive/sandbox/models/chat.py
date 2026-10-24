"""Minimal chat adapter over **bare** ``litellm``.

``swe_dive`` standardizes on LiteLLM with default options: the model is taken
from the ``LLM_MODEL`` environment variable and all provider credentials (API
keys, base URLs, regions, ...) are read by LiteLLM directly from the
environment. Those variables live in the Harbor container's environment (set via
``.env2`` -> ``scripts/run.sh`` -> ``task.toml``/``job.yaml`` ``[environment].env``);
there is no provider auto-detection or config layer to maintain here.

The only thing this class adds on top of ``litellm.completion`` is the small
amount of conversation-history bookkeeping the user simulator / validator expect
(``prompt``, ``chat_with_model``, ``reset_chat``, ``delete_last_message``,
``get_chat_history``).
"""

import logging
import os
import time

import litellm

logger = logging.getLogger("swe_dive.models.chat")

# When the LLM limit is hit (HTTP 429), wait this long and retry this many
# times before giving up and treating the limit as reached.
_RATE_LIMIT_WAIT_SECONDS = 60
_RATE_LIMIT_MAX_RETRIES = 3


class LiteLLMChat:
    """Stateful chat wrapper around bare ``litellm.completion``."""

    def __init__(self, model: str | None = None, *, max_retries: int = 5):
        self.MODEL_NAME = (model or os.getenv("LLM_MODEL") or "").strip()
        if not self.MODEL_NAME:
            raise RuntimeError(
                "No model configured. Set the LLM_MODEL environment variable "
                "(e.g. LLM_MODEL=openai/gpt-4o). See .env2.example."
            )
        self._num_retries = max_retries
        self.history: list[dict[str, str]] = []
        # Accumulated usage across all calls on this instance.
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0

    def _complete(self, messages):
        """Thin wrapper around litellm.completion with built-in retry.

        On an HTTP 429 (LLM rate/quota limit) the call is retried
        ``_RATE_LIMIT_MAX_RETRIES`` times, waiting ``_RATE_LIMIT_WAIT_SECONDS``
        between attempts.  If the limit is still hit after those retries it is
        treated as reached and re-raised so callers don't mistake it for a
        retriable parse failure.
        """
        for rate_limit_attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            try:
                response = litellm.completion(
                    model=self.MODEL_NAME,
                    messages=messages,
                    num_retries=self._num_retries,
                )
                usage = getattr(response, "usage", None)
                if usage:
                    self._total_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self._total_output_tokens += getattr(usage, "completion_tokens", 0) or 0
                try:
                    cost = litellm.completion_cost(completion_response=response)
                    if cost:
                        self._total_cost_usd += cost
                except Exception:  # noqa: BLE001
                    pass
                return response.choices[0].message.content
            except litellm.RateLimitError:
                # 429 — the LLM limit was reached. Wait and retry a few times in
                # case it is a transient/windowed rate limit that resets.
                if rate_limit_attempt < _RATE_LIMIT_MAX_RETRIES:
                    logger.warning(
                        "LLM limit reached (HTTP 429) for model '%s'. Waiting %ds "
                        "before retry %d/%d.",
                        self.MODEL_NAME,
                        _RATE_LIMIT_WAIT_SECONDS,
                        rate_limit_attempt + 1,
                        _RATE_LIMIT_MAX_RETRIES,
                    )
                    time.sleep(_RATE_LIMIT_WAIT_SECONDS)
                    continue
                # Still limited after all retries — treat as permanently reached.
                logger.error(
                    "LLM limit still reached (HTTP 429) for model '%s' after %d "
                    "retries. Giving up.",
                    self.MODEL_NAME,
                    _RATE_LIMIT_MAX_RETRIES,
                )
                raise
            except (
                litellm.ContentPolicyViolationError,
                litellm.AuthenticationError,
                litellm.ContextWindowExceededError,
            ):
                # Permanent failures — retrying the same content or credentials
                # will never succeed.  Re-raise so callers don't treat this as a
                # retriable parse failure.
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("Error generating completion: %s", e)
                return None

    @property
    def usage(self) -> dict:
        """Accumulated token usage and cost across all calls."""
        return {
            "input_tokens": self._total_input_tokens or None,
            "output_tokens": self._total_output_tokens or None,
            "cost_usd": self._total_cost_usd or None,
        }

    def prompt(self, prompt):
        """Single-shot completion with no conversation history."""
        return self._complete([{"role": "user", "content": prompt}])

    def chat_with_model(self, message):
        """Multi-turn chat: appends to history and returns the model reply."""
        self.history.append({"role": "user", "content": message})
        reply = self._complete(self.history)
        if reply is None:
            self.history.pop()
            return None
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset_chat(self):
        self.history = []

    def get_chat_history(self):
        return self.history

    def delete_last_message(self):
        """Drop the last user+assistant turn so it can be retried."""
        if len(self.history) >= 2:
            self.history.pop()  # assistant reply
            self.history.pop()  # user message
        elif len(self.history) == 1:
            self.history = []
