# pip install tenacity langchain-core
from typing import Any, Dict, Optional, Iterable, Union
from langchain_core.runnables import Runnable, RunnableConfig
from tenacity import retry, stop_after_attempt, wait_exponential
from google.cloud import logging

GraphLike = Dict[str, Any]  # your GraphState type

# Configure GCP logging
client = logging.Client()
gcp_logger = client.logger(__name__)

class PolicyNode(Runnable[GraphLike, GraphLike]):
    def __init__(
        self,
        inner: Union[Runnable, callable],     # does the real work; takes/returns GraphState
        name: str,
        max_attempts: int = 3,
        fallback: Optional[Union[Runnable, callable]] = None,
        tweak_input: Optional[callable] = None,   # (state, exc, attempt) -> state
        on_error: Optional[callable] = None,      # (exc, attempt, state) -> None
    ):
        self.inner = inner
        self.name = name
        self.max_attempts = max_attempts
        self.fallback = fallback
        self.tweak_input = tweak_input
        self.on_error = on_error

    def _invoke_once(self, state: GraphLike, config: Optional[RunnableConfig]) -> GraphLike:
        # inner can be a Runnable or plain function
        if hasattr(self.inner, "invoke"):
            return self.inner.invoke(state, config=config)
        return self.inner(state)

    def invoke(self, state: GraphLike, config: Optional[RunnableConfig] = None) -> GraphLike:
        attempt = 0
        last_exc = None
        cur_state = {**state, "last_node": self.name}

        while attempt < self.max_attempts:
            try:
                out = self._invoke_once(cur_state, config)
                # Normalize: succeed with error=None
                out = {**cur_state, **out, "error": None, "last_node": self.name}
                print("=================NO EXCEPTION=================")
                return out
            except Exception as e:  # catch *everything* here
                print(f"Attempt {attempt + 1} failed in node '{self.name}': {str(e)}")
                print("=================EXCEPTION=================")
                last_exc = e
                attempt += 1
                if self.on_error:
                    self.on_error(e, attempt, cur_state)
                # Optional: mutate state between attempts
                if self.tweak_input:
                    cur_state = self.tweak_input(cur_state, e, attempt)

        # Attempts exhausted → try fallback once
        if self.fallback is not None:
            try:
                if hasattr(self.fallback, "invoke"):
                    out = self.fallback.invoke(cur_state, config=config)
                else:
                    out = self.fallback(cur_state)
                return {**cur_state, **out, "error": None, "last_node": self.name}
            except Exception as e2:
                last_exc = e2

        # Return a structured error instead of raising
        return {
            **cur_state,
            "error": {
                "node": self.name,
                "type": type(last_exc).__name__,
                "msg": str(last_exc),
            },
        }

    # Optional: streaming passthrough with graceful error event
    def stream(self, state: GraphLike, config: Optional[RunnableConfig] = None) -> Iterable:
        if hasattr(self.inner, "stream"):
            try:
                yield from self.inner.stream(state, config=config)
                return
            except Exception as e:
                yield {"event": "error", "error": {"node": self.name, "type": type(e).__name__, "msg": str(e)}}
        # Fallback to non-stream invoke if inner doesn't support stream
        yield self.invoke(state, config=config)

    def on_error(self, exc: Exception, attempt: int, state: GraphLike) -> None:
        """
        Categorizes the error and logs it appropriately using GCP logger.

        Args:
            exc (Exception): The exception that occurred.
            attempt (int): The current attempt number.
            state (GraphLike): The current state of the graph.
        """
        error_type = type(exc).__name__
        error_message = str(exc)

        # Categorize the error
        if isinstance(exc, ValueError):
            category = "Validation Error"
        elif isinstance(exc, KeyError):
            category = "Key Error"
        elif isinstance(exc, RuntimeError):
            category = "Runtime Error"
        else:
            category = "Unknown Error"

        # Log the error using GCP logger
        gcp_logger.log_text(
            f"Error occurred in node '{self.name}' (Attempt {attempt}): "
            f"[{category}] {error_type}: {error_message}",
            severity="ERROR"
        )

        # Optional: Add custom handling logic here
        # For example, update the state with error details
        state["error_details"] = {
            "category": category,
            "type": error_type,
            "message": error_message,
            "attempt": attempt,
        }
