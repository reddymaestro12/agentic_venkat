from typing import Callable
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    AgentState,
    ExtendedModelResponse
)
from langgraph.types import Command
from typing_extensions import NotRequired

class UsageTrackingState(AgentState):
    """Agent state with token usage tracking."""

    last_model_call_tokens: NotRequired[int]


@wrap_model_call(state_schema=UsageTrackingState)
def track_usage(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ExtendedModelResponse:
    response = handler(request)
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={"last_model_call_tokens": 150}),
    )