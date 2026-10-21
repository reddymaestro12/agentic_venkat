from dotenv import load_dotenv
load_dotenv()

from typing import Optional

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ToolRetryMiddleware,
    ToolErrorMiddleware,
)
from langchain.agents.middleware.types import ToolCallRequest


@tool
def divide(a: int, b: int) -> int:
    """Performs division operation."""
    
    if b == 0:
        raise ValueError("denominator cannot be zero")

    return a // b


def error_handler(
    exc: Exception,
    request: ToolCallRequest,
) -> Optional[str]:

    tool_name = request.tool_call["name"]

    if isinstance(exc, ValueError):
        return f"Input error in tool '{tool_name}': {exc}"

    if isinstance(exc, TimeoutError):
        return (
            f"Tool '{tool_name}' timed out. "
            "Please try again later."
        )

    raise exc


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[divide],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            retry_on=(TimeoutError, ConnectionError),
        ),
        ToolErrorMiddleware(error_handler),
    ],
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Use the divide tool to calculate 10 divided by 0"
            }
        ]
    }
)

print(response)
