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


attempt = 0

@tool
def get_balance1(account_id: str) -> str:
    """Returns account balance."""

    global attempt
    attempt += 1

    print(f"Attempt #{attempt}")

    raise ConnectionError("Database unavailable")

attempt = 0

@tool
def get_balance2(account_id: str) -> str:
    """Returns account balance."""

    global attempt
    attempt += 1

    try:
        raise ConnectionError("Database unavailable")
    except Exception as e:
        print(
            f"[Retry {attempt}] "
            f"Tool=get_balance "
            f"Error={type(e).__name__}: {e}"
        )
        raise


def error_handler(
    exc: Exception,
    request: ToolCallRequest,
) -> Optional[str]:

    tool_name = request.tool_call["name"]

    if isinstance(exc, ConnectionError):
        return (
            f"Database is currently unavailable for "
            f"'{tool_name}'. Please try again later."
        )

    raise exc

attempt = 0

@tool
def get_balance3(account_id: str) -> str:
    """Returns account balance."""

    global attempt
    attempt += 1

    try:
        raise ConnectionError("Database unavailable")
    except Exception as e:
        print(
            f"[Retry {attempt}] "
            f"Tool=get_balance "
            f"Error={type(e).__name__}: {e}"
        )
        raise

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_balance3,],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            retry_on=(ConnectionError,),
        ),
        ToolErrorMiddleware(error_handler),
    ],
)
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the balance of account 123?"
            }
        ]
    }
)

print(response)
