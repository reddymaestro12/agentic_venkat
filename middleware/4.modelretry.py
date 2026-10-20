from dotenv import load_dotenv
load_dotenv()

from typing import Optional

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    ToolErrorMiddleware,
)
from langchain.agents.middleware.types import ToolCallRequest


# --------------------------------------------------
# Search Tool
# --------------------------------------------------

search_attempt = 0

@tool
def search_tool(query: str) -> str:
    """
    Search the knowledge base.
    """

    global search_attempt
    search_attempt += 1

    print(f"[SEARCH] Attempt #{search_attempt}")

    # Simulate temporary API failure
    if search_attempt < 3:
        raise ConnectionError("Search API unavailable")

    return f"Search results for '{query}'"


# --------------------------------------------------
# Database Tool
# --------------------------------------------------

db_attempt = 0

@tool
def database_tool(account_id: str) -> str:
    """
    Retrieve account information.
    """

    global db_attempt
    db_attempt += 1

    print(f"[DATABASE] Attempt #{db_attempt}")

    # Simulate database outage
    raise ConnectionError("Database unavailable")


# --------------------------------------------------
# Tool Error Handler
# --------------------------------------------------

def tool_error_handler(
    exc: Exception,
    request: ToolCallRequest,
) -> Optional[str]:

    tool_name = request.tool_call["name"]

    if isinstance(exc, ConnectionError):
        return (
            f"Tool '{tool_name}' failed after all retries. "
            f"Reason: {exc}"
        )

    if isinstance(exc, TimeoutError):
        return (
            f"Tool '{tool_name}' timed out after "
            f"multiple retry attempts."
        )

    raise exc


# --------------------------------------------------
# Model Error Handler
# --------------------------------------------------

def model_error_handler(exc: Exception) -> str:
    return (
        f"Model call failed after all retries. "
        f"Error: {type(exc).__name__}: {exc}"
    )


# --------------------------------------------------
# Create Agent
# --------------------------------------------------

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[
        search_tool,
        database_tool,
    ],
    middleware=[

        # Retries model/API failures
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            on_failure=model_error_handler,
        ),

        # Retries tool failures
        ToolRetryMiddleware(
            max_retries=3,
            retry_on=(
                ConnectionError,
                TimeoutError,
            ),
            on_failure="error",
        ),

        # Handles tool failures after retries
        ToolErrorMiddleware(
            tool_error_handler
        ),
    ]
)


# --------------------------------------------------
# Invoke Agent
# --------------------------------------------------

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the database tool to get "
                    "the balance for account 12345"
                ),
            }
        ]
    }
)

print("\n===== FINAL RESPONSE =====\n")
print(response)