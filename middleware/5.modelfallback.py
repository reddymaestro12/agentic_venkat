from dotenv import load_dotenv

load_dotenv()

from typing import Optional

from langchain.agents import create_agent
from langchain.tools import tool

from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ModelFallbackMiddleware,
    ToolRetryMiddleware,
    ToolErrorMiddleware,
)

from langchain.agents.middleware.types import ToolCallRequest


# =====================================================
# TOOL 1 : SEARCH
# =====================================================

search_attempt = 0


@tool
def search_tool(query: str) -> str:
    """
    Search public information.
    """

    global search_attempt

    search_attempt += 1

    print(f"[SEARCH] Attempt #{search_attempt}")

    if search_attempt < 3:
        raise ConnectionError(
            "Search service temporarily unavailable"
        )

    return f"Search results found for '{query}'"


# =====================================================
# TOOL 2 : DATABASE
# =====================================================

db_attempt = 0


@tool
def database_tool(customer_id: str) -> str:
    """
    Lookup account information.
    """

    global db_attempt

    db_attempt += 1

    print(f"[DATABASE] Attempt #{db_attempt}")

    raise ConnectionError(
        "Database connection unavailable"
    )


# =====================================================
# TOOL ERROR HANDLER
# =====================================================

def tool_error_handler(
    exc: Exception,
    request: ToolCallRequest,
) -> Optional[str]:

    tool_name = request.tool_call["name"]

    if isinstance(exc, ConnectionError):
        return (
            f"Tool '{tool_name}' failed after all "
            f"retry attempts.\n"
            f"Reason: {exc}"
        )

    if isinstance(exc, TimeoutError):
        return (
            f"Tool '{tool_name}' timed out "
            f"after multiple attempts."
        )

    raise exc


# =====================================================
# MODEL FAILURE HANDLER
# =====================================================

def model_error_handler(
    exc: Exception,
) -> str:

    return (
        f"All model retries failed.\n"
        f"Error Type: {type(exc).__name__}\n"
        f"Message: {exc}"
    )


# =====================================================
# CREATE AGENT
# =====================================================

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[
        search_tool,
        database_tool,
    ],
    middleware=[

        # ---------------------------------
        # Retry failed model calls
        # ---------------------------------
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            on_failure=model_error_handler,
        ),

        # ---------------------------------
        # Fallback to another model if
        # primary model ultimately fails
        # ---------------------------------
        ModelFallbackMiddleware(
            "openai:gpt-4.1-mini",
            "openai:gpt-4o-mini",
        ),

        # ---------------------------------
        # Retry tool failures
        # ---------------------------------
        ToolRetryMiddleware(
            max_retries=3,
            retry_on=(
                ConnectionError,
                TimeoutError,
            ),
            on_failure="error",
        ),

        # ---------------------------------
        # Handle tool failures gracefully
        # ---------------------------------
        ToolErrorMiddleware(
            tool_error_handler
        ),
    ],
)


# =====================================================
# INVOKE AGENT
# =====================================================

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content":
                (
                    "Use the search tool to search "
                    "for LangChain middleware"
                ),
            }
        ]
    }
)

print("\n===== FINAL RESPONSE =====\n")

for message in response["messages"]:
    print(type(message).__name__)
    print(message.content)
    print()