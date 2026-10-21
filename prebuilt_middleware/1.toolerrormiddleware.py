from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents.middleware.types import ToolCallRequest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware,ToolRetryMiddleware

@tool
def divide(a:int,b:int)->int:
    "perofrms division operation"
    if b==0:
        raise ValueError("denominator cant not be zero")
    return a//b

def error_handler(
    exc: Exception,
    request: ToolCallRequest
) -> Optional[str]:

    tool_name = request.tool_call["name"]

    if isinstance(exc, ValueError):
        return f"Input error in tool '{tool_name}': {exc}"

    if isinstance(exc, TimeoutError):
        return (
            f"Tool '{tool_name}' timed out. "
            "Please try again later."
        )

    # Let LangChain handle unexpected errors
    raise exc

# Create agent
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[divide],
    middleware=[ToolErrorMiddleware(error_handler) ]
)


# Invoke agent
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 10 divided by 0?"
            }
        ]
    }
)

print(response)