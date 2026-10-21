
from typing import Callable, Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ToolCallRequest,
)
from langchain.messages import ToolMessage
from langchain.tools import tool

load_dotenv()


# 1. Define a tool
@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide a by b and return the result."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


# 2. Define middleware around tool execution
class ToolLoggingMiddleware(AgentMiddleware):

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:

        tool_name = request.tool_call["name"]
        tool_args = request.tool_call["args"]

        print(f"\n[TOOL START] {tool_name}")
        print(f"Arguments: {tool_args}")

        try:
            result = handler(request)

            print(f"[TOOL SUCCESS] {tool_name}")
            print(f"Result: {result.content}")

            return result

        except Exception as exc:
            print(
                f"[TOOL ERROR] {tool_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            raise


# 3. Create the agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[divide_numbers],
    middleware=[ToolLoggingMiddleware()],
    system_prompt=(
        "You are a helpful math assistant. "
        "Use the divide_numbers tool for division. "
        "If division fails, explain the error to the user."
    ),
)


# 4. Invoke the agent
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 100 divided by 5?",
            }
        ]
    }
)


# 5. Print the final answer
print("\n[FINAL ANSWER]")
print(result["messages"][-1].content)
