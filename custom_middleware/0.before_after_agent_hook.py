
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_agent,
    after_agent,
    AgentState,
)
from langgraph.runtime import Runtime

load_dotenv()


# 1. Runs before the agent starts
@before_agent
def log_before_agent(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    messages = state["messages"]

    print("\n[BEFORE AGENT]")
    print(f"Messages received: {len(messages)}")

    if messages:
        print(f"Latest message: {messages[-1].content}")

    return None


# 2. Runs after the agent finishes
@after_agent
def log_after_agent(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    messages = state["messages"]

    print("\n[AFTER AGENT]")
    print(f"Total messages: {len(messages)}")

    if messages:
        print(f"Final message: {messages[-1].content}")

    return None


# 3. Create the agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    middleware=[
        log_before_agent,
        log_after_agent,
    ],
    system_prompt="You are a helpful assistant.",
)


# 4. Invoke the agent
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain the difference between before_agent and after_agent.",
            }
        ]
    }
)


# 5. Print the final answer
print("\n[FINAL RESULT]")
print(result["messages"][-1].content)
