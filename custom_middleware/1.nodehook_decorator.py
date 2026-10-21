
from dotenv import load_dotenv
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_model,
    after_model,
    AgentState,
)
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

load_dotenv()


# 1. Stop the agent when the message limit is reached
@before_model(can_jump_to=["end"])
def check_message_limit(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:

    if len(state["messages"]) >= 50:
        return {
            "messages": [
                AIMessage(content="Conversation limit reached.")
            ],
            "jump_to": "end",
        }

    return None


# 2. Log the model response
@after_model
def log_response(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:

    last_message = state["messages"][-1]

    print(f"Model returned: {last_message.content}")

    return None


# 3. Create the agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    middleware=[
        check_message_limit,
        log_response,
    ],
    system_prompt="You are a helpful assistant.",
)


# 4. Invoke the agent
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain what middleware does in LangChain.",
            }
        ]
    }
)


# 5. Print the final response
print("\nFinal response:")
print(result["messages"][-1].content)
