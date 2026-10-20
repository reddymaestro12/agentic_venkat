from langchain.agents import AgentState
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from dotenv import load_dotenv
load_dotenv()

class CustomState(AgentState):
    user_name: str


@tool
def set_user_name(new_name: str, runtime: ToolRuntime[None, CustomState]) -> Command:
    """Set the user's name in the conversation state."""
    return Command(
        update={
            "user_name": new_name,
            "messages": [
                ToolMessage(
                    content=f"User name set to {new_name}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

from langchain.agents import create_agent
agent = create_agent(model="openai:gpt-5-mini",tools=[set_user_name],state_schema=CustomState)
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "My name is John. Please save it."
            }
        ]
    }
)

for msg in result["messages"]:
    print(type(msg).__name__)
    print(msg.content)
    print("-" * 50)