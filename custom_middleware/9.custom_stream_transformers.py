
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langgraph.stream import (
    ProtocolEvent,
    StreamChannel,
    StreamTransformer,
)
from dotenv import load_dotenv
load_dotenv()

class ToolActivity(TypedDict):
    name: str
    status: str


class ToolActivityTransformer(StreamTransformer):
    required_stream_modes = ("tools",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.activity = StreamChannel[ToolActivity]("tool_activity")

    def init(self) -> dict:
        return {"tool_activity": self.activity}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "tools":
            return True

        data = event["params"]["data"]

        if isinstance(data, dict):
            tool_name = data.get("tool_name")
            event_type = data.get("event")

            if tool_name and event_type:
                status = {
                    "tool-started": "started",
                    "tool-finished": "finished",
                    "tool-error": "error",
                }.get(event_type)

                if status:
                    self.activity.push(
                        {
                            "name": tool_name,
                            "status": status,
                        }
                    )

        return True


class ToolActivityMiddleware(AgentMiddleware):
    transformers = (ToolActivityTransformer,)


@tool
def get_weather(city: str) -> str:
    """Return sample weather information for a city."""
    return f"It's sunny in {city}."


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[ToolActivityMiddleware()],
    system_prompt="Answer weather questions using the weather tool.",
)

stream = agent.stream_events(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Bengaluru?",
            }
        ]
    },
    version="v3",
)

# Consume the custom tool activity projection
for activity in stream.extensions["tool_activity"]:
    print(activity)

# Read the final agent state after consuming the stream
final_state = stream.output
print("\nFinal answer:")
print(final_state["messages"][-1].content)