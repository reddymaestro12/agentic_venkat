from typing import Annotated
from langchain.agents import create_agent
from langchain.tools import tool,ToolRuntime

from dotenv import load_dotenv
load_dotenv()


from typing import Annotated
from langchain.tools import tool,ToolRuntime

#weather tool accessing location
@tool
def get_weather(
    runtime: Annotated[ToolRuntime, "Tool runtime"]
) -> str:
    """
    Get weather using user's saved city.
    """

    city = runtime.state.get("city", "Unknown")

    return f"Fetching weather for {city}"


@tool
def get_user_id(runtime:ToolRuntime):
    "agent + tool state"
    return runtime.state.get("user_id","user id is not found")

agent = create_agent(model="openai:gpt-5-mini",tools=[get_user_id,get_weather])
state ={"user_id":"USER123"}
state1 = {
    "city": "Bengaluru"
}
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Bangalore?"
            }
        ]
    },
    config={
        "configurable": {
            "state": state1
        }
    }
)
print(result)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is my user id?"
            }
        ]
    },
    config={
        "configurable": {
            "state": state
        }
    }
)

print(response)