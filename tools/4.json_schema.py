from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()


weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location", "units", "include_forecast"]
}

@tool(args_schema=weather_schema)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result

from langchain.agents import create_agent
agent = create_agent(
    model = "openai:gpt-5-mini",
    tools = [get_weather]
)
result =  agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the weather in Bangalore?"
        }
    ]
})
for i, msg in enumerate(result["messages"]):
    if i==2:
        print(f"\nMessage {i}")
        print(type(msg))
        print(msg)