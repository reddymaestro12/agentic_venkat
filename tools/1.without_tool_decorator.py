"""
in the Langchain latest versions python function will be converted as a tool internally
but using @tool is the best recommendation
"""

import os
from dotenv import load_dotenv

load_dotenv()

def get_weather(city: str) -> str:
    """adds the weather information for the given city"""
    return f"The weather in {city} is sunny."

from langchain.agents import create_agent

agent = create_agent(
    model = "openai:gpt-5-mini",
    tools = [get_weather]
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the weather in Bangalore?"
        }
    ]
})
response = result["messages"][-1]

for i, msg in enumerate(result["messages"]):
    print(f"\nMessage {i}")
    print(type(msg))
    print(msg)