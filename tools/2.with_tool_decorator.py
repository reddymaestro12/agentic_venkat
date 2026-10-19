"""
Key takeaway: @tool exposes your Python function to the model,
while create_agent() automatically manages tool execution and sends the result back to the model.
"""
import os
from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """adds the weather information for the given city"""
    return f"The weather in {city} is sunny."

#by default function name is the tool name- if we want to override/customize the tool name
@tool("customized_weather")
def get_weather1(city: str) -> str:
    """adds the weather information for the given city"""
    return f"The weather in {city} is sunny."

#Custom tool description:Override the auto-generated tool description for clearer model guidance:
@tool("customized_weather2",description="take the city an provide the detailed information")
def get_weather2(city: str) -> str:
    """adds the weather information for the given city"""
    return f"The weather in {city} is sunny."

from langchain.agents import create_agent

agent = create_agent(
    model = "openai:gpt-5-mini",
    tools = [get_weather2]
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

print("tool name",get_weather2.name)
print("tool description",get_weather2.description)
print("tool args",get_weather2.args)
for i, msg in enumerate(result["messages"]):
    if i==2:
        print(f"\nMessage {i}")
        print(type(msg))
        print(msg)