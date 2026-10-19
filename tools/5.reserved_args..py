
"""
The main purpose of ToolRuntime is to give a tool access to execution-time information without making that information part of the tool's schema that the LLM must generate.

"""
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

import os
from dotenv import load_dotenv
load_dotenv()


@tool
def greet_runtime(name: str, runtime: ToolRuntime) -> str:
    """Greet a person."""

    print(f"Runtime object type: {type(runtime)}")

    return f"Hello {name}"

from langchain_core.tools import tool

@tool
def greet_config(name: str, config: dict):
    """this is the config of dict type not the ToolRuntime time"""
    return f"Hello {name}"


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[greet_config]
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Greet Ravi"
            }
        ]
    }
)

print(response["messages"][-1].content)