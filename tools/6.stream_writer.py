"""
        Stream real-time updates from tools during execution. This is useful for providing progress feedback to users during long-running operations.
        Use runtime.stream_writer to emit custom updates
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import  create_agent
from langchain.tools import tool,ToolRuntime

@tool
def get_weather(city:str,runtime:ToolRuntime)-> str:
    """chekcing the stream writer fruncitonality"""
    writer =runtime.stream_writer
 
    # Stream custom updates as the tool executes
    writer(f"looking up data for city :{city}")
    writer(f"acquired data for the city :{city}")

    return f"it is always sunny in the city {city}"

#Requires deepagents>=0.5.0 (or langgraph>=1.1.5).
@tool
def log_execution_context(runtime: ToolRuntime) -> str:
    """Log execution identity information."""
    info = runtime.execution_info
    print(f"Thread: {info.thread_id}, Run: {info.run_id}")
    print(f"Attempt: {info.node_attempt}")
    return "done"

agent = create_agent(   model="openai:gpt-5-mini",tools=[log_execution_context])
result =  agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the weather in Bangalore?"
        }
    ]
})

print(result["messages"][-1])
    

