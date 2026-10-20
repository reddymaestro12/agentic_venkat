from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# Create the agent
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,  # Across thread
            run_limit=5,      # Per invocation
            exit_behavior="end",
        )
    ],
)

# Required because we're using a checkpointer
config = {
    "configurable": {
        "thread_id": "user-1"
    }
}

# First conversation
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write a short poem about AI."
            }
        ]
    },
    config=config
)

print(result["messages"][-1].content)

# Continue same thread
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Now make it funny."
            }
        ]
    },
    config=config
)

print(result["messages"][-1].content)