from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver


@tool
def search(query: str) -> str:
    """Search for information."""
    print(f"\n[SEARCH TOOL CALLED] query={query}")
    return f"Results for: {query}"


@tool
def database(table: str) -> str:
    """Read data from database."""
    print(f"\n[DATABASE TOOL CALLED] table={table}")
    return f"Rows returned from {table}"


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search, database],
    checkpointer=InMemorySaver(),
    middleware=[
        # Limit all tool calls
        ToolCallLimitMiddleware(
            thread_limit=20,
            run_limit=10,
        ),

        # Limit search tool only
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=5,
            run_limit=3,
        ),
    ],
)

config = {
    "configurable": {
        "thread_id": "tool-limit-demo"
    }
}

queries = [
    "Search for AI",
    "Search for LangChain",
    "Search for LangGraph",
    "Search for RAG",
    "Search for Agents",
    "Read users table"
]

for i, query in enumerate(queries, 1):
    print(f"\n{'='*50}")
    print(f"REQUEST {i}")
    print(f"{'='*50}")

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            },
            config=config
        )

        print("\nAssistant:")
        print(result["messages"][-1].content)

    except Exception as e:
        print(f"\nLIMIT REACHED: {e}")
