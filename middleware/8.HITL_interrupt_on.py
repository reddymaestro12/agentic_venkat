from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return f"Wrote {len(content)} characters to {path}"

@tool
def execute_sql(query: str) -> str:
    """Run a SQL query."""
    return "Deleted 42 rows"

@tool
def read_data(table: str) -> str:
    """Read rows from a table."""
    return f"3 rows from {table}"

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[write_file, execute_sql, read_data],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,
                "execute_sql": {
                    "allowed_decisions": ["approve", "reject"]
                },
                "read_data": False,
            },
            
        ),
    ],
    checkpointer=InMemorySaver(),
)

config = {
    "configurable": {
        "thread_id": "thread-1"
    }
}

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Delete all rows from customers table"
            }
        ]
    },
    config=config
)

print(response)