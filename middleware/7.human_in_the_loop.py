from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from langgraph.checkpoint.memory import InMemorySaver


# ------------------------------------
# Tool 1
# ------------------------------------

@tool
def read_email_tool(email_id: str) -> str:
    """Read email content."""

    return f"Email content for ID: {email_id}"


# ------------------------------------
# Tool 2
# ------------------------------------

@tool
def send_email_tool(
    recipient: str,
    subject: str,
    body: str,
) -> str:
    """Send an email."""

    return (
        f"Email sent to {recipient} "
        f"with subject '{subject}'"
    )


# ------------------------------------
# Agent
# ------------------------------------

agent = create_agent(
    model="openai:gpt-4o-mini",

    tools=[
        read_email_tool,
        send_email_tool,
    ],

    checkpointer=InMemorySaver(),

    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": [
                        "approve",
                        "edit",
                        "reject",
                    ]
                },

                "read_email_tool": False,
            }
        )
    ],
)
config = {
    "configurable": {
        "thread_id": "email-thread-1"
    }
}

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content":
                (
                    "Send an email to "
                    "john@example.com "
                    "saying hello"
                )
            }
        ]
    },
    config=config, # type: ignore
)

print(result)
if "__interrupt__" in result:
    print("Human approval required")
    print(result["__interrupt__"])