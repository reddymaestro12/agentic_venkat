from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ToolErrorMiddleware
)
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()


@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    return a / b


class LoggingMiddleware(AgentMiddleware):

    def before_model(self, state, runtime):
        print("Before model call")

    def after_model(self, state, runtime):
        print("After model call")
        


def handle_tool_error(exc: Exception, request):
    return f"Tool failed: {str(exc)}"


agent = create_agent(
    model="gpt-4o-mini",
    tools=[divide],
    middleware=[
        LoggingMiddleware(),
        ToolErrorMiddleware(on_error=handle_tool_error)
    ]
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "divide 10 by 0"
        }
    ]
})

print(result["messages"][-1].content)