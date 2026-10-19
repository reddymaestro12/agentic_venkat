from langchain.tools import tool

#Return a string when the tool should provide plain text for the model to read and use in its next response.
#Use this when the result is naturally human-readable text.
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It is currently sunny in {city}."

"""
Return an object (for example, a dict) when your tool produces structured data that the model should inspect.
Behavior:
The object is serialized and sent back as tool output.
The model can read specific fields and reason over them.
Like string returns, this does not directly update graph state.
Use this when downstream reasoning benefits from explicit fields instead of free-form text.
"""
@tool
def get_weather_data(city: str) -> dict:
    """Get structured weather data for a city."""
    return {
        "city": city,
        "temperature_c": 22,
        "conditions": "sunny",
    }

"""
Tools are not limited to plain text. When the model supports multimodal tool results, the tool can return standard content blocks so the model receives text, images, and other media in one tool result.
"""
@tool
def capture_screenshot() -> list[dict]:
    """Capture a screenshot of the current page."""
    return [
        {"type": "text", "text": "Screenshot of the current page:"},
        {"type": "image", "url": "https://example.com/page.png"},
    ]

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

tool
def set_language(language: str, runtime: ToolRuntime) -> Command:
    """Set the preferred response language."""
    return Command(
        update={
            "preferred_language": language,
            "messages": [
                ToolMessage(
                    content=f"Language set to {language}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


"""

Return directly from a tool
Set return direct on a tool to short-circuit the agent loop: the agent returns the tool’s output to the caller immediately, 
without sending it back through the model for further processing.

"""
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


@tool(return_direct=True)
def fetch_order_status(order_id: str) -> str:
    """Fetch the current status of a customer order."""
    # In production, query your order management system here
    return f"Order {order_id} is shipped and will arrive in 2 days."


agent = create_agent(
    ChatOpenAI(model="gpt-4o-mini"),
    tools=[fetch_order_status],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the status of order #12345?"}]
})
# The agent returns the tool output directly without another LLM call:
# "Order 12345 is shipped and will arrive in 2 days."