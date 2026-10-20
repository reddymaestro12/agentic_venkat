from dotenv import load_dotenv
load_dotenv()

from typing import Optional
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import (
    SummarizationMiddleware,ToolCallRequest,ToolErrorMiddleware
)

@tool
def weather_tool(city: str) -> str:
    """
    Get weather information for a city.
    """

    weather_data = {
        "bangalore": "25°C, Sunny",
        "hyderabad": "30°C, Partly Cloudy",
        "chennai": "33°C, Humid",
        "mumbai": "29°C, Rainy",
    }

    result = weather_data.get(city.lower())

    if result:
        return f"Weather in {city}: {result}"

    return f"No weather information available for {city}"


@tool
def calculator_tool(
    operation: str,
    a: float,
    b: float,
) -> float:
    """
    Perform basic arithmetic operations.
    """

    operation = operation.lower()

    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":

        if b == 0:
            raise ValueError(
                "Division by zero is not allowed."
            )

        return a / b

    raise ValueError(
        f"Unsupported operation: {operation}"
    )

from typing import Optional
from langchain.agents.middleware.types import ToolCallRequest


def tool_error_handler(
    exc: Exception,
    request: ToolCallRequest,
) -> Optional[str]:

    tool_name = request.tool_call["name"]

    if isinstance(exc, ValueError):
        return (
            f"Tool '{tool_name}' failed: "
            f"{exc}"
        )

    raise exc

#create agent - # Single condition: trigger if tokens >= 4000
agent1 = create_agent(
    model="gpt-4o-mini",
    tools=[weather_tool,calculator_tool],
    middleware=[
        ToolErrorMiddleware(tool_error_handler),
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens",4000),
            keep=("messages",20)
        )
    ]
)

# Multiple conditions: trigger if number of tokens >= 3000 OR messages >= 6
#we can use different model for summarization and agent model
agent2 = create_agent(
    model="gpt-5.5",
    tools=[weather_tool,calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=[
                ("tokens", 3000),
                ("messages", 6),
            ],
            keep=("messages", 20),
        ),
    ],
)

#AND logic: trigger only when tokens >= 4000 AND messages >= 10
agent3 = create_agent(
    model="gpt-5.5",
    tools=[weather_tool, calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger={"tokens": 4000, "messages": 10},
            keep=("messages", 20),
        ),
    ],
)

# Combine AND and OR: trigger if (tokens >= 5000 AND messages >= 3)
# OR (tokens >= 3000 AND messages >= 6)
agent4 = create_agent(
    model="gpt-5.5",
    tools=[weather_tool, calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=[
                {"tokens": 5000, "messages": 3},
                {"tokens": 3000, "messages": 6},
            ],
            keep=("messages", 20),
        ),
    ],
)

# Using fractional limits
agent5 = create_agent(
    model="gpt-5.5",
    tools=[weather_tool,calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("fraction", 0.8),
            keep=("fraction", 0.3),
        ),
    ],
)

#inovke the agent
response = agent5.invoke(
    {
        "messages":[
            {
                "role":"user",
                "content":"what is the weather in bagalore"
            }
            
        ]
    }
)

print(response["messages"][-1].content)