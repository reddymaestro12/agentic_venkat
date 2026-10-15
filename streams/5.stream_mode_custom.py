
import logging
import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@tool
def get_weather(city: str) -> str:
    """Get mock weather information for a specified city."""

    writer = get_stream_writer()

    # Custom progress update 1
    writer({
        "stage": "started",
        "message": f"Starting weather lookup for {city}",
    })

    # Custom progress update 2
    city = city.strip()

    if not city:
        writer({
            "stage": "error",
            "message": "City name is missing.",
        })
        return "Please provide a valid city name."

    writer({
        "stage": "processing",
        "message": f"Validating city: {city}",
    })

    # Mock weather lookup — no real weather API is called
    weather = f"The weather is sunny in {city}."

    # Custom progress update 3
    writer({
        "stage": "completed",
        "message": f"Weather lookup completed for {city}",
    })

    return weather


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not configured.")

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather],
        checkpointer=InMemorySaver(),
        system_prompt=(
            "You are a helpful weather assistant. "
            "Use get_weather when asked about weather. "
            "The tool returns mock weather data."
        ),
    )

    config = {
        "configurable": {
            "thread_id": str(uuid4())
        }
    }

    inputs = {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Bengaluru?",
            }
        ]
    }

    # Stream only the custom updates emitted by the tool
    for chunk in agent.stream(
        inputs,
        config=config,
        stream_mode="custom",
    ):
        print("CUSTOM UPDATE:", chunk)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Agent execution failed")
        raise
