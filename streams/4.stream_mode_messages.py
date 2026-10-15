import logging

from dotenv import load_dotenv
from uuid6 import uuid7

from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's always sunny in {city}!"


def main():

    try:
        load_dotenv()

        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[get_weather],
            checkpointer=InMemorySaver()
        )

        config = {
            "configurable": {
                "thread_id": str(uuid7())
            }
        }

        print("\nStreaming Response:\n")
        print("-" * 60)

        for token, metadata in agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather in SF?"
                    }
                ]
            },
            config=config,
            stream_mode="messages"
        ):

            # Print metadata
            print(
                f"\nNode: {metadata.get('langgraph_node')}"
            )

            # Print generated token
            if token.content:
                print(token.content, end="", flush=True)

    except Exception:
        logging.exception("Agent execution failed")


if __name__ == "__main__":
    main()