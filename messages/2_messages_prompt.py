import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import load_config
from langchain.agents import create_agent
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)


def main() -> None:
    try:
        if not load_config():
            raise RuntimeError(
                "Configuration loading failed. "
                "Check your .env file and required variables."
            )

        messages = [
            SystemMessage(
                content="You are a poetry expert."
            ),
            HumanMessage(
                content="Write a haiku about spring."
            ),
        ]

        agent = create_agent(
            model="openai:gpt-4o-mini"
        )

        response = agent.invoke(
            {"messages": messages}
        )

        if "messages" in response:
            print("\nGenerated Content:")
            print(
                response["messages"][-1].content
            )

    except Exception:
        logging.exception(
            "Agent execution failed."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main()