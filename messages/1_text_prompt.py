import logging
import sys
from pathlib import Path

from langchain.agents import create_agent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import load_config


def main() -> None:
    try:
        load_config()

        agent = create_agent(
            model="openai:gpt-4o-mini"
        )

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a haiku about spring"
                    }
                ]
            }
        )

        print(f"Response Type: {type(response)}")

        print("\nResponse Object:")
        print(response)

        if "messages" in response:
            print("\nGenerated Content:")
            print(response["messages"][-1].content)

    except Exception:
        logging.exception("Agent execution failed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()