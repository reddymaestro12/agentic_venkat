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

        agent = create_agent(model="openai:gpt-4o-mini")
        response = agent.invoke(
             {
               "messages": [
                    HumanMessage(content="write a haiku about spring")
                ]
            }
        )
        last_message = response["messages"][-1]
        print(f"Content is {last_message.content}")
        print(f"metadata is :{last_message.response_metadata}")
        print(f"usage metadata is :{last_message.usage_metadata}")

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