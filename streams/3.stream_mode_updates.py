from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
import logging
import time
logger = logging.getLogger()
def get_weather(city):
    "provides weather information"
    return f"the weather is sunny in {city}"

def main():
    try:
        load_dotenv()

        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[get_weather],
            checkpointer=InMemorySaver(),
        )

        config = {
            "configurable": {
                "thread_id": str(uuid7())
            }
        }

        inputs = {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in SF?",
                }
            ]
        }
        for chunk in agent.stream(inputs,
                                  config=config,
                                  stream_mode="updates",#nstead of the entire state, you get only what changed.
                                ):
            for node_name,update in chunk.items():
                 logger.info("Agent node updated: %s", node_name)
                 for message in update.get("messages", []):
                    message.pretty_print()

    except Exception:
        logging.exception("Agent execution failed")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()