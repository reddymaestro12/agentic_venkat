from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
import logging
import time
def get_weather(city):
    "provides weather information"
    return f"the weather is sunny in {city}"

def main():
    try:
        load_dotenv()
        agent = create_agent(model="openai:gpt-4o-mini",tools=[get_weather],checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": str(uuid7())}}
        chunks =[]
        final_message =None
        for chunk in  agent.stream(
            {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
            config=config,
            version="v3",  ):
            print(chunk)
            chunks.append(chunk)
            time.sleep(1)
          
    except Exception:
        logging.exception("Agent execution is failed") 

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()