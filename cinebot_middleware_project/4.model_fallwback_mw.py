from dotenv import load_dotenv
load_dotenv()
from IPython.display import Image, display
from langgraph.types import Command
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import(ModelFallbackMiddleware)
from common import check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map
from utils import pretty_print_messages,run_interactive_hitl_demo
from langgraph.checkpoint.memory import InMemorySaver

cinebot_tools = [check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map]
resilient_agent = create_agent(
    model="openai:gpt-5.5-haiku",     # primary, most capable
    tools=cinebot_tools,
    middleware=[
        ModelFallbackMiddleware(
            "openai:gpt-5-mini",   # fallback -- cheaper, still OpenAI, needs no extra setup
            # "ollama:llama3.2",   # a further, fully-local last resort -- uncomment if you have
                                    # `pip install langchain-ollama` AND a local Ollama server running.
                                    # Left commented here so this cell runs with nothing beyond
                                    # what Setup already installed.
        ),
    ],
)

result = resilient_agent.invoke( {"messages": [("user", "Summarize my chat? ")]},)
print(result)
print("Fallback chain: gpt-5.5 -> gpt-5-mini.")
print("If the primary model call fails for any reason, this silently tries the next one.")
graph = resilient_agent.get_graph()

display(Image(graph.draw_mermaid_png()))