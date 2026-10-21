
from dotenv import load_dotenv
load_dotenv()
from langgraph.types import Command
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import(PIIMiddleware)
from common import check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map
from utils import pretty_print_messages,run_interactive_hitl_demo
from langgraph.checkpoint.memory import InMemorySaver

cinebot_tools = [check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map]

pii_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=cinebot_tools,
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    ],
)
result = pii_agent.invoke({
    "messages": [("user", "My email is priya@example.com and my credit card is 4111-1111-1111-1234, can you check showtimes for Dune?")]
})

print(result['messages'][-1].content)