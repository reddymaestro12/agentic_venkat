
from dotenv import load_dotenv
load_dotenv()
from langgraph.types import Command
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import(PIIMiddleware)
from common import check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map,detect_booking_code
from utils import pretty_print_messages,run_interactive_hitl_demo
from langgraph.checkpoint.memory import InMemorySaver

cinebot_tools = [check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map,detect_booking_code]

custom_pii_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=cinebot_tools,
    middleware=[PIIMiddleware("booking_code", detector=detect_booking_code, strategy="mask")],
)

result = custom_pii_agent.invoke({
    "messages": [("user", "Can you check the status of my booking BK1044 for me?")]
})

print(result)