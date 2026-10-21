from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import(SummarizationMiddleware)
from common import check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map
from utils import pretty_print_messages

cinebot_tools = [check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map]

summarizing_agent = create_agent(
    model="gpt-5.4-mini",
    tools=cinebot_tools,
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("tokens", 300),
            keep=("messages", 1),
        )
    ],
)

#print(summarizing_agent.invoke({"messages": [("user", "Hi I am mayank ")]}))
#print(summarizing_agent.invoke({"messages": [("user", "Who am I ? ")]}))
result = summarizing_agent.invoke({"messages": [("user", "Is Interstellar showing tonight? also please make sure that you book me a ticket, refund me if it is not available,also share the refund policy for me to go through, also check my order status for book_1234")]})
pretty_print_messages(result)