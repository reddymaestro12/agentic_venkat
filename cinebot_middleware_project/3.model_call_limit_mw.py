from dotenv import load_dotenv
load_dotenv()
from langgraph.types import Command
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import(ModelCallLimitMiddleware)
from common import check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map
from utils import pretty_print_messages,run_interactive_hitl_demo
from langgraph.checkpoint.memory import InMemorySaver

cinebot_tools = [check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map]

call_limited_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=cinebot_tools,
    checkpointer=InMemorySaver(),  # required for thread_limit to persist across calls
    middleware=[
        ModelCallLimitMiddleware( # low values taken just for example
            thread_limit=5,   # across the WHOLE conversation
            run_limit=2,       # per single .invoke() call
            exit_behavior="end",  # graceful stop, not an exception
        ),
    ],
)
result1 = call_limited_agent.invoke(
    {"messages": [("user", "Can you tell me cinema's refund policy? ")]},
    config={"configurable": {"thread_id": "call-limit-demo-4"}},
)

result_invoke_2= call_limited_agent.invoke(
    {"messages": [("user", "cancel my booking B123? ")]},
    config={"configurable": {"thread_id": "call-limit-demo-4"}},
)


result_invoke3 = call_limited_agent.invoke(
    {"messages": [("user", "What all movies are being shown? ")]},
    config={"configurable": {"thread_id": "call-limit-demo-4"}},
)

result_invoke4 = call_limited_agent.invoke(
    {"messages": [("user", "Summarize my chat? ")]},
    config={"configurable": {"thread_id": "call-limit-demo-4"}},
)


print(result1)