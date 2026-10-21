from dotenv import load_dotenv
load_dotenv()
from langgraph.types import Command
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import(ToolCallLimitMiddleware)
from common import check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map
from utils import pretty_print_messages,run_interactive_hitl_demo
from langgraph.checkpoint.memory import InMemorySaver

cinebot_tools = [check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map]
tool_limited_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=cinebot_tools,
    checkpointer=InMemorySaver(),
    middleware=[
        ToolCallLimitMiddleware(run_limit=8),                              # global, this turn
        ToolCallLimitMiddleware(tool_name="cancel_booking", thread_limit=2),  # tighter, one tool, whole conversation
    ],
)

print(tool_limited_agent)

tool_limited_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=cinebot_tools,
    checkpointer=InMemorySaver(),
    middleware=[
        ToolCallLimitMiddleware(run_limit=8),                              # global, this turn
        ToolCallLimitMiddleware(tool_name="cancel_booking", thread_limit=2, run_limit=1),  # tighter, one tool, whole conversation
    ],
)

config = {"configurable": {"thread_id": "tool-limit-demo"}}
for i in range(3):
  result = tool_limited_agent.invoke({"messages": [("user", f"Please cancel my Booking with ID B{100+i} ? ")]}, config=config)
  print(result)