from dotenv import load_dotenv
load_dotenv()
from langgraph.types import Command
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import(HumanInTheLoopMiddleware)
from common import check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map
from utils import pretty_print_messages,run_interactive_hitl_demo
from langgraph.checkpoint.memory import InMemorySaver

cinebot_tools = [check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map]

guarded_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=cinebot_tools,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"cancel_booking": {"allowed_decisions": ["approve", "edit", "reject", "respond"]}}
        ),
    ],
    checkpointer=InMemorySaver(),  # REQUIRED -- HITL needs to pause and later resume
)

config = {'configurable':{'thread_id':'hitl-demo-live'}}

result = guarded_agent.invoke({"messages": [("user", "Please cancel booking BK1042")]}, config=config)
from rich import print
# print(result)
# state = guarded_agent.get_state(result)
# print(config)
resumed_result = guarded_agent.invoke(Command(resume={"decisions":[{"type":"approve"}]}),config=config)
print(resumed_result)