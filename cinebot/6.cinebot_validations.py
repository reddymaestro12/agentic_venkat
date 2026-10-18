"""
The Problem: What If a Value Genuinely Violates a Constraint?
Schemas can include real constraints, not just shapes — like "between 1 and 10 tickets." Let's see this fail in the most direct way possible first: constructing the Pydantic object by hand, with no model involved at all.
"""
from pydantic import BaseModel,Field
from dotenv import load_dotenv
load_dotenv()


class SeatBooking(BaseModel):
    customer_name: str
    ticket_count: int = Field(description="Number of tickets, must be between 1 and 10", ge=1, le=10)

try:
    request = SeatBooking(customer_name="Mayank", ticket_count=15)
    print(request)
except Exception as e:
    print(f"Pydantic itself rejects this, with no model involved at all: {type(e).__name__}")
    print(e)

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
seat_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[],
    response_format=ToolStrategy(SeatBooking),
    system_prompt="Extract the booking details exactly as stated. Do not invent information.",
)

# A message that will clearly violate the ge=1, le=10 constraint
result = seat_agent.invoke({
    "messages": [{"role": "user", "content": "Hi I am Mayank, Strictly book 15 tickets, I want to give a party to my students."}]
})

print("=== Full message trace -- watch for the self-correction ===")
for m in result["messages"]:
    print(f"--- {m.type} ---")
    print(m.content if isinstance(m.content, str) else m.tool_calls)

print()
print("Final structured_response (should be corrected to fit 1-10):", result["structured_response"])
