"""
The Problem: What If a Message Could Be More Than One Kind of Thing?
BookingRequest assumes we already know it's a booking. Real support systems see many kinds of requests — what if a customer might have ten different intents: book, cancel, modify, shift, check, and more? Forcing everything through one schema stops making sense.

1.7 The Fix: Union — Letting the Model Choose Which Schema Fits
Analogy: two different intake forms at a service counter — a "New Request" form and a "Cancellation" form. The customer picks whichever form actually matches their situation.
"""

from langchain_core.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()
assert os.environ.get("OPENAI_API_KEY")

from typing import Union
from pydantic import BaseModel, Field
from typing import Literal

class NewBooking(BaseModel):
    """A request to book NEW tickets."""
    customer_name: str
    movie_title: str
    ticket_count: int

class CancelBooking(BaseModel):
    """A request to CANCEL an existing booking."""
    customer_name: str
    movie_title: str


from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
union_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[],
    response_format=ToolStrategy(Union[NewBooking, CancelBooking]),
)

print("Agent built -- it can now return EITHER schema shape, chosen by the model itself.")

result1 = union_agent.invoke({
    "messages": [{"role": "user", "content": "I want to cancel my movie Oppenheimer, I'm Aisha"}]
})
print("Result 1:", result1["structured_response"])
print("Type:", type(result1["structured_response"]).__name__)

result2 = union_agent.invoke({
    "messages": [{"role": "user", "content": "Book one ticket for Oppenheimer for Mayank"}]
})
print()
print("Result 2:", result2["structured_response"])
print("Type:", type(result2["structured_response"]).__name__)

if isinstance(result2["structured_response"], NewBooking):
    print("We got a new booking! Ticket count:", result2["structured_response"].ticket_count)
elif isinstance(result2["structured_response"], CancelBooking):
    print("We got a cancellation request.")