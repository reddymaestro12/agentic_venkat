from langchain_core.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()
assert os.environ.get("OPENAI_API_KEY")

from pydantic import BaseModel, Field
from typing import Literal

@tool
def peek_showtimes(movie_title: str) -> str:
    """Check showtimes for a movie."""
    return "7:00 PM and 10:15 PM"

class BookingRequest(BaseModel):
    customer_name: str = Field(description="The customer's name")
    movie_title: str = Field(description="The movie they want to see")
    action: Literal["book", "cancel"] = Field(description="Whether this is a new booking or a cancellation")
    ticket_count: int = Field(description="How many tickets, default 1 if not mentioned", default=1)

print("Schema defined -- this is the 'door policy' the model will now have to follow.")

from langchain.agents import create_agent
booking_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[peek_showtimes],
    response_format=BookingRequest,
)

result = booking_agent.invoke({
    "messages": [{"role": "user", "content": "can u book me a seat for the 9:30 showing of dune part two? im rohan"}]
})
print(result['structured_response'])