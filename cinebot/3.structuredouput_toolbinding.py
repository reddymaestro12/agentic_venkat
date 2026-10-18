"""
Problem : tool calling will never work with_structured_output on the bind_tool 
with the base model
so the fix is create_aggent which is explained in the next program
i.e the structured output at the agent level
"""
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

from langchain.chat_models import init_chat_model
model = init_chat_model("openai:gpt-5-mini")
print("Environment ready. CineBot's brain (the model) is connected.")

#define the structured model
incomplete_model = model.bind_tools([peek_showtimes]).with_structured_output(BookingRequest)
try:
    result = incomplete_model.invoke("Is Interstellar showing tonight? Book 2 seats for Rohan")
    print("Result:", result)
    print()
    print("Even if this ran, there's no way here to see whether the tool was actually called,")
    print("or to continue the conversation with a tool's result -- with_structured_output()")
    print("returns ONLY the parsed object, nothing else.")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
    print()
    print("Either way, the underlying issue is the same: with_structured_output() was never")
    print("designed to ALSO manage a tool-calling loop.")
"""
output : chema defined -- this is the 'door policy' the model will now have to follow.
Environment ready. CineBot's brain (the model) is connected.
Result: customer_name='Rohan' movie_title='Interstellar' action='book' ticket_count=2

Even if this ran, there's no way here to see whether the tool was actually called,
or to continue the conversation with a tool's result -- with_structured_output()
returns ONLY the parsed object, nothing else.
"""