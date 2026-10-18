#CineBot try to understand booking requests using nothing but plain text extraction.
import os
from dotenv import load_dotenv
load_dotenv()
assert os.environ.get("OPENAI_API_KEY")

from pydantic import BaseModel, Field
from typing import Literal

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
structered_model = model.with_structured_output(BookingRequest)

booking_requests = [
    "Hi, I'd like 2 tickets for Interstellar at the 7pm show tonight, name is Priya.",
    "can u book me a seat for the 9:30 showing of dune part two? im rohan",
    "URGENT - need to CANCEL my booking for Oppenheimer, confirmation was under Aisha",
]

for msg in booking_requests:
    result = structered_model.invoke(f"extract a booking request from: {msg}")
    print(result)