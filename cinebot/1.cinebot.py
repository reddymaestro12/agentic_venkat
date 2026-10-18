#CineBot try to understand booking requests using nothing but plain text extraction.
import os
from dotenv import load_dotenv

def load_config():
    """Load environment variables from .env file."""
    return load_dotenv()

assert os.environ.get("OPENAI_API_KEY")

from langchain.chat_models import init_chat_model
model = init_chat_model("openai:gpt-5-mini")
print("Environment ready. CineBot's brain (the model) is connected.")

booking_requests = [
    "Hi, I'd like 2 tickets for Interstellar at the 7pm show tonight, name is Priya.",
    "can u book me a seat for the 9:30 showing of dune part two? im rohan",
    "URGENT - need to CANCEL my booking for Oppenheimer, confirmation was under Aisha",
]

for msg in booking_requests:
    result = model.invoke(f"Extract the customer's name, movie, and what they want (book or cancel) from: {msg}")
    print(result.content)

"""
Look closely at the three answers. They might all sound reasonable individually — but they are not the same shape. If you tried to write one line of code that reliably pulls out just the movie name from all three answers, you couldn't — because the model decided the format fresh each time, not you.

The problem, stated plainly: free text isn't wrong, it's just not a contract. The model decides the shape of its answer fresh every single time — fine for a human reading it, and genuinely broken for code that needs to reliably extract a specific piece of data


"""