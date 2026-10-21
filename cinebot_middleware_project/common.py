from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
    LLMToolSelectorMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    LLMToolEmulator,
    ContextEditingMiddleware,
    ClearToolUsesEdit,
)
import re

def detect_booking_code(content: str) -> list[dict]:
    """Detect CineBot's own booking code format: BK followed by 4 digits."""
    matches = []
    for match in re.finditer(r"BK\d{4}", content):
        matches.append({"text": match.group(0), "start": match.start(), "end": match.end()})
    return matches


@tool
def check_showtimes(movie_title: str) -> str:
    """Check available showtimes for a movie at the cinema."""
    fake_showtimes = {
        "interstellar": "7:00 PM and 10:15 PM",
        "dune part two": "9:30 PM only",
        "oppenheimer": "Sold out for tonight",
    }
    return fake_showtimes.get(movie_title.lower(), "No showtimes found for that title.")

@tool
def book_seats(movie_title: str, seat_count: int) -> str:
    """Book seats for a movie. Irreversible once confirmed."""
    return f"Booked {seat_count} seat(s) for {movie_title}."

@tool
def cancel_booking(booking_id: str) -> str:
    """Cancel an existing booking. Irreversible."""
    return f"Booking {booking_id} cancelled."


@tool
def check_order_status(booking_id: str) -> str:
    """Check the status of an existing booking."""
    return f"Booking {booking_id}: confirmed, 2 seats, Interstellar, 7:00 PM."

@tool
def get_refund_policy() -> str:
    """Get the cinema's refund policy -- exact wording, not to be paraphrased."""
    return "Refunds available up to 2 hours before showtime. No refunds after that."

@tool
def lookup_seat_map(movie_title: str, seat_number: str) -> str:
    """Look up a specific seat -- fails if the seat number format is wrong."""
    if not seat_number or not seat_number[0].isalpha():
        raise ValueError(f"Malformed seat number '{seat_number}' -- expected a letter+number like 'A12'.")
    return f"Seat {seat_number} for {movie_title}: available."

cinebot_tools = [check_showtimes, book_seats, cancel_booking, check_order_status, get_refund_policy, lookup_seat_map]
def pretty_print_messages(result):
    for i, message in enumerate(result.get("messages", []), 1):
        print(f"\n{'=' * 80}")
        print(f"Message {i}: {message.__class__.__name__}")
        print("=" * 80)

        # Basic message information
        print(f"ID: {getattr(message, 'id', None)}")

        # Message content
        content = getattr(message, "content", "")
        if content:
            print("\nContent:")
            print(content)

        # Tool calls
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            print("\nTool Calls:")
            for tool in tool_calls:
                print(f"  • {tool['name']}")
                print(f"    Args: {tool['args']}")
                print(f"    ID:   {tool['id']}")

        # Tool message information
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            print(f"\nTool Call ID: {tool_call_id}")

        # Summarization information
        additional_kwargs = getattr(message, "additional_kwargs", {})
        if additional_kwargs.get("lc_source"):
            print(f"\nSource: {additional_kwargs['lc_source']}")

    print(f"\n{'=' * 80}")
    print("END OF MESSAGE HISTORY")
    print("=" * 80)
