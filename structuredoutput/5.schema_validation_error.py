
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

load_dotenv()


class ProductRating(BaseModel):
    rating: int | None = Field(
        description="Rating from 1-5",
        ge=1,
        le=5
    )
    comment: str = Field(
        description="Review comment"
    )


agent = create_agent(
    model="gpt-5.5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors=True
    ),
    system_prompt=(
        "You are a helpful assistant that parses product reviews. "
        "Do not make any field or value up. "
        "The rating must be an integer between 1 and 5. "
        "If the review gives a rating outside this range, "
        "map it to the nearest valid rating. "
        "For example, 10/10 maps to 5/5. "
        "Do not invent a rating if the review contains no rating."
    )
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Parse this: Amazing product, 10/10!"
        }
    ]
})

# Print the complete execution trace
for message in result["messages"]:
    message.pretty_print()

# Print the final validated Pydantic object
print("\nFinal structured response:")
print(result["structured_response"])

# Optional: print as JSON
print("\nFinal JSON:")
print(result["structured_response"].model_dump_json(indent=2))
