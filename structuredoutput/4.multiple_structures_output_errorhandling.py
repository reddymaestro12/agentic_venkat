
from pydantic import BaseModel, Field
from typing import Union

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from dotenv import load_dotenv
load_dotenv()

class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")


class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")


agent = create_agent(
    model="gpt-5.5",
    tools=[],
    response_format=ToolStrategy(
        Union[ContactInfo, EventDetails]
    )
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "Extract info: John Doe (john@email.com) is "
                "organizing Tech Conference on March 15th"
            )
        }
    ]
})

# Print the complete agent execution trace
for message in result["messages"]:
    message.pretty_print()

# Print the final structured response separately
print("\nFinal structured response:")
print(result["structured_response"])
