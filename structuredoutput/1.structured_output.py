from pydantic import BaseModel, Field
from langchain.agents import create_agent
from dotenv import load_dotenv
import logging
import time
load_dotenv()

class ContactInfo(BaseModel):
    """contact information for a person"""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

def main():
    agent = create_agent(model="openai:gpt-4o-mini",response_format=ContactInfo)
    result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
    })
    print(result["structured_response"])

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()