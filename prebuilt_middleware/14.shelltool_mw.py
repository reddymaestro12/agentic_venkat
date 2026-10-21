from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()


@tool
def create_folder(folder_name: str) -> str:
    """
    Create a folder under /agenticai.
    """
    base_path = Path("/agenticai")
    folder_path = base_path / folder_name

    folder_path.mkdir(parents=True, exist_ok=True)

    return f"Folder created successfully: {folder_path}"


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[create_folder],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Create a folder with name custom_middleware"
            }
        ]
    }
)

print(result["messages"][-1].content)