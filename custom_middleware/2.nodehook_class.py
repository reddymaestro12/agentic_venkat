from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

class MessageLimitMiddleware(AgentMiddleware):
    def __init__(self, max_messages: int = 50):
        super().__init__()
        self.max_messages = max_messages

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if len(state["messages"]) >= self.max_messages:
            return {
                "messages": [AIMessage("Conversation limit reached.")],
                "jump_to": "end"
            }
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None

#create the middleware instance and pass it to middleware section while creating the agent
ml = MessageLimitMiddleware(max_messages=100)

#create agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    middleware=[ml],
    system_prompt="You are a helpful assistant.",
)

# Invoke the agent
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain middleware in LangChain with an example.",
            }
        ]
    }
)

# Display the final response
print("\nFinal response:")
print(result["messages"][-1].content)
