from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.agents.middleware import AgentState, AgentMiddleware
from typing_extensions import NotRequired
from typing import Any
from dotenv import load_dotenv
load_dotenv()

class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]


class CallCounterMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)
        if count > 10:
            return {"jump_to": "end"}
        return None

    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        return {"model_call_count": state.get("model_call_count", 0) + 1}


agent = create_agent(
    model="gpt-5.5",
    middleware=[CallCounterMiddleware()],
    tools=[],
)

# Invoke with custom state
result = agent.invoke({
    "messages": [HumanMessage("Hello")],
    "model_call_count": 0,
    "user_id": "user-123",
})