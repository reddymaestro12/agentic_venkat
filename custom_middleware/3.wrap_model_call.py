
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

load_dotenv()

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")

class RetryMiddleware(AgentMiddleware):
    def __init__(
        self,
        max_retries: int = 3,
    ):
        super().__init__()

        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        self.max_retries = max_retries

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:

        for attempt in range(1, self.max_retries + 1):
            try:
                print(
                    f"Model call attempt "
                    f"{attempt}/{self.max_retries}"
                )

                response = handler(request)

                print("Model call succeeded.")
                return response

            except Exception as exc:
                print(
                    f"Model call failed on attempt "
                    f"{attempt}/{self.max_retries}: "
                    f"{type(exc).__name__}: {exc}"
                )

                if attempt == self.max_retries:
                    print("Maximum retries reached.")
                    raise

        raise RuntimeError("Unexpected retry loop termination.")


# Create the middleware
retry_middleware = RetryMiddleware(max_retries=3)


# Create the agent
agent_class = create_agent(
    model="gpt-4o-mini",
    tools=[],
    middleware=[retry_middleware],
    system_prompt="You are a helpful assistant.",
)

# Invoke the agent
try:
    result = agent_class.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain retries in LangChain middleware.",
                }
            ]
        }
    )

    print("\nFinal answer:")
    print(result["messages"][-1].content)

except Exception as exc:
    print(
        f"\nAgent execution failed: "
        f"{type(exc).__name__}: {exc}"
    )

agent_decorator= create_agent(
    model="gpt-4o-mini",
    tools=[],
    middleware=[retry_middleware],
    system_prompt="You are a helpful assistant.",
)

#Invoke the agent
try:
    result1 = agent_decorator.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain retries in LangChain middleware.",
                }
            ]
        }
    )

    print("\nFinal answer:")
    print(result1["messages"][-1].content)

except Exception as exc:
    print(
        f"\nAgent execution failed: "
        f"{type(exc).__name__}: {exc}"
    )
