
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()


# 1. Define all tools at startup
@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"The weather in {city} is sunny."


@tool
def calculator(expression: str) -> str:
    """Perform basic arithmetic calculations."""
    # Demo only: replace with a safe arithmetic parser in production.
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        raise ValueError("Unsupported expression")

    return str(evaluate(ast.parse(expression, mode="eval").body))


@tool
def search_documents(query: str) -> str:
    """Search internal company documents."""
    return f"Found internal documents matching: {query}"


# 2. Store the tools in a registry
ALL_TOOLS = {
    "get_weather": get_weather,
    "calculator": calculator,
    "search_documents": search_documents,
}


# 3. Define permissions for different users
USER_PERMISSIONS = {
    "basic": {"get_weather", "calculator"},
    "admin": {"get_weather", "calculator", "search_documents"},
}


# 4. Create an agent dynamically for the current permissions
def create_user_agent(user_role: str):
    allowed_names = USER_PERMISSIONS.get(user_role, set())

    allowed_tools = [
        tool
        for name, tool in ALL_TOOLS.items()
        if name in allowed_names
    ]

    return create_agent(
        model="openai:gpt-5-mini",
        tools=allowed_tools,
        system_prompt=(
            "You are a helpful assistant. "
            "Use available tools when needed."
        ),
    )


# 5. Invoke the agent for a basic user
agent = create_user_agent("basic")

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the weather in Bengaluru?"
        }
    ]
})

for message in result["messages"]:
    message.pretty_print()
