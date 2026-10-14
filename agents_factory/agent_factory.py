from langchain.agents import create_agent

from tools.weather import get_weather


def create_weather_agent():
    """
    Create and return a configured LangChain agent.
    """
    return create_agent(
        model="openai:gpt-5.5",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

def agent_edge_case():
    """
    Create and return a configured LangChain agent for edge case testing.
    """
    agent = create_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="""
    You are a helpful weather assistant.
    Use get_weather when the user asks about weather.
    If the city is missing, ask the user which city
    they want to check.
    Do not invent a city.
    """,
)