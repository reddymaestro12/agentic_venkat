from configs.config import load_config
from agents_factory.agent_factory import create_weather_agent

def print_messages(messages):
    for message in messages:
        print("Role:", message.type)
        print("Content:", message.content)
        print("-" * 40)

def main():
    load_config()    
    agent = create_weather_agent()
    result = agent.invoke(
    {"messages": [
        {"role": "user", "content": "What's the weather in San Francisco?"}
                ]
    }
)
    print_messages(result["messages"])


def multi_test_conditions():
    load_config()    
    agent = create_weather_agent()
    cities = [
        "What's the weather in Bengaluru?",
        "Tell me the weather in London.",
        "Will it be sunny in Tokyo?"
    ]

    for question in cities:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        print_messages(result["messages"])

def test_edge_case():
    load_config()    
    agent = create_weather_agent()
    result = agent.invoke(
        {"messages": [
            {"role": "user", "content": "What's the weather like?"}
        ]}
    )
    print_messages(result["messages"])

if __name__ == "__main__":
    #main()
    #multi_test_conditions()
    test_edge_case()