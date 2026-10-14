
cities = [
    "What's the weather in Bengaluru?",
    "Tell me the weather in London.",
    "Will it be sunny in Tokyo?"
]

for question in cities:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    for message in result["messages"]:
        print("Role:", message.type)
        print("Content:", message.content)
        print("-" * 40)