from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[
        # Remove email completely
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),

        # Mask credit card
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),

        # Block IP addresses
        PIIMiddleware(
            "ip",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

# result = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": """
#                 Email: john@gmail.com
#                 Card: 4111111111111111
#                 IP: 192.168.1.10
#                 """
#             }
#         ]
#     }
# )

try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "My email is john@gmail.com"
                }
            ]
        }
    )

    print(result["messages"][-1].content)

except Exception as e:
    print(f"PII validation failed: {e}")