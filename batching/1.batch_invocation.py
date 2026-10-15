from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model(
    "gpt-5.5",
    # Kwargs passed to the model:
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
    max_retries=6,  # Default; increase for unreliable networks
)

responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    # Print the complete AI message
    response.pretty_print()

    # Print only the response text
    print("\nAnswer:")
    print(response.content)

    # Optional: inspect metadata
    print("\nToken usage:")
    print(response.usage_metadata)
