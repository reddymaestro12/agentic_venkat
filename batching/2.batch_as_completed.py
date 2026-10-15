
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "gpt-5.5",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
    max_retries=6,
)

questions = [
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]

# Process batches and receive each result as it completes
for index, response in model.batch_as_completed(questions):
    print(f"\n{'=' * 20} Batch Input {index + 1} {'=' * 20}")

    # Print the complete LangChain AI message
    response.pretty_print()

    # Print only the generated answer
    print("\nAnswer:")
    print(response.content)

    # Print token usage if available
    print("\nToken usage:")
    print(response.usage_metadata)
