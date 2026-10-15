from langchain.messages import HumanMessage, AIMessage, SystemMessage

#dictionary format messages
conversation1 = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

#message objects 
conversation2 = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

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

# Invoke the model for both conversations
response1 = model.invoke(conversation1)
response2 = model.invoke(conversation2)

# Collect responses into a list
results = [response1, response2]

# Process each response
for i, response in enumerate(results, start=1):
    print(f"\n{'=' * 20} Response {i} {'=' * 20}")

    # Print the complete AI message
    response.pretty_print()

    # Print only the response text
    print("\nAnswer:")
    print(response.content)

    # Print token usage metadata
    print("\nToken usage:")
    print(response.usage_metadata)
