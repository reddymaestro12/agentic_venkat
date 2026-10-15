def get_weather(city):
    return f"the weather is sunny in {city}"

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
model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("what is the weather in bangalore")
# Print the complete AI message
response.pretty_print()

# Print only the response text
print("\nAnswer:")
print(response.content)

# Optional: inspect metadata
print("\nToken usage:")
print(response.usage_metadata)
