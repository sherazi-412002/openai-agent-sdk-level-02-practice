# -----------------------------------------
# Import Required Libraries
# -----------------------------------------
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv


# -----------------------------------------
# Load environment variables
# -----------------------------------------
set_tracing_disabled(True)
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")  # Fetch Gemini API key from .env file

# -----------------------------------------
# Configure external OpenAI-compatible client for Gemini
# -----------------------------------------
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini OpenAI-compatible endpoint
)

# Define which model we’re going to use
external_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",   # Gemini flash model
    openai_client=external_client
)



# -----------------------------------------
# Agent Definition
# -----------------------------------------
agent = Agent(
    name="Haiku Agent",
    instructions="You are a helpful assistant.",
)

# -----------------------------------------
# Running the Agent
# -----------------------------------------

result = Runner.run_sync(
    agent,
    "Hi?",  # user input
)

# Print final agent output
print(result.final_output)