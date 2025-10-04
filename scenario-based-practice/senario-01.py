# -----------------------------------------
# Import Required Libraries
# -----------------------------------------
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled,function_tool,ModelSettings
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

@function_tool
def get_weather(city: str) -> str:
    """Returns weather info for the specified city."""
    return f" {city} is Sunny!."

# -----------------------------------------
# Agent Definition
# -----------------------------------------
agent = Agent(
    name="Haiku Agent",
    instructions="You are a helpful assistant.",
    model=external_model,
    tools=[get_weather],     
    # model_settings=ModelSettings(
    #    tool_choice="none"  
    # )  
)

# -----------------------------------------
# Running the Agent
# -----------------------------------------

result = Runner.run_sync(
    agent,
    "hello?",  # user input
    max_turns=1
    
)

# Print final agent output
print(result.final_output)
