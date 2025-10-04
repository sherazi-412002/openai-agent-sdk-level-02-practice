# -----------------------------------------
# Import Required Libraries
# -----------------------------------------
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled,ModelSettings
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio
from openai.types.shared import Reasoning 


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


async def main():
    agent = Agent(
        name="Knowledgable GPT-5 Assistant",
        instructions="You're a knowledgable assistant. You always provide an interesting answer.",
        model=external_model,
        
        #****
        # The reasoning: Reasoning field is exclusive to OpenAI’s o1-series models
        # No other LLM provider (Anthropic, Google, Meta, Mistral, etc.) supports this exact parameter

        # model_settings=ModelSettings(
        #     reasoning=Reasoning(effort="minimal"),  # "minimal", "low", "medium", "high"
        #     # verbosity="low",  # "low", "medium", "high"
        # ),
    )
    result = await Runner.run(agent, "Tell me something about recursion in programming.")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())