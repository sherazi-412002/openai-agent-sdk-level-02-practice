# -----------------------------------------
# Import Required Libraries
# -----------------------------------------
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled,function_tool,ModelSettings
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from agents.tool import FunctionTool


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



@function_tool(name_override= "check_budget",description_override="Checks if a budget is sufficient for a given project type. Takes 'budget' (int) and 'project_type' (str) as input.")
def check_budget(self, budget: int, project_type: str) -> str:
    if project_type == "short_term":
        if budget >= 10000:
            return f"Budget of ${budget} is sufficient for a short-term project. The condition is met."
        else:
            return f"Budget of ${budget} is insufficient for a short-term project. The condition is NOT met."
    elif project_type == "long_term":
        if budget >= 50000:
            return f"Budget of ${budget} is sufficient for a long-term project. The condition is met."
        else:
            return f"Budget of ${budget} is insufficient for a long-term project. The condition is NOT met."
    else:
        return f"Unknown project type: {project_type}."

    
@function_tool(name_override="check_company_type",description_override="Checks if a company is qualified for a project type. Takes 'company_type' (str) and 'project_type' (str) as input.")
def check_company_type(self, company_type: str, project_type: str) -> str:
    if company_type == "basic":
        if project_type == "short_term":
            return "Basic company is eligible for short-term projects. The condition is met."
        else:
            return "Basic company is NOT eligible for long-term projects. The condition is NOT met."
    elif company_type == "vip":
        return "VIP company is eligible for all types of projects. The condition is met."
    else:
        return f"Unknown company type: {company_type}."


# -----------------------------------------
# Agent Definition
# -----------------------------------------
agent = Agent(
    name="EligibilityChecker",
    instructions=(
        "You are an expert project eligibility checker. Your job is to evaluate if a company "
        "is suitable for a project based on dynamic conditions provided in a JSON format. "
        "Use the 'check_budget' and 'check_company_type' tools to verify all conditions. "
        "Once you have a final verdict, summarize the findings and state clearly if the company is eligible or not."
    ),
    model=external_model,
    tools=[check_company_type, check_company_type]
)
# -----------------------------------------
# Running the Agent
# -----------------------------------------

result = Runner.run_sync(
    agent,
    "A feroze company witha a status of 'basic' and budget of '$60,000' wants to proceed with long term project?",  # user input
)

# Print final agent output
print(result.final_output)