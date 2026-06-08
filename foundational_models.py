from dotenv import load_dotenv
from pprint import pprint
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(override=True)


# -------------------------------------------------------
# 1. Initialising and invoking a model
# -------------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)

response = model.invoke("What's the capital of the Moon?")

print("\n--- Basic model response ---")
print(response.content)

print("\n--- Response metadata ---")
pprint(response.response_metadata)


# -------------------------------------------------------
# 2. Customising the model
# -------------------------------------------------------

custom_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0
)

response = custom_model.invoke("What's the capital of the Moon?")

print("\n--- Custom model response ---")
print(response.content)


# -------------------------------------------------------
# 3. Initialising and invoking an agent
# -------------------------------------------------------

agent = create_agent(
    model=custom_model
)

response = agent.invoke(
    {"messages": [HumanMessage(content="What's the capital of the Moon?")]}
)

print("\n--- Basic agent response ---")
print(response["messages"][-1].content)


# -------------------------------------------------------
# 4. Agent with conversation history
# -------------------------------------------------------

response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What's the capital of the Moon?"),
            AIMessage(content="The capital of the Moon is Luna City."),
            HumanMessage(content="Interesting, tell me more about Luna City."),
        ]
    }
)

print("\n--- Agent with conversation history ---")
print(response["messages"][-1].content)


# -------------------------------------------------------
# 5. Basic prompting with a system prompt
# -------------------------------------------------------

question = HumanMessage(content="What's the capital of the Moon?")

system_prompt = """
You are a science fiction writer.
Create a fictional capital city at the user's request.
"""

scifi_agent = create_agent(
    model=custom_model,
    system_prompt=system_prompt
)

response = scifi_agent.invoke(
    {"messages": [question]}
)

print("\n--- Science fiction agent response ---")
print(response["messages"][-1].content)


# -------------------------------------------------------
# 6. Few-shot examples
# -------------------------------------------------------

few_shot_prompt = """
You are a science fiction writer.
Create a space capital city at the user's request.

User: What is the capital of Mars?
Scifi Writer: Marsialis

User: What is the capital of Venus?
Scifi Writer: Venusovia
"""

few_shot_agent = create_agent(
    model=custom_model,
    system_prompt=few_shot_prompt
)

response = few_shot_agent.invoke(
    {"messages": [question]}
)

print("\n--- Few-shot agent response ---")
print(response["messages"][-1].content)


# -------------------------------------------------------
# 7. Structured prompt
# -------------------------------------------------------

structured_prompt = """
You are a science fiction writer.
Create a space capital city at the user's request.

Please keep to the below structure:

Name: The name of the capital city
Location: Where it is based
Vibe: 2-3 words to describe its vibe
Economy: Main industries
"""

structured_agent = create_agent(
    model=custom_model,
    system_prompt=structured_prompt
)

response = structured_agent.invoke(
    {"messages": [question]}
)

print("\n--- Structured prompt response ---")
print(response["messages"][-1].content)


# -------------------------------------------------------
# 8. Structured output
# -------------------------------------------------------

class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str


structured_output_agent = create_agent(
    model=custom_model,
    system_prompt="You are a science fiction writer. Create a capital city at the user's request.",
    response_format=CapitalInfo
)

response = structured_output_agent.invoke(
    {"messages": [HumanMessage(content="What is the capital of the Moon?")]}
)

capital_info = response["structured_response"]

print("\n--- Structured output response ---")
print(capital_info)

print("\n--- Final sentence ---")
print(f"{capital_info.name} is a city located at {capital_info.location}.")