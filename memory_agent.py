from dotenv import load_dotenv
from pprint import pprint

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv(override=True)


# -------------------------------------------------------
# 1. Create Gemini model
# -------------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0
)


# -------------------------------------------------------
# 2. Agent without memory
# -------------------------------------------------------

agent_without_memory = create_agent(
    model=model
)

print("\n--- Agent without memory ---")

question = HumanMessage(
    content="Hello, my name is Seán and my favourite colour is green."
)

response = agent_without_memory.invoke(
    {"messages": [question]}
)

print("\nFirst response:")
print(response["messages"][-1].content)

question = HumanMessage(
    content="What is my favourite colour?"
)

response = agent_without_memory.invoke(
    {"messages": [question]}
)

print("\nSecond response:")
print(response["messages"][-1].content)


# -------------------------------------------------------
# 3. Agent with memory
# -------------------------------------------------------

agent_with_memory = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

print("\n--- Agent with memory ---")

question = HumanMessage(
    content="Hello, my name is Seán and my favourite colour is green."
)

response = agent_with_memory.invoke(
    {"messages": [question]},
    config
)

print("\nFirst response:")
print(response["messages"][-1].content)

question = HumanMessage(
    content="What is my favourite colour?"
)

response = agent_with_memory.invoke(
    {"messages": [question]},
    config
)

print("\nSecond response:")
print(response["messages"][-1].content)


# Optional: print full message history
print("\n--- Full memory response object ---")
pprint(response)