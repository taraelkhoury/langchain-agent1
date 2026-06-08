from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0
)

agent = create_agent(
    model=model
)

question = HumanMessage(content="What is the capital of the moon?")

response = agent.invoke(
    {"messages": [question]}
)

print(response["messages"][-1].content)