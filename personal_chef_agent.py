from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from scrapfly import ScrapflyClient, ScrapeConfig
import os


load_dotenv(override=True)


# -------------------------------------------------------
# 1. Scrapfly web scraping tool
# -------------------------------------------------------

SCRAPFLY_API_KEY = os.getenv("SCRAPFLY_API_KEY")

if not SCRAPFLY_API_KEY:
    raise ValueError("SCRAPFLY_API_KEY is missing. Add it to your .env file.")

scrapfly_client = ScrapflyClient(key=SCRAPFLY_API_KEY)


@tool
def web_scrape(url: str) -> str:
    """Scrape a webpage and return its content."""
    try:
        result = scrapfly_client.scrape(
            ScrapeConfig(
                url=url,
                render_js=True,
                asp=True,
                format="markdown"
            )
        )
        return result.scrape_result["content"][:3000]
    except Exception as e:
        return f"Scrapfly scraping failed: {type(e).__name__}: {e}"


# -------------------------------------------------------
# 2. System prompt
# -------------------------------------------------------

system_prompt = """
You are a personal chef.

The user will give you ingredients they have at home.

If the user provides a recipe webpage URL, use the web_scrape tool to read the page and suggest recipes based on it.

If the user does not provide a URL, suggest recipe ideas based on your own knowledge.

Return clear recipe suggestions and, if requested, give step-by-step recipe instructions.
"""


# -------------------------------------------------------
# 3. Create Gemini model
# -------------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0
)


# -------------------------------------------------------
# 4. Create agent with memory
# -------------------------------------------------------

agent = create_agent(
    model=model,
    tools=[web_scrape],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)


# -------------------------------------------------------
# 5. Test conversation
# -------------------------------------------------------

config = {
    "configurable": {
        "thread_id": "personal-chef-thread"
    }
}

response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="I have some leftover chicken and rice. What can I make?"
            )
        ]
    },
    config
)

print("\n--- Personal chef response ---")
print(response["messages"][-1].content)


response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Give me one simple recipe with steps."
            )
        ]
    },
    config
)

print("\n--- Follow-up response with memory ---")
print(response["messages"][-1].content)