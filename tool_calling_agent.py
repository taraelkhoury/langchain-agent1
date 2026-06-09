from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from scrapfly import ScrapflyClient, ScrapeConfig
import os


load_dotenv(override=True)


# -------------------------------------------------------
# 1. Create Gemini model
# -------------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0
)


# -------------------------------------------------------
# 2. Define a simple arithmetic tool
# -------------------------------------------------------

@tool("square_root", description="Calculate the square root of a number")
def square_root_tool(x: float) -> float:
    return x ** 0.5


# Test the tool alone
print("\n--- Square root tool test ---")
print(square_root_tool.invoke({"x": 467}))


# -------------------------------------------------------
# 3. Add the tool to an agent
# -------------------------------------------------------

math_agent = create_agent(
    model=model,
    tools=[square_root_tool],
    system_prompt="You are an arithmetic wizard. Use your tools to calculate the square root of any number."
)

question = HumanMessage(content="What is the square root of 467?")

response = math_agent.invoke(
    {"messages": [question]}
)

print("\n--- Math agent response ---")
print(response["messages"][-1].content)


# -------------------------------------------------------
# 4. Agent without web search
# -------------------------------------------------------

basic_agent = create_agent(
    model=model
)

question = HumanMessage(content="How up to date is your training knowledge?")

response = basic_agent.invoke(
    {"messages": [question]}
)

print("\n--- Agent without web search response ---")
print(response["messages"][-1].content)


# -------------------------------------------------------
# 5. Scrapfly web scrape tool
# -------------------------------------------------------

SCRAPFLY_API_KEY = os.getenv("SCRAPFLY_API_KEY")

if not SCRAPFLY_API_KEY:
    print("\nSCRAPFLY_API_KEY is missing. Web scrape tool will not run.")
else:
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

    print("\n--- Web scrape tool test ---")
    result = web_scrape.invoke("https://www.sf.gov/departments--office-mayor")
    print(result[:1000])

    web_agent = create_agent(
        model=model,
        tools=[web_scrape],
        system_prompt="You are a helpful assistant. Use the web_scrape tool when you need information from a webpage."
    )

    question = HumanMessage(
        content="Use this webpage to answer who the current mayor of San Francisco is: https://www.sf.gov/departments--office-mayor"
    )

    response = web_agent.invoke(
        {"messages": [question]}
    )

    print("\n--- Web agent response ---")
    print(response["messages"][-1].content)