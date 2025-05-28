import os
import phi
import phi.api
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
from agno.playground import Playground,serve_playground_app
load_dotenv()
phi.api=os.getenv("PHI_API_KEY")

Web_Search_agent=Agent(
    name="Web_Search_agent",
    role="Search the web for information",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    # show_tools_calls=True,
    # markdown=True,

)

Financial_agent=Agent(
    name="Financial_agent",
    # role="Search the web for information",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=["Use tables to display the data"],
    # show_tools_calls=True,
    # markdown=True,

)

app=Playground(agents=[Web_Search_agent,Financial_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)