from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
load_dotenv()


Web_Search_agent=Agent(
    name="Web_Search_agent",
    role="Search the web for information",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,

)

Financial_agent=Agent(
    name="Financial_agent",
    # role="Search the web for information",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=["Use tables to display the data"],
    show_tools_calls=True,
    markdown=True,

)

multi_ai_agent=Agent(
    team=[Web_Search_agent,Financial_agent],
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    instructions=["Always include sources","Use tables to display the data"],
    show_tools_calls=True,
    markdown=True,

)

multi_ai_agent.print_response("Summarize analyst recommendations and share latest news for NVIDIA",stream=True)