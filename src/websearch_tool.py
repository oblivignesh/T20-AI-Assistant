"""Web search tool for answering T20 NEWS questions."""
from langchain_community.tools import DuckDuckGoSearchRun

NEWS_TOOL_NAME = "t20_news_search"
NEWS_TOOL_DESCRIPTION = (
    "Use this tool to answer questions about recent T20 CRICKET NEWS: match results, "
    "live scores, schedules, squads, player transfers/injuries, tournaments such as "
    "the T20 World Cup or IPL, and any other time-sensitive information. Do NOT use "
    "this tool for questions about official playing rules/regulations. Input should "
    "be a concise search query."
)


def load_news_search_tool():
    """Build a free, no-API-key web search tool for T20 cricket news."""
    tool = DuckDuckGoSearchRun(name=NEWS_TOOL_NAME)
    tool.description = NEWS_TOOL_DESCRIPTION
    return tool
