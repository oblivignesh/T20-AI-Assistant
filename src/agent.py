"""LangChain tool-calling agent that decides between the rules retriever and web search.

Note: this uses `langchain_classic.agents` (the classic LCEL AgentExecutor loop)
rather than `langchain.agents.create_agent`, since the latter is built on top of
the LangGraph runtime and this project intentionally avoids LangGraph.
"""
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import get_llm
from src.prompts import load_chat_examples
from src.rag_tool import load_rules_retriever_tool
from src.websearch_tool import load_news_search_tool

SYSTEM_PROMPT = """You are "T20 Buddy", an expert assistant for T20 cricket.

You have access to two tools and MUST pick the right one for every question:
1. `t20_rules_search` - for questions about official T20 RULES / regulations
   (overs, powerplay, super over, DLS, no-balls, wides, declarations, etc.).
   This runs a semantic search over an indexed PDF rulebook.
2. `t20_news_search` - for questions about T20 CRICKET NEWS (recent match
   results, scores, schedules, player news, tournaments). This performs a
   live web search.

Rules for choosing a tool:
- If the question is about what is/isn't allowed, or how the game is
  officially played -> use `t20_rules_search`.
- If the question is about something recent, time-sensitive, or asks things
  like "who won", "when is", "latest" -> use `t20_news_search`.
- If a question mixes both, call both tools and combine the results.
- Never fabricate an answer; only answer using information returned by your
  tools. If the tools don't have the answer, say so honestly.
- Answer concisely and directly.

Example Q&A pairs illustrating the expected answer style:
{examples}
"""


def build_chat_agent() -> AgentExecutor:
    """Construct the chatbot agent with the rules retriever and web search tools."""
    llm = get_llm()
    tools = [load_rules_retriever_tool(), load_news_search_tool()]

    system_message = SystemMessage(
        content=SYSTEM_PROMPT.format(examples=load_chat_examples())
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def extract_text(output) -> str:
    """Normalize an AgentExecutor `output` value into a plain string.

    Some chat models (e.g. newer Gemini models) return message content as a
    list of content blocks (e.g. `[{"type": "text", "text": "..."}]`) instead
    of a plain string. This flattens either shape into plain text.
    """
    if isinstance(output, str):
        return output

    if isinstance(output, list):
        parts = []
        for block in output:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    return str(output)
