"""Retriever tool for answering T20 RULES questions from the indexed PDF."""
from pathlib import Path

from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore

from src.config import VECTORSTORE_PATH, get_embeddings

RULES_TOOL_NAME = "t20_rules_search"
RULES_TOOL_DESCRIPTION = (
    "Use this tool to answer questions about the OFFICIAL T20 CRICKET RULES and "
    "regulations: overs per innings, powerplay/fielding restrictions, super over, "
    "wides, no-balls, free hits, the DLS method, declarations, boundaries, "
    "dismissals, and other law/regulation questions. Do NOT use this tool for news, "
    "live scores, schedules or player updates. Input should be the user's rules "
    "question."
)


def load_rules_retriever_tool(k: int = 4, index_path: str = VECTORSTORE_PATH):
    """Build a LangChain retriever tool backed by the persisted vector index."""
    index_file = Path(index_path)
    if not index_file.exists():
        raise FileNotFoundError(
            f"Vector index not found at '{index_path}'. Run `python -m src.ingestion` "
            "first to index your T20 rules PDF."
        )

    store = InMemoryVectorStore.load(str(index_file), get_embeddings())
    retriever = store.as_retriever(search_kwargs={"k": k})

    return create_retriever_tool(
        retriever,
        name=RULES_TOOL_NAME,
        description=RULES_TOOL_DESCRIPTION,
    )
