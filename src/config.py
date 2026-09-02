"""Central configuration: env loading and LLM/embeddings factories.

LLM_PROVIDER selects the chat model used by the chatbot and quiz generator:
- "gemini"    -> ChatGoogleGenerativeAI
- "anthropic" -> ChatAnthropic

Embeddings always use Google's Generative AI embedding model. This keeps the app
free of heavy local ML dependencies (e.g. torch), so a GOOGLE_API_KEY is required
even when LLM_PROVIDER=anthropic.
"""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
PDF_PATH = os.getenv("PDF_PATH", "knowledgebase/T20_rules.pdf")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/t20_rules_index.json")


def get_llm(temperature: float = 0.3):
    """Return a LangChain chat model based on LLM_PROVIDER."""
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return ChatAnthropic(model=model, temperature=temperature)

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise ValueError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Set it to 'gemini' or 'anthropic' in .env."
    )


def get_embeddings():
    """Return the embeddings model used to build/query the vector store."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    return GoogleGenerativeAIEmbeddings(model=model)
