# T20 AI Assistant

An AI assistant for T20 cricket with two features:

1. **Chatbot** - ask anything about T20 cricket. A LangChain agent decides
   whether to answer from the indexed rulebook (semantic search) or from a
   live web search, and calls the right tool automatically.
2. **Take Quiz** - generates a multiple-choice quiz (you choose how many
   questions) grounded in the indexed rulebook.

Built with **LangChain only** (no LangGraph) and **Streamlit** for the UI.

## How the agent decides: vector DB vs. web search

`src/agent.py` builds a classic LangChain tool-calling agent
(`langchain_classic.agents.create_tool_calling_agent` + `AgentExecutor`) with
two tools:

- `t20_rules_search` ([src/rag_tool.py](src/rag_tool.py)) - a retriever tool
  (`create_retriever_tool`) over an `InMemoryVectorStore` built from the
  indexed PDF rulebook. Used for RULES questions (overs, powerplay, super
  over, DLS, no-balls, declarations, etc.).
- `t20_news_search` ([src/websearch_tool.py](src/websearch_tool.py)) - a
  DuckDuckGo web search tool. Used for NEWS questions (results, scores,
  schedules, tournaments, player news).

The system prompt instructs the LLM on which tool to pick per question (and
to call both if a question mixes rules and news). The example Q&A pairs in
[knowledgebase/chat_prompts.md](knowledgebase/chat_prompts.md) are injected
into the system prompt to steer the answer style.

## Quiz generation

`src/quiz.py` pulls a diverse sample of chunks from the same vector store,
then prompts the LLM (with `knowledgebase/quiz_prompts.md` as a style
reference and a `PydanticOutputParser` for structured output) to generate N
unique multiple-choice questions with 4 options each.

The number-of-questions dropdown is hardcoded to `[3, 5, 10, 15, 20]`
(`QUIZ_LENGTH_OPTIONS` in [src/quiz.py](src/quiz.py)). 20 is the max, chosen
so the LLM has enough distinct, grounded facts from the rulebook to avoid
repeating or inventing questions.

## Project structure

```
app.py                        # Streamlit UI (Chatbot + Take Quiz pages)
src/
  config.py                   # env config, LLM + embeddings factories
  prompts.py                  # loads the two knowledgebase/*.md example files
  ingestion.py                # PDF -> InMemoryVectorStore indexer (CLI)
  rag_tool.py                 # rules retriever tool
  websearch_tool.py           # news web-search tool
  agent.py                    # chatbot agent (tools + decision prompt)
  quiz.py                     # quiz generation chain
knowledgebase/
  T20_rules.pdf                # source rulebook PDF, indexed by src/ingestion.py
  chat_prompts.md              # few-shot examples for chatbot answer style
  quiz_prompts.md               # few-shot examples for quiz question format
vectorstore/                  # generated vector index (gitignored)
```

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set GOOGLE_API_KEY (always required, used for embeddings)
# and, if LLM_PROVIDER=anthropic, also set ANTHROPIC_API_KEY
```

> Embeddings always use Google's `text-embedding-004` model to avoid heavy
> local ML dependencies (e.g. torch). This means `GOOGLE_API_KEY` is required
> even if you set `LLM_PROVIDER=anthropic` to use Claude as the chat model.

Build the vector index from `knowledgebase/T20_rules.pdf` (or your own PDF
via `--pdf`):

```bash
python -m src.ingestion
```

Run the app:

```bash
streamlit run app.py
```

## Choosing the LLM

Set in `.env`:

```
LLM_PROVIDER=gemini      # or: anthropic
```

## Notes

- Re-run `python -m src.ingestion` whenever you replace the source PDF.
- The vector index is persisted to `vectorstore/t20_rules_index.json` and
  reloaded on each app/tool invocation; delete it to force a rebuild.
