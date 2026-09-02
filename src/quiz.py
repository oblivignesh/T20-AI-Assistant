"""Quiz generation chain: produces multiple-choice T20 rules questions."""
import random
from pathlib import Path
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

from src.config import VECTORSTORE_PATH, get_embeddings, get_llm
from src.prompts import load_quiz_examples

# Hardcoded dropdown choices for "number of questions". 20 is chosen as the max
# because the demo rulebook only has enough distinct facts to reliably support
# ~20 unique, non-repetitive questions before the LLM would start repeating itself.
QUIZ_LENGTH_OPTIONS = [3, 5, 10, 15, 20]
MAX_QUIZ_QUESTIONS = max(QUIZ_LENGTH_OPTIONS)

# Broad topic queries used to pull a diverse sample of chunks from the vector
# store so generated questions aren't all about the same rule.
_TOPIC_QUERIES = [
    "overs powerplay fielding restrictions",
    "super over tie result",
    "no ball wide free hit extras",
    "declaration innings result",
    "DLS method rain interruption",
    "boundary six four dismissal run out",
]


class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question text")
    options: List[str] = Field(description="Exactly 4 possible answers, in any order")
    answer: str = Field(description="The correct option, copied verbatim from options")


class QuizSet(BaseModel):
    questions: List[QuizQuestion]


QUIZ_SYSTEM_PROMPT = """You are a T20 cricket quiz master. You write clear, \
unambiguous multiple-choice questions about official T20 cricket rules, based \
ONLY on the provided context - never invent facts that aren't supported by it. \
Every question must have exactly 4 options with exactly one correct answer, and \
the "answer" field must match one of the options exactly (verbatim). Do not \
repeat the same question twice."""


def _load_context_chunks(index_path: str = VECTORSTORE_PATH, sample_size: int = 12) -> str:
    index_file = Path(index_path)
    if not index_file.exists():
        raise FileNotFoundError(
            f"Vector index not found at '{index_path}'. Run `python -m src.ingestion` "
            "first to index your T20 rules PDF."
        )

    store = InMemoryVectorStore.load(str(index_file), get_embeddings())

    seen = {}
    for query in _TOPIC_QUERIES:
        for doc in store.similarity_search(query, k=4):
            seen[doc.page_content] = doc

    chunks = list(seen.values())
    random.shuffle(chunks)
    return "\n\n---\n\n".join(d.page_content for d in chunks[:sample_size])


def generate_quiz(num_questions: int, index_path: str = VECTORSTORE_PATH) -> List[QuizQuestion]:
    """Generate `num_questions` unique MCQs grounded in the indexed rules PDF."""
    if not 1 <= num_questions <= MAX_QUIZ_QUESTIONS:
        raise ValueError(f"num_questions must be between 1 and {MAX_QUIZ_QUESTIONS}.")

    llm = get_llm(temperature=0.7)
    parser = PydanticOutputParser(pydantic_object=QuizSet)
    context = _load_context_chunks(index_path)
    examples = load_quiz_examples()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUIZ_SYSTEM_PROMPT),
            (
                "human",
                "Context from the T20 rules document:\n{context}\n\n"
                "Example question format (style reference only, do not repeat "
                "verbatim):\n{examples}\n\n"
                "Generate exactly {n} unique multiple-choice questions strictly "
                "based on the context above.\n{format_instructions}",
            ),
        ]
    )

    chain = prompt | llm | parser
    result: QuizSet = chain.invoke(
        {
            "context": context,
            "examples": examples,
            "n": num_questions,
            "format_instructions": parser.get_format_instructions(),
        }
    )
    return result.questions[:num_questions]
