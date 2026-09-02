"""Loads the few-shot example files used to steer answer/quiz formatting."""
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "knowledgebase"


def load_chat_examples() -> str:
    """Example Q&A pairs illustrating the desired chatbot answer style."""
    path = KB_DIR / "chat_prompts.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_quiz_examples() -> str:
    """Example MCQ format illustrating the desired quiz question style."""
    path = KB_DIR / "quiz_prompts.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""
