"""T20 AI Assistant - Streamlit app with Chatbot and Take Quiz features."""
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import build_chat_agent, extract_text
from src.config import VECTORSTORE_PATH
from src.quiz import QUIZ_LENGTH_OPTIONS, generate_quiz

st.set_page_config(page_title="T20 AI Assistant", page_icon="🏏", layout="centered")

INDEX_READY = Path(VECTORSTORE_PATH).exists()

st.sidebar.title("🏏 T20 AI Assistant")
page = st.sidebar.radio("Choose a feature", ["Chatbot", "Take Quiz"])


def start_new_conversation() -> None:
    """Reset chatbot history and any in-progress/completed quiz state."""
    st.session_state.chat_messages = []
    st.session_state.chat_history = []
    st.session_state.quiz_questions = None
    st.session_state.quiz_submitted = False
    for key in [k for k in st.session_state if k.startswith("quiz_q_")]:
        del st.session_state[key]


if st.sidebar.button("🆕 New Conversation", use_container_width=True):
    start_new_conversation()
    st.rerun()

if not INDEX_READY:
    st.sidebar.warning(
        "No vector index found. Place your T20 rules PDF in `knowledgebase/docs/`, "
        "set your API keys in `.env`, then run `python -m src.ingestion`."
    )


@st.cache_resource(show_spinner="Warming up the T20 assistant...")
def get_agent():
    return build_chat_agent()


def render_chatbot() -> None:
    st.header("Chat with the T20 Assistant")
    st.caption(
        "Ask about official rules (answered via semantic search over the indexed "
        "rulebook) or recent news (answered via live web search) - the agent "
        "decides which to use automatically."
    )

    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("chat_history", [])

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about T20 cricket rules or news...")
    if not question:
        return

    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = get_agent()
                result = agent.invoke(
                    {"input": question, "chat_history": st.session_state.chat_history}
                )
                answer = extract_text(result["output"])
            except Exception as exc:  # surface config/runtime errors in the UI
                answer = f"Sorry, something went wrong: {exc}"
        st.markdown(answer)

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))


def render_quiz() -> None:
    st.header("Take a T20 Rules Quiz")
    st.caption("Pick how many questions you want, then test your T20 rules knowledge.")

    num_questions = st.selectbox("Number of questions", QUIZ_LENGTH_OPTIONS, index=1)

    if st.button("Generate Quiz"):
        with st.spinner("Generating quiz questions..."):
            try:
                st.session_state.quiz_questions = generate_quiz(num_questions)
                st.session_state.quiz_submitted = False
            except Exception as exc:
                st.error(f"Could not generate quiz: {exc}")
                st.session_state.quiz_questions = None

    questions = st.session_state.get("quiz_questions")
    if not questions:
        return

    with st.form("quiz_form"):
        for i, q in enumerate(questions):
            st.radio(f"{i + 1}. {q.question}", q.options, key=f"quiz_q_{i}", index=None)
        submitted = st.form_submit_button("Submit Answers")

    if submitted:
        st.session_state.quiz_submitted = True

    if st.session_state.get("quiz_submitted"):
        score = 0
        for i, q in enumerate(questions):
            selected = st.session_state.get(f"quiz_q_{i}")
            correct = selected == q.answer
            score += int(correct)
            icon = "✅" if correct else "❌"
            st.markdown(f"{icon} **Q{i + 1}. {q.question}**")
            st.markdown(f"- Your answer: {selected if selected else '_no answer_'}")
            if not correct:
                st.markdown(f"- Correct answer: {q.answer}")
        st.success(f"Score: {score} / {len(questions)}")


if page == "Chatbot":
    render_chatbot()
else:
    render_quiz()
