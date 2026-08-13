from pathlib import Path

import streamlit as st

from src.context.huggingface_token_counter import HuggingFaceTokenCounter
from src.conversation.context import ContextManager
from src.conversation.manager import ConversationManager
from src.llm.local_llm import LocalLLM


MODEL_PATH = (
    Path(r"D:\HuggingFaceCache\hub")
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / "7ae557604adf67be50417f59c2c2f167def9a775"
)


@st.cache_resource
def create_conversation() -> ConversationManager:
    llm = LocalLLM(MODEL_PATH)

    token_counter = HuggingFaceTokenCounter(MODEL_PATH)

    context_manager = ContextManager(
        token_counter=token_counter,
        max_tokens=512,
    )

    return ConversationManager(
        llm=llm,
        context_manager=context_manager,
    )


def initialize_session() -> None:
    if "conversation" not in st.session_state:
        st.session_state.conversation = create_conversation()


def render_history(conversation: ConversationManager) -> None:
    for message in conversation.get_history():
        with st.chat_message(message["role"]):
            st.write(message["content"])


def main() -> None:
    st.set_page_config(
        page_title="Local LLM Conversation Memory",
        page_icon="🧠",
    )

    st.title("🧠 Local LLM Conversation Memory")

    st.caption(
        "Local Qwen2.5-0.5B-Instruct • Token-aware context • "
        "Persistent conversation memory"
    )

    initialize_session()

    conversation: ConversationManager = st.session_state.conversation

    with st.sidebar:
        st.header("Conversation")

        if st.button("Reset conversation"):
            conversation.reset()
            st.rerun()

        st.divider()

        st.subheader("Stored Memory")

        memories = conversation.get_memories()

        if memories:
            for key, value in memories.items():
                st.write(f"**{key}:** {value}")
        else:
            st.caption("No memories stored yet.")

        st.divider()

        context = conversation.get_context()

        st.write(f"History messages: {len(conversation.get_history())}")
        st.write(f"Context messages: {len(context)}")

    render_history(conversation)

    user_input = st.chat_input("Type your message...")

    if user_input:
        conversation.add_user_message(user_input)

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = conversation.generate_response()

            st.write(response)


if __name__ == "__main__":
    main()
