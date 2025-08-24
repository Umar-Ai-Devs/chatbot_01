import os
import json
import time 
import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# ----------------------------
# Safe dotenv loading
# ----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("⚠️ python-dotenv not installed. Environment variables may not load.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="Groq Chatbot with Memory", page_icon="💬")
st.title("💬 Groq Chatbot with Memory")

# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.subheader("Controls")
    model_name = st.selectbox(
        "Groq Model",
        ["deepseek-r1-distill-llama-70b", "gemma2-9b-it", "llama-3.1-8b-instant"],
        index=2
    )

    temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 300, 150)

    system_prompt = st.text_area(
        "System prompt (Rules)",
        value="You are a helpful concise teaching assistant. Use short, clear explanations."
    )
    st.caption("Tip: Lower temperature for factual tasks; raise for brainstorming")

    if st.button("🧹 Clear Chat"):
        st.session_state.pop("history", None)
        st.session_state.pop("memory", None)
        st.rerun()

if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY. Add it in .env or system environment variables.")

# ----------------------------
# Memory & History
# ----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Chat LLM Setup
# ----------------------------
llm = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    api_key=GROQ_API_KEY
)

conv = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

# ----------------------------
# Chat UI
# ----------------------------
# show history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# new input
if user_input := st.chat_input("Type your message..."):
    # user msg
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # bot reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conv.run(user_input)
            st.markdown(response)

    st.session_state.history.append({"role": "assistant", "content": response})

# ----------------------------
# Download Chat (below chat UI)
# ----------------------------
if st.session_state.history:
    chat_json = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
    st.download_button(
        label="⬇️ Download Chat (JSON)",
        data=chat_json,
        file_name="chat_history.json",
        mime="application/json"
    )
